"""Realtime conditional MCTS planner for multi-target tracking.

This version consumes standardized PlannerInput.

Planner interface:
    choose_track(planner_input, rng) -> int

Realtime conditional interface:
    start_conditional_planning(planner_input, rng, current_action) -> None
    plan_during_execution(planner_input, rng, planning_seconds) -> None
    finish_conditional_planning(outcome, planner_input, rng) -> int | None

Key design goals
----------------
1. Every planner receives the same valid action list through PlannerInput.
2. MCTS does not privately invent a different action set.
3. Rollout/value scoring is aligned with final evaluation:
      uncertainty cost
      lost-target penalty
      optional travel-distance penalty
      optional detection reward
4. Repeated easy detections are discouraged through marginal value, not a
   hard-coded cooldown.
5. Tracks approaching the lost threshold receive higher priority through a
   principled loss-risk multiplier.

This planner uses a State -> Action -> Outcome -> State tree:

    state
      action: pursue track i
        outcome: find
          next state
        outcome: miss
          next state

The low-level spiral search is approximated by a coverage estimator inside
MCTS. The real simulator still executes the actual pursuit/search behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from core.planners.action_space import validate_action_or_raise
from core.planners.planner_state import PlannerInput
from core.sim.drone import Drone
from core.sim.tracks import Track, constant_velocity_F, constant_velocity_Q


Outcome = Literal["find", "miss"]


# ---------------------------------------------------------------------------
# Lightweight belief/state copies used only inside MCTS
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BeliefState:
    """Copy of one track belief used inside the planner."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    existence_probability: float = 1.0
    time_since_seen: float = 0.0

    @classmethod
    def from_track(cls, track: Track) -> "BeliefState":
        return cls(
            track_id=int(track.track_id),
            mean=track.mean.copy(),
            covariance=track.covariance.copy(),
            existence_probability=float(track.existence_probability),
            time_since_seen=float(track.time_since_seen),
        )

    @property
    def position(self) -> np.ndarray:
        return self.mean[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[2:4]

    @property
    def position_covariance(self) -> np.ndarray:
        return self.covariance[:2, :2]

    @property
    def position_trace(self) -> float:
        return float(np.trace(self.position_covariance))

    @property
    def position_logdet(self) -> float:
        det = max(float(np.linalg.det(self.position_covariance)), 1e-12)
        return float(np.log(det))

    def copy(self) -> "BeliefState":
        return BeliefState(
            track_id=int(self.track_id),
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            existence_probability=float(self.existence_probability),
            time_since_seen=float(self.time_since_seen),
        )

    def predict(self, dt: float, acceleration_noise_std: float) -> None:
        """Kalman-style prediction: mean/covariance grow forward in time."""

        if dt <= 0.0:
            return

        F = constant_velocity_F(dt)
        Q = constant_velocity_Q(dt, acceleration_noise_std)

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.time_since_seen += dt

    def kalman_position_update_at_mean(self, measurement_noise_std: float) -> None:
        """Shrink covariance as if the target was detected at predicted mean.

        For planning, the exact measurement innovation is less important than
        the covariance reduction caused by receiving a position measurement.
        Using z = H @ mean gives zero innovation but the correct covariance
        shrinkage.
        """

        if measurement_noise_std <= 0.0:
            raise ValueError("measurement_noise_std must be positive.")

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        R = measurement_noise_std**2 * np.eye(2)
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        I = np.eye(4)

        self.covariance = (
            (I - K @ H)
            @ self.covariance
            @ (I - K @ H).T
            + K @ R @ K.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        self.time_since_seen = 0.0
        self.existence_probability = 1.0


@dataclass(slots=True)
class PlanningState:
    """Payload stored in MCTS state nodes."""

    drone_position: np.ndarray
    remaining_budget: float
    beliefs: dict[int, BeliefState]
    available_actions: tuple[int, ...]
    depth: int = 0

    # Imagined rollout bookkeeping.
    distance_traveled: float = 0.0
    detections: int = 0

    # Number of tracks already lost in the real simulator before this MCTS
    # branch begins. The root beliefs only contain currently valid/active tracks,
    # so this prevents MCTS from forgetting previous losses.
    lost_count_initial: int = 0

    def copy(self) -> "PlanningState":
        return PlanningState(
            drone_position=self.drone_position.copy(),
            remaining_budget=float(self.remaining_budget),
            beliefs={int(k): v.copy() for k, v in self.beliefs.items()},
            available_actions=tuple(int(a) for a in self.available_actions),
            depth=int(self.depth),
            distance_traveled=float(self.distance_traveled),
            detections=int(self.detections),
            lost_count_initial=int(self.lost_count_initial),
        )


@dataclass(slots=True)
class CoverageEstimate:
    """Estimated low-level coverage outcome for one target pursuit."""

    p_find: float
    expected_find_time: float
    miss_time: float


# ---------------------------------------------------------------------------
# Tree nodes: State -> Action -> Outcome -> State
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StateNode:
    state: PlanningState
    parent_action: Optional["ActionNode"] = None
    outcome_from_parent: Optional[Outcome] = None

    action_nodes: dict[int, "ActionNode"] = field(default_factory=dict)
    unexpanded_actions: list[int] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.unexpanded_actions:
            self.unexpanded_actions = list(self.state.available_actions)

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass(slots=True)
class ActionNode:
    parent_state: StateNode
    action: int
    p_find: float
    expected_find_time: float
    miss_time: float

    outcome_states: dict[Outcome, StateNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MCTSPlanner:
    """Realtime conditional MCTS target-selection planner."""

    # Initial/blocking call budget used when choose_track is called from scratch.
    iterations: int = 750

    # Background compute model.
    iterations_per_second: float = 80.0
    min_background_iterations_per_call: int = 1
    max_background_iterations_per_call: int = 250

    # Tree search settings.
    max_depth: int = 5
    exploration_weight: float = 1.4

    # Internal model parameters. Keep these roughly aligned with SimConfig.
    max_search_time: float = 65.0
    acceleration_noise_std: float = 0.03
    measurement_noise_std: float = 20.0
    covariance_scale_for_detection: float = 3.0

    # Objective settings.
    use_logdet_objective: bool = False
    lost_trace_threshold: float = 250_000.0

    # Make losing targets very expensive inside MCTS. This should be at least as
    # large as the simulator's lost_target_penalty.
    lost_target_penalty: float = 1_000_000.0

    # Shared-objective-style scoring weights.
    uncertainty_weight: float = 1.0
    travel_distance_weight: float = 0.0
    detection_reward: float = 0.0

    # Heuristic shaping.
    # These do not force rotation. They estimate marginal usefulness.
    distance_bias_seconds: float = 30.0
    loss_risk_max_multiplier: float = 50.0
    measurement_trace_floor_scale: float = 2.0

    # Terminal evaluation.
    #
    # If max_depth is small and remaining_budget is large, predicting all the way
    # to mission end can make many branches collapse into identical catastrophic
    # values. Keep 0.0 while debugging. Later, try 30.0 or 65.0.
    terminal_tail_time: float = 0.0

    # Tie-breakers for saturated catastrophic branches.
    # These prevent every "all targets lost" rollout from receiving exactly
    # the same value.
    lost_trace_cost_weight: float = 1.0
    active_loss_risk_weight: float = 50_000.0

    # Rollout policy settings.
    rollout_random_action_probability: float = 0.10

    # If True, a miss removes that target from the imagined branch's action set.
    # Keep False if the real simulator uses permanent loss thresholds instead of
    # immediate removal on miss.
    remove_missed_target_from_branch: bool = False

    # "value" tends to work better during debugging because it directly chooses
    # the action with best estimated objective. "visits" is the classic robust
    # MCTS choice once the model is well tuned.
    root_selection: Literal["visits", "value"] = "value"

    # Realtime conditional planning cache.
    current_action: int | None = None
    cached_recommendations: dict[str, int | None] = field(default_factory=dict)
    cached_values: dict[str, float] = field(default_factory=dict)
    total_background_iterations: int = 0
    last_background_iterations: int = 0
    background_planning_calls: int = 0

    # ------------------------------------------------------------------
    # Public planner APIs
    # ------------------------------------------------------------------

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        """Choose an action from scratch."""

        valid_actions = planner_input.require_valid_actions("MCTSPlanner")
        candidate_tracks = self._candidate_tracks_from_planner_input(planner_input)

        root_state = self._make_root_state(
            tracks=candidate_tracks,
            drone=planner_input.drone,
            valid_action_ids=valid_actions,
            lost_count_initial=planner_input.lost_count,
        )
        root = StateNode(state=root_state)

        for _ in range(max(1, int(self.iterations))):
            leaf = self._tree_policy(root, planner_input.drone, rng)
            value = self._rollout(leaf.state.copy(), planner_input.drone, rng)
            self._backup_state_path(leaf, value)

        if not root.action_nodes:
            chosen = int(rng.choice(valid_actions))
        else:
            chosen = int(self._best_action_node(root).action)

        return validate_action_or_raise(
            chosen,
            valid_actions,
            planner_name="MCTSPlanner",
        )

    def start_conditional_planning(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
        current_action: int,
    ) -> None:
        """Start planning next actions while the current action is executing."""

        valid_actions = planner_input.require_valid_actions("MCTSPlanner")

        self.current_action = validate_action_or_raise(
            int(current_action),
            valid_actions,
            planner_name="MCTSPlanner.start_conditional_planning",
        )
        self.cached_recommendations = {"find": None, "miss": None}
        self.cached_values = {"find": -float("inf"), "miss": -float("inf")}
        self.last_background_iterations = 0

    def plan_during_execution(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
        planning_seconds: float,
    ) -> None:
        """Run conditional MCTS for the real execution time available."""

        if self.current_action is None or planning_seconds <= 0.0:
            self.last_background_iterations = 0
            return

        valid_actions = planner_input.valid_action_ids

        if self.current_action not in valid_actions:
            self.cached_recommendations["find"] = None
            self.cached_recommendations["miss"] = None
            self.cached_values["find"] = -float("inf")
            self.cached_values["miss"] = -float("inf")
            self.last_background_iterations = 0
            return

        iterations = int(round(self.iterations_per_second * planning_seconds))
        iterations = max(self.min_background_iterations_per_call, iterations)
        iterations = min(self.max_background_iterations_per_call, iterations)

        candidate_tracks = self._candidate_tracks_from_planner_input(planner_input)

        recommendations, values = self._conditional_plan_fixed_action(
            tracks=candidate_tracks,
            drone=planner_input.drone,
            rng=rng,
            fixed_action=int(self.current_action),
            valid_action_ids=valid_actions,
            iterations=iterations,
            lost_count_initial=planner_input.lost_count,
        )

        for outcome in ("find", "miss"):
            rec = recommendations.get(outcome)
            val = values.get(outcome, -float("inf"))

            if rec is not None and int(rec) in valid_actions:
                self.cached_recommendations[outcome] = int(rec)
                self.cached_values[outcome] = float(val)

        self.last_background_iterations = iterations
        self.total_background_iterations += iterations
        self.background_planning_calls += 1

    def finish_conditional_planning(
        self,
        outcome: str,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int | None:
        """Return the cached next action for the realized outcome."""

        normalized = "find" if outcome == "find" else "miss"
        rec = self.cached_recommendations.get(normalized)

        self.current_action = None

        if rec is not None and int(rec) in planner_input.valid_action_ids:
            return int(rec)

        if not planner_input.valid_action_ids:
            return None

        return int(self.choose_track(planner_input=planner_input, rng=rng))

    def diagnostics(self) -> dict:
        return {
            "mcts_current_action": self.current_action,
            "mcts_cached_find": self.cached_recommendations.get("find"),
            "mcts_cached_miss": self.cached_recommendations.get("miss"),
            "mcts_cached_find_value": self.cached_values.get("find"),
            "mcts_cached_miss_value": self.cached_values.get("miss"),
            "mcts_last_background_iterations": self.last_background_iterations,
            "mcts_total_background_iterations": self.total_background_iterations,
            "mcts_background_planning_calls": self.background_planning_calls,
            "mcts_iterations_per_second": float(self.iterations_per_second),
            "mcts_lost_target_penalty": float(self.lost_target_penalty),
            "mcts_detection_reward": float(self.detection_reward),
            "mcts_travel_distance_weight": float(self.travel_distance_weight),
            "mcts_terminal_tail_time": float(self.terminal_tail_time),
            "mcts_loss_risk_max_multiplier": float(self.loss_risk_max_multiplier),
            "mcts_measurement_trace_floor_scale": float(self.measurement_trace_floor_scale),
            "mcts_lost_trace_cost_weight": float(self.lost_trace_cost_weight),
            "mcts_active_loss_risk_weight": float(self.active_loss_risk_weight),
        }

    # ------------------------------------------------------------------
    # Conditional fixed-current-action planning
    # ------------------------------------------------------------------

    def _conditional_plan_fixed_action(
        self,
        tracks: list[Track],
        drone: Drone,
        rng: np.random.Generator,
        fixed_action: int,
        valid_action_ids: tuple[int, ...],
        iterations: int,
        lost_count_initial: int,
    ) -> tuple[dict[str, int | None], dict[str, float]]:
        root_state = self._make_root_state(
            tracks=tracks,
            drone=drone,
            valid_action_ids=valid_action_ids,
            lost_count_initial=lost_count_initial,
        )

        if fixed_action not in root_state.available_actions:
            return (
                {"find": None, "miss": None},
                {"find": -float("inf"), "miss": -float("inf")},
            )

        root = StateNode(state=root_state)
        estimate = self._estimate_coverage_outcome(root_state, fixed_action, drone)

        fixed_action_node = ActionNode(
            parent_state=root,
            action=int(fixed_action),
            p_find=estimate.p_find,
            expected_find_time=estimate.expected_find_time,
            miss_time=estimate.miss_time,
        )
        root.action_nodes[int(fixed_action)] = fixed_action_node
        root.unexpanded_actions = [
            int(a) for a in root.unexpanded_actions if int(a) != int(fixed_action)
        ]

        # Create both outcome states so we can extract both recommendations.
        for outcome in ("find", "miss"):
            next_state = self._transition(
                root_state,
                int(fixed_action),
                outcome,
                estimate,
                drone,
            )
            child = StateNode(
                state=next_state,
                parent_action=fixed_action_node,
                outcome_from_parent=outcome,
            )
            fixed_action_node.outcome_states[outcome] = child

        for _ in range(max(1, int(iterations))):
            outcome = self._sample_outcome(fixed_action_node, rng)
            outcome_node = fixed_action_node.outcome_states[outcome]

            leaf = self._tree_policy(outcome_node, drone, rng)
            value = self._rollout(leaf.state.copy(), drone, rng)
            self._backup_state_path(leaf, value)

        recommendations: dict[str, int | None] = {}
        values: dict[str, float] = {}

        for outcome in ("find", "miss"):
            outcome_node = fixed_action_node.outcome_states[outcome]
            best = (
                self._best_action_node(outcome_node)
                if outcome_node.action_nodes
                else None
            )
            recommendations[outcome] = None if best is None else int(best.action)
            values[outcome] = -float("inf") if best is None else float(best.mean_value)

        return recommendations, values

    # ------------------------------------------------------------------
    # Standard MCTS
    # ------------------------------------------------------------------

    def _tree_policy(
        self,
        node: StateNode,
        drone: Drone,
        rng: np.random.Generator,
    ) -> StateNode:
        while not self._is_terminal(node.state):
            if node.unexpanded_actions:
                return self._expand_state_node(node, drone, rng)

            action_node = self._select_action_ucb(node)
            outcome = self._sample_outcome(action_node, rng)

            if outcome not in action_node.outcome_states:
                estimate = CoverageEstimate(
                    p_find=action_node.p_find,
                    expected_find_time=action_node.expected_find_time,
                    miss_time=action_node.miss_time,
                )
                next_state = self._transition(
                    node.state,
                    action_node.action,
                    outcome,
                    estimate,
                    drone,
                )
                child = StateNode(
                    state=next_state,
                    parent_action=action_node,
                    outcome_from_parent=outcome,
                )
                action_node.outcome_states[outcome] = child
                return child

            node = action_node.outcome_states[outcome]

        return node

    def _expand_state_node(
        self,
        node: StateNode,
        drone: Drone,
        rng: np.random.Generator,
    ) -> StateNode:
        idx = int(rng.integers(0, len(node.unexpanded_actions)))
        action = int(node.unexpanded_actions.pop(idx))

        estimate = self._estimate_coverage_outcome(node.state, action, drone)

        action_node = ActionNode(
            parent_state=node,
            action=action,
            p_find=estimate.p_find,
            expected_find_time=estimate.expected_find_time,
            miss_time=estimate.miss_time,
        )
        node.action_nodes[action] = action_node

        outcome = self._sample_outcome(action_node, rng)
        next_state = self._transition(node.state, action, outcome, estimate, drone)

        child = StateNode(
            state=next_state,
            parent_action=action_node,
            outcome_from_parent=outcome,
        )
        action_node.outcome_states[outcome] = child

        return child

    def _select_action_ucb(self, node: StateNode) -> ActionNode:
        def score(action_node: ActionNode) -> float:
            if action_node.visits == 0:
                return float("inf")

            exploit = action_node.mean_value
            explore = self.exploration_weight * math.sqrt(
                math.log(max(1, node.visits)) / action_node.visits
            )
            return exploit + explore

        return max(node.action_nodes.values(), key=score)

    @staticmethod
    def _sample_outcome(action_node: ActionNode, rng: np.random.Generator) -> Outcome:
        return "find" if rng.random() <= action_node.p_find else "miss"

    def _backup_state_path(self, leaf: StateNode, value: float) -> None:
        node: Optional[StateNode] = leaf

        while node is not None:
            node.visits += 1
            node.value_sum += value

            action = node.parent_action
            if action is None:
                break

            action.visits += 1
            action.value_sum += value
            node = action.parent_state

    # ------------------------------------------------------------------
    # Rollout and value
    # ------------------------------------------------------------------

    def _rollout(
        self,
        state: PlanningState,
        drone: Drone,
        rng: np.random.Generator,
    ) -> float:
        while not self._is_terminal(state):
            action = self._rollout_policy(state, drone, rng)
            estimate = self._estimate_coverage_outcome(state, action, drone)
            outcome = "find" if rng.random() <= estimate.p_find else "miss"
            state = self._transition(state, action, outcome, estimate, drone)

        return self._terminal_value(state)

    def _terminal_value(self, state: PlanningState) -> float:
        """Evaluate a rollout leaf.

        We intentionally do not always propagate to full budget depletion.

        If max_depth is small and remaining_budget is large, predicting all the
        way to mission end can make every branch collapse into the same
        catastrophic outcome. That destroys action ranking.

        Instead:
            - Evaluate the state reached by the rollout.
            - Optionally predict a short terminal tail if terminal_tail_time > 0.
        """

        terminal = state.copy()

        if self.terminal_tail_time > 0.0 and terminal.remaining_budget > 0.0:
            tail_dt = min(float(self.terminal_tail_time), terminal.remaining_budget)
            self._predict_all(terminal, tail_dt)
            terminal.remaining_budget = max(0.0, terminal.remaining_budget - tail_dt)

        return -self._state_cost(terminal)

    def _state_cost(self, state: PlanningState) -> float:
        """System-level cost for a planning state.

        Lower is better.

        Cost terms:
            active uncertainty
            lost target penalty
            lost-state severity tie-breaker
            active near-loss risk
            optional travel distance penalty
            optional detection bonus

        The lost-state severity term is important. Without it, many deep rollouts
        can collapse to exactly -N * lost_target_penalty, making MCTS unable to rank
        actions when several futures are bad.
        """

        active_uncertainty_cost = 0.0
        newly_lost_count = 0
        lost_severity_cost = 0.0
        active_risk_cost = 0.0

        for belief in state.beliefs.values():
            trace = float(belief.position_trace)

            if self._belief_is_lost(belief):
                newly_lost_count += 1

                # Tie-breaker: not all lost states are equally bad.
                # Cap the trace contribution so numerical scale stays sane.
                capped_trace = min(trace, 5.0 * self.lost_trace_threshold)
                lost_severity_cost += self.lost_trace_cost_weight * capped_trace
                continue

            if self.use_logdet_objective:
                active_uncertainty_cost += belief.position_logdet
            else:
                active_uncertainty_cost += trace

            # Soft risk term for active tracks approaching loss.
            # This starts mattering before the hard lost threshold is crossed.
            if self.lost_trace_threshold > 0.0:
                loss_fraction = trace / self.lost_trace_threshold

                if loss_fraction > 0.5:
                    normalized_risk = (loss_fraction - 0.5) / 0.5
                    active_risk_cost += (
                        self.active_loss_risk_weight
                        * float(normalized_risk**2)
                    )

        uncertainty_cost = self.uncertainty_weight * active_uncertainty_cost

        lost_target_cost = self.lost_target_penalty * (
            int(state.lost_count_initial) + int(newly_lost_count)
        )

        travel_cost = self.travel_distance_weight * state.distance_traveled
        detection_bonus = self.detection_reward * state.detections

        return float(
            uncertainty_cost
            + lost_target_cost
            + lost_severity_cost
            + active_risk_cost
            + travel_cost
            - detection_bonus
        )

    def _belief_is_lost(self, belief: BeliefState) -> bool:
        return belief.position_trace >= self.lost_trace_threshold

    def _rollout_policy(
        self,
        state: PlanningState,
        drone: Drone,
        rng: np.random.Generator,
    ) -> int:
        """Choose a rollout action.

        This does not use a hard cooldown.

        Repeated revisits are discouraged naturally because a recently detected
        target has little marginal uncertainty left to reduce.
        """

        actions = [
            int(action)
            for action in state.available_actions
            if int(action) in state.beliefs
            and not self._belief_is_lost(state.beliefs[int(action)])
        ]

        if not actions:
            raise RuntimeError("MCTS rollout policy received no available active actions.")

        if rng.random() < self.rollout_random_action_probability:
            return int(rng.choice(actions))

        return int(
            max(
                actions,
                key=lambda action: self._heuristic_action_score(state, action, drone),
            )
        )

    def _heuristic_action_score(
        self,
        state: PlanningState,
        action: int,
        drone: Drone,
    ) -> float:
        """Heuristic score for choosing an action inside rollouts/tree expansion.

        Higher is better.

        Principle:
            Score expected marginal value, not raw uncertainty.

        This avoids repeatedly choosing an easy target that was just detected.
        If its covariance is already low, another detection has little value.
        """

        belief = state.beliefs[int(action)]

        if self._belief_is_lost(belief):
            return -float("inf")

        marginal_uncertainty_value = self._expected_detection_value(belief)

        if marginal_uncertainty_value <= 0.0:
            marginal_uncertainty_value = 1e-6

        travel_time = self._travel_time(
            state.drone_position,
            belief.position,
            drone.speed,
        )

        loss_risk_multiplier = self._loss_risk_multiplier(belief)

        # This is a soft staleness bonus, not a rule. Marginal uncertainty is
        # still the main value term.
        stale_bonus = 1.0 + min(2.0, belief.time_since_seen / 120.0)

        score = (
            stale_bonus
            * loss_risk_multiplier
            * marginal_uncertainty_value
            / (travel_time + self.distance_bias_seconds)
        )

        return float(score)

    def _expected_detection_value(self, belief: BeliefState) -> float:
        """Approximate marginal value of detecting this track.

        A recently detected track may have very low covariance. Detecting it
        again should not be considered very valuable. This naturally discourages
        camping without banning revisits.
        """

        if self.use_logdet_objective:
            # Most debugging should use trace mode. Keep logdet mode simple.
            return float(max(0.0, belief.position_logdet))

        # Approximate post-detection trace floor. For a 2D position measurement
        # with measurement_noise_std, the best reasonable position trace is on
        # the order of 2 * R.
        measurement_trace_floor = (
            self.measurement_trace_floor_scale
            * (self.measurement_noise_std ** 2)
        )

        return float(max(0.0, belief.position_trace - measurement_trace_floor))

    def _loss_risk_multiplier(self, belief: BeliefState) -> float:
        """Increase priority for tracks approaching permanent loss.

        This is not a target-rotation rule. It encodes the actual mission
        objective: crossing the lost threshold is catastrophic.
        """

        if self.lost_trace_threshold <= 0.0:
            return 1.0

        loss_fraction = belief.position_trace / self.lost_trace_threshold

        # Below half the threshold, do not distort the heuristic.
        if loss_fraction <= 0.5:
            return 1.0

        multiplier = 1.0 / max(1e-3, 1.0 - loss_fraction)

        return float(
            np.clip(
                multiplier,
                1.0,
                self.loss_risk_max_multiplier,
            )
        )

    # ------------------------------------------------------------------
    # Transition and coverage estimation
    # ------------------------------------------------------------------

    def _transition(
        self,
        state: PlanningState,
        action: int,
        outcome: Outcome,
        estimate: CoverageEstimate,
        drone: Drone,
    ) -> PlanningState:
        next_state = state.copy()

        if int(action) not in next_state.beliefs:
            next_state.available_actions = tuple(
                a for a in next_state.available_actions if int(a) != int(action)
            )
            return next_state

        selected_before_prediction = next_state.beliefs[int(action)]

        travel_distance = self._travel_distance(
            next_state.drone_position,
            selected_before_prediction.position,
        )
        travel_time = travel_distance / drone.speed

        search_time = (
            estimate.expected_find_time
            if outcome == "find"
            else estimate.miss_time
        )

        elapsed = min(next_state.remaining_budget, travel_time + search_time)

        # Everyone's covariance grows while the drone travels/searches.
        self._predict_all(next_state, elapsed)

        selected = next_state.beliefs[int(action)]
        found = outcome == "find" and elapsed > travel_time

        if found:
            selected.kalman_position_update_at_mean(self.measurement_noise_std)
            next_state.detections += 1

        new_position = self._position_after_action(
            start=state.drone_position,
            goal=selected.position,
            speed=drone.speed,
            elapsed=elapsed,
        )

        actual_distance = self._travel_distance(state.drone_position, new_position)

        next_state.drone_position = new_position
        next_state.distance_traveled += actual_distance
        next_state.remaining_budget = max(0.0, next_state.remaining_budget - elapsed)
        next_state.depth += 1

        active_actions = tuple(
            int(track_id)
            for track_id, belief in next_state.beliefs.items()
            if not self._belief_is_lost(belief)
        )

        if outcome == "miss" and self.remove_missed_target_from_branch:
            active_actions = tuple(a for a in active_actions if int(a) != int(action))

        next_state.available_actions = active_actions
        return next_state

    def _estimate_coverage_outcome(
        self,
        state: PlanningState,
        action: int,
        drone: Drone,
    ) -> CoverageEstimate:
        """Estimate P(find), E[T_find], and T_miss for pursuing a target.

        Approximation:
            covered area ~= sensor_width * drone_speed * t + initial footprint
            normalized area ~= covered_area / effective_covariance_area
            P(find by t) ~= 1 - exp(-normalized / 2)
        """

        belief = state.beliefs[int(action)].copy()

        travel_time = self._travel_time(
            state.drone_position,
            belief.position,
            drone.speed,
        )

        remaining_after_travel = max(0.0, state.remaining_budget - travel_time)
        miss_time = min(self.max_search_time, remaining_after_travel)

        if miss_time <= 0.0:
            return CoverageEstimate(
                p_find=0.0,
                expected_find_time=0.0,
                miss_time=0.0,
            )

        # Belief grows while we travel to the target's current predicted center.
        belief.predict(travel_time, self.acceleration_noise_std)

        num_steps = max(8, int(math.ceil(miss_time / 1.0)))
        times = np.linspace(0.0, miss_time, num_steps + 1)
        cdf = np.zeros_like(times)

        for i, t in enumerate(times):
            b = belief.copy()
            b.predict(float(t), self.acceleration_noise_std)
            cdf[i] = self._coverage_cdf(b, drone, float(t))

        cdf = np.maximum.accumulate(cdf)
        p_find = float(np.clip(cdf[-1] * belief.existence_probability, 0.0, 1.0))

        if p_find <= 1e-12:
            expected_find_time = miss_time
        else:
            increments = np.diff(cdf, prepend=0.0)
            increments = np.maximum(increments, 0.0)

            if increments.sum() <= 1e-12:
                expected_find_time = miss_time
            else:
                expected_find_time = float(
                    np.sum(times * increments) / increments.sum()
                )
                expected_find_time = float(
                    np.clip(expected_find_time, 0.0, miss_time)
                )

        return CoverageEstimate(
            p_find=p_find,
            expected_find_time=expected_find_time,
            miss_time=float(miss_time),
        )

    def _coverage_cdf(
        self,
        belief: BeliefState,
        drone: Drone,
        search_time: float,
    ) -> float:
        sensor_width = 2.0 * drone.sensor_range
        initial_area = math.pi * drone.sensor_range**2
        covered_area = initial_area + sensor_width * drone.speed * max(0.0, search_time)

        det = max(float(np.linalg.det(belief.position_covariance)), 1e-12)
        effective_area = (
            math.pi
            * math.sqrt(det)
            * (self.covariance_scale_for_detection**2)
        )

        normalized = covered_area / max(effective_area, 1e-12)

        return float(np.clip(1.0 - math.exp(-0.5 * normalized), 0.0, 1.0))

    def _predict_all(self, state: PlanningState, dt: float) -> None:
        for belief in state.beliefs.values():
            if self._belief_is_lost(belief):
                continue
            belief.predict(dt, self.acceleration_noise_std)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_root_state(
        self,
        tracks: list[Track],
        drone: Drone,
        valid_action_ids: tuple[int, ...],
        lost_count_initial: int,
    ) -> PlanningState:
        valid_set = set(int(a) for a in valid_action_ids)

        beliefs = {
            int(track.track_id): BeliefState.from_track(track)
            for track in tracks
            if int(track.track_id) in valid_set
        }

        active_actions = tuple(
            int(track_id)
            for track_id in valid_action_ids
            if int(track_id) in beliefs
            and not self._belief_is_lost(beliefs[int(track_id)])
        )

        return PlanningState(
            drone_position=drone.position.copy(),
            remaining_budget=float(drone.remaining_budget),
            beliefs=beliefs,
            available_actions=active_actions,
            depth=0,
            distance_traveled=0.0,
            detections=0,
            lost_count_initial=int(lost_count_initial),
        )

    def _is_terminal(self, state: PlanningState) -> bool:
        return (
            state.remaining_budget <= 0.0
            or state.depth >= self.max_depth
            or not state.available_actions
        )

    def _best_action_node(self, state_node: StateNode) -> ActionNode:
        if self.root_selection == "visits":
            return max(state_node.action_nodes.values(), key=lambda node: node.visits)

        return max(state_node.action_nodes.values(), key=lambda node: node.mean_value)

    @staticmethod
    def _candidate_tracks_from_planner_input(
        planner_input: PlannerInput,
    ) -> list[Track]:
        valid_actions = set(int(a) for a in planner_input.valid_action_ids)

        return [
            track
            for track in planner_input.tracks.tracks
            if int(track.track_id) in valid_actions
        ]

    @staticmethod
    def _travel_distance(start: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(goal - start))

    @staticmethod
    def _travel_time(start: np.ndarray, goal: np.ndarray, speed: float) -> float:
        if speed <= 0.0:
            raise ValueError("drone speed must be positive.")

        return float(np.linalg.norm(goal - start) / speed)

    @staticmethod
    def _position_after_action(
        start: np.ndarray,
        goal: np.ndarray,
        speed: float,
        elapsed: float,
    ) -> np.ndarray:
        max_distance = speed * max(0.0, elapsed)
        delta = goal - start
        distance = float(np.linalg.norm(delta))

        if distance <= 1e-12:
            return goal.copy()

        if max_distance >= distance:
            return goal.copy()

        return start + delta / distance * max_distance