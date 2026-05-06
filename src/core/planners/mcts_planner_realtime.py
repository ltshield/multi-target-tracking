"""
Realtime conditional MCTS planner for multi-target tracking.

Plug-in interface expected by simulate_run.py:

    choose_track(tracks, drone, targets, rng) -> int

Realtime conditional interface used by the rewritten simulator:

    start_conditional_planning(tracks, drone, targets, rng, current_action)
    plan_during_execution(tracks, drone, targets, rng, planning_seconds)
    finish_conditional_planning(outcome, tracks, drone, targets, rng) -> int | None

The planner is intended to model the paper-style behavior:

1. The current pursuit action is fixed while the drone is executing it.
2. MCTS runs during that execution time.
3. The tree evaluates conditional branches:
      current action -> find -> next action
      current action -> miss -> next action
4. When the real outcome occurs, the simulator asks for the recommendation
   from the matching branch.

This implementation rebuilds a fixed-current-action tree from the latest live
belief at each planning call. That deliberately acts like repeated online
replanning from the current belief while the drone is in flight/search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from core.sim.drone import Drone
from core.sim.target import TargetSet
from core.sim.tracks import Track, TrackSet, constant_velocity_F, constant_velocity_Q


Outcome = Literal["find", "miss"]


@dataclass(slots=True)
class BeliefState:
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
            track_id=self.track_id,
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            existence_probability=self.existence_probability,
            time_since_seen=self.time_since_seen,
        )

    def predict(self, dt: float, acceleration_noise_std: float) -> None:
        if dt <= 0.0:
            return

        F = constant_velocity_F(dt)
        Q = constant_velocity_Q(dt, acceleration_noise_std)
        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.time_since_seen += dt

    def kalman_position_update_at_mean(self, measurement_noise_std: float) -> None:
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

        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.time_since_seen = 0.0
        self.existence_probability = 1.0


@dataclass(slots=True)
class PlanningState:
    drone_position: np.ndarray
    remaining_budget: float
    beliefs: dict[int, BeliefState]
    available_actions: tuple[int, ...]
    depth: int = 0

    def copy(self) -> "PlanningState":
        return PlanningState(
            drone_position=self.drone_position.copy(),
            remaining_budget=float(self.remaining_budget),
            beliefs={k: v.copy() for k, v in self.beliefs.items()},
            available_actions=tuple(self.available_actions),
            depth=int(self.depth),
        )


@dataclass(slots=True)
class CoverageEstimate:
    p_find: float
    expected_find_time: float
    miss_time: float


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


@dataclass(slots=True)
class MCTSPlanner:
    # Initial/blocking call budget used only if no conditional recommendation is ready.
    iterations: int = 750

    # Background compute model.
    iterations_per_second: float = 80.0
    min_background_iterations_per_call: int = 1
    max_background_iterations_per_call: int = 250

    max_depth: int = 5
    exploration_weight: float = 1.4

    # Internal model parameters. Keep these roughly aligned with SimConfig.
    max_search_time: float = 65.0
    acceleration_noise_std: float = 0.03
    measurement_noise_std: float = 20.0
    covariance_scale_for_detection: float = 3.0

    use_logdet_objective: bool = False
    lost_trace_threshold: float = 250_000.0
    lost_target_penalty: float = 1_000_000.0

    recent_revisit_penalty_window: float = 60.0
    rollout_random_action_probability: float = 0.10
    distance_bias_seconds: float = 30.0
    remove_missed_target_from_branch: bool = False
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
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        """Choose an action from scratch.

        This is used for the first action, for non-conditional fallback, or if
        the simulator asks the planner normally.
        """

        candidate_tracks = self._active_candidate_tracks(tracks)
        if not candidate_tracks:
            raise ValueError("MCTSPlanner cannot choose from an empty active TrackSet.")

        root_state = self._make_root_state(candidate_tracks, drone)
        root = StateNode(state=root_state)

        for _ in range(max(1, int(self.iterations))):
            leaf = self._tree_policy(root, drone, rng)
            value = self._rollout(leaf.state.copy(), drone, rng)
            self._backup_state_path(leaf, value)

        if not root.action_nodes:
            return int(rng.choice([track.track_id for track in candidate_tracks]))

        best = self._best_action_node(root)
        return int(best.action)

    def start_conditional_planning(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
        current_action: int,
    ) -> None:
        """Start planning next actions while the current action is executing."""

        self.current_action = int(current_action)
        self.cached_recommendations = {"find": None, "miss": None}
        self.cached_values = {"find": -float("inf"), "miss": -float("inf")}
        self.last_background_iterations = 0

    def plan_during_execution(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
        planning_seconds: float,
    ) -> None:
        """Run conditional MCTS for the amount of real execution time available."""

        if self.current_action is None or planning_seconds <= 0.0:
            self.last_background_iterations = 0
            return

        iterations = int(round(self.iterations_per_second * planning_seconds))
        iterations = max(self.min_background_iterations_per_call, iterations)
        iterations = min(self.max_background_iterations_per_call, iterations)

        candidate_tracks = self._active_candidate_tracks(tracks)
        candidate_ids = {track.track_id for track in candidate_tracks}

        if self.current_action not in candidate_ids:
            self.cached_recommendations["find"] = None
            self.cached_recommendations["miss"] = None
            self.last_background_iterations = 0
            return

        recommendations, values = self._conditional_plan_fixed_action(
            tracks=candidate_tracks,
            drone=drone,
            rng=rng,
            fixed_action=int(self.current_action),
            iterations=iterations,
        )

        for outcome in ("find", "miss"):
            rec = recommendations.get(outcome)
            val = values.get(outcome, -float("inf"))
            if rec is not None:
                self.cached_recommendations[outcome] = int(rec)
                self.cached_values[outcome] = float(val)

        self.last_background_iterations = iterations
        self.total_background_iterations += iterations
        self.background_planning_calls += 1

    def finish_conditional_planning(
        self,
        outcome: str,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int | None:
        """Return the cached next action for the realized outcome."""

        normalized = "find" if outcome == "find" else "miss"
        rec = self.cached_recommendations.get(normalized)

        self.current_action = None

        if rec is not None and self._track_id_is_active(tracks, int(rec)):
            return int(rec)

        # Fallback: if no useful background plan exists, choose from scratch.
        if len(tracks.tracks) == 0:
            return None
        return int(self.choose_track(tracks, drone, targets, rng))

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
            "mcts_iterations_per_second": self.iterations_per_second,
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
        iterations: int,
    ) -> tuple[dict[str, int | None], dict[str, float]]:
        root_state = self._make_root_state(tracks, drone)

        if fixed_action not in root_state.available_actions:
            return {"find": None, "miss": None}, {"find": -float("inf"), "miss": -float("inf")}

        root = StateNode(state=root_state)
        estimate = self._estimate_coverage_outcome(root_state, fixed_action, drone)

        fixed_action_node = ActionNode(
            parent_state=root,
            action=fixed_action,
            p_find=estimate.p_find,
            expected_find_time=estimate.expected_find_time,
            miss_time=estimate.miss_time,
        )
        root.action_nodes[fixed_action] = fixed_action_node
        root.unexpanded_actions = [
            a for a in root.unexpanded_actions if a != fixed_action
        ]

        # Ensure both conditional outcome states exist so we can extract both
        # next-action recommendations.
        for outcome in ("find", "miss"):
            next_state = self._transition(root_state, fixed_action, outcome, estimate, drone)
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
            best = self._best_action_node(outcome_node) if outcome_node.action_nodes else None
            recommendations[outcome] = None if best is None else int(best.action)
            values[outcome] = -float("inf") if best is None else float(best.mean_value)

        return recommendations, values

    # ------------------------------------------------------------------
    # Standard MCTS
    # ------------------------------------------------------------------

    def _tree_policy(self, node: StateNode, drone: Drone, rng: np.random.Generator) -> StateNode:
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
                next_state = self._transition(node.state, action_node.action, outcome, estimate, drone)
                child = StateNode(
                    state=next_state,
                    parent_action=action_node,
                    outcome_from_parent=outcome,
                )
                action_node.outcome_states[outcome] = child
                return child

            node = action_node.outcome_states[outcome]

        return node

    def _expand_state_node(self, node: StateNode, drone: Drone, rng: np.random.Generator) -> StateNode:
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
        child = StateNode(state=next_state, parent_action=action_node, outcome_from_parent=outcome)
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

    def _rollout(self, state: PlanningState, drone: Drone, rng: np.random.Generator) -> float:
        while not self._is_terminal(state):
            action = self._rollout_policy(state, drone, rng)
            estimate = self._estimate_coverage_outcome(state, action, drone)
            outcome = "find" if rng.random() <= estimate.p_find else "miss"
            state = self._transition(state, action, outcome, estimate, drone)

        return self._terminal_value(state)

    def _terminal_value(self, state: PlanningState) -> float:
        terminal = state.copy()
        if terminal.remaining_budget > 0.0:
            self._predict_all(terminal, terminal.remaining_budget)
            terminal.remaining_budget = 0.0
        return -self._state_cost(terminal)

    def _state_cost(self, state: PlanningState) -> float:
        cost = 0.0
        for belief in state.beliefs.values():
            if self._belief_is_lost(belief):
                cost += self.lost_target_penalty
            elif self.use_logdet_objective:
                cost += belief.position_logdet
            else:
                cost += belief.position_trace
        return float(cost)

    def _belief_is_lost(self, belief: BeliefState) -> bool:
        return belief.position_trace >= self.lost_trace_threshold

    def _rollout_policy(self, state: PlanningState, drone: Drone, rng: np.random.Generator) -> int:
        actions = [
            int(action)
            for action in state.available_actions
            if not self._belief_is_lost(state.beliefs[int(action)])
        ]

        if not actions:
            raise RuntimeError("Realtime MCTS rollout policy received no available active actions.")

        if rng.random() < self.rollout_random_action_probability:
            return int(rng.choice(actions))

        return int(
            max(
                actions,
                key=lambda action: self._heuristic_action_score(state, action, drone),
            )
        )

    def _heuristic_action_score(self, state: PlanningState, action: int, drone: Drone) -> float:
        b = state.beliefs[action]
        if self._belief_is_lost(b):
            return -float("inf")

        uncertainty = b.position_logdet + 50.0 if self.use_logdet_objective else b.position_trace
        uncertainty = max(1e-6, uncertainty)
        travel_time = self._travel_time(state.drone_position, b.position, drone.speed)

        recent_factor = self._recent_revisit_factor(b.time_since_seen)
        stale_bonus = 1.0 + min(3.0, b.time_since_seen / 120.0)

        return float(recent_factor * stale_bonus * uncertainty / (travel_time + self.distance_bias_seconds))

    def _recent_revisit_factor(self, time_since_seen: float) -> float:
        if self.recent_revisit_penalty_window <= 0.0:
            return 1.0
        return float(np.clip(time_since_seen / self.recent_revisit_penalty_window, 0.05, 1.0))

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
        selected_before_prediction = next_state.beliefs[action]

        travel_time = self._travel_time(
            next_state.drone_position,
            selected_before_prediction.position,
            drone.speed,
        )
        search_time = estimate.expected_find_time if outcome == "find" else estimate.miss_time
        elapsed = min(next_state.remaining_budget, travel_time + search_time)

        self._predict_all(next_state, elapsed)

        selected = next_state.beliefs[action]
        if outcome == "find" and elapsed > travel_time:
            selected.kalman_position_update_at_mean(self.measurement_noise_std)

        next_state.drone_position = self._position_after_action(
            start=state.drone_position,
            goal=selected.position,
            speed=drone.speed,
            elapsed=elapsed,
        )
        next_state.remaining_budget = max(0.0, next_state.remaining_budget - elapsed)
        next_state.depth += 1

        active_actions = tuple(
            track_id
            for track_id, belief in next_state.beliefs.items()
            if not self._belief_is_lost(belief)
        )

        if outcome == "miss" and self.remove_missed_target_from_branch:
            active_actions = tuple(a for a in active_actions if a != action)

        next_state.available_actions = active_actions
        return next_state

    def _estimate_coverage_outcome(self, state: PlanningState, action: int, drone: Drone) -> CoverageEstimate:
        belief = state.beliefs[action].copy()
        travel_time = self._travel_time(state.drone_position, belief.position, drone.speed)
        remaining_after_travel = max(0.0, state.remaining_budget - travel_time)
        miss_time = min(self.max_search_time, remaining_after_travel)

        if miss_time <= 0.0:
            return CoverageEstimate(p_find=0.0, expected_find_time=0.0, miss_time=0.0)

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
                expected_find_time = float(np.sum(times * increments) / increments.sum())
                expected_find_time = float(np.clip(expected_find_time, 0.0, miss_time))

        return CoverageEstimate(
            p_find=p_find,
            expected_find_time=expected_find_time,
            miss_time=float(miss_time),
        )

    def _coverage_cdf(self, belief: BeliefState, drone: Drone, search_time: float) -> float:
        sensor_width = 2.0 * drone.sensor_range
        initial_area = math.pi * drone.sensor_range**2
        covered_area = initial_area + sensor_width * drone.speed * max(0.0, search_time)

        det = max(float(np.linalg.det(belief.position_covariance)), 1e-12)
        effective_area = math.pi * math.sqrt(det) * (self.covariance_scale_for_detection**2)
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

    def _make_root_state(self, tracks: list[Track], drone: Drone) -> PlanningState:
        beliefs = {int(track.track_id): BeliefState.from_track(track) for track in tracks}
        active_actions = tuple(
            track_id
            for track_id, belief in beliefs.items()
            if not self._belief_is_lost(belief)
        )
        return PlanningState(
            drone_position=drone.position.copy(),
            remaining_budget=float(drone.remaining_budget),
            beliefs=beliefs,
            available_actions=active_actions,
            depth=0,
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
    def _active_candidate_tracks(tracks: TrackSet) -> list[Track]:
        valid_actions = set(tracks.valid_action_ids())
        return [
            track
            for track in tracks.tracks
            if int(track.track_id) in valid_actions
        ]

    @staticmethod
    def _track_id_is_active(tracks: TrackSet, track_id: int) -> bool:
        return int(track_id) in set(tracks.valid_action_ids())

    @staticmethod
    def _travel_time(start: np.ndarray, goal: np.ndarray, speed: float) -> float:
        if speed <= 0.0:
            raise ValueError("drone speed must be positive.")
        return float(np.linalg.norm(goal - start) / speed)

    @staticmethod
    def _position_after_action(start: np.ndarray, goal: np.ndarray, speed: float, elapsed: float) -> np.ndarray:
        max_distance = speed * max(0.0, elapsed)
        delta = goal - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return goal.copy()
        if max_distance >= distance:
            return goal.copy()
        return start + delta / distance * max_distance
