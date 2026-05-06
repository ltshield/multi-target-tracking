"""Paper-style MCTS planner for multi-target tracking.

Plug-in interface expected by simulate_run.py:

    choose_track(tracks, drone, targets, rng) -> int

This version is designed around the actual objective we care about:
minimize future system uncertainty and avoid losing targets.

Main changes from the earlier prototype
---------------------------------------
1. State -> Action -> Outcome -> State tree structure.
2. Each pursuit action has two possible outcomes: "find" and "miss".
3. Terminal value propagates remaining budget to depletion before scoring.
   This prevents the tree from ignoring targets after max_depth is reached.
4. State cost includes a large lost-target penalty.
5. Rollouts include a recent-seen penalty to discourage immediate pointless
   revisits to tracks that were just detected.
6. Root defaults to selecting by value rather than visits.

The low-level spiral is still approximated by a coverage estimator inside MCTS.
That is intentional: the high-level planner reasons about target order and
expected outcomes without executing every spiral waypoint in each rollout.
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
            track_id=self.track_id,
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            existence_probability=self.existence_probability,
            time_since_seen=self.time_since_seen,
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

        self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
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
    """MCTS target-selection planner that minimizes expected uncertainty."""

    # Search budget.
    iterations: int = 1500
    max_depth: int = 5
    exploration_weight: float = 1.4

    # Internal model parameters. These should roughly match SimConfig.
    max_search_time: float = 65.0
    acceleration_noise_std: float = 0.03
    measurement_noise_std: float = 20.0
    covariance_scale_for_detection: float = 3.0

    # Objective settings.
    use_logdet_objective: bool = False
    lost_trace_threshold: float = 250_000.0
    lost_target_penalty: float = 1_000_000.0

    # Recent detections should not be immediately revisited unless uncertainty
    # has had time to regrow.
    recent_revisit_penalty_window: float = 60.0

    # Rollout policy settings.
    rollout_random_action_probability: float = 0.10
    distance_bias_seconds: float = 30.0

    # If True, a miss removes that target from the imagined branch's action set.
    # If your real simulator uses permanent loss thresholds instead of immediate
    # removal on miss, keep this False.
    remove_missed_target_from_branch: bool = False

    # "value" tends to work better during debugging because it directly chooses
    # the action with best estimated objective. "visits" is the classic robust
    # MCTS choice once the model is well tuned.
    root_selection: Literal["visits", "value"] = "value"

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        """Return the track_id that the drone should pursue next."""

        candidate_tracks = self._active_candidate_tracks(tracks)
        if not candidate_tracks:
            raise ValueError("MCTSPlanner cannot choose from an empty active TrackSet.")

        root_state = self._make_root_state(candidate_tracks, drone)
        root = StateNode(state=root_state)

        for _ in range(self.iterations):
            leaf = self._tree_policy(root, drone, rng)
            value = self._rollout(leaf.state.copy(), drone, rng)
            self._backup_state_path(leaf, value)

        if not root.action_nodes:
            return int(rng.choice([track.track_id for track in candidate_tracks]))

        if self.root_selection == "visits":
            best = max(root.action_nodes.values(), key=lambda node: node.visits)
        else:
            best = max(root.action_nodes.values(), key=lambda node: node.mean_value)

        return int(best.action)

    # ------------------------------------------------------------------
    # Tree policy: selection + expansion
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
        """Evaluate a terminal leaf after propagating to budget depletion.

        This is critical. If max_depth is reached while budget remains, ignored
        targets should continue accumulating uncertainty until the mission ends.
        """

        terminal = state.copy()
        if terminal.remaining_budget > 0.0:
            self._predict_all(terminal, terminal.remaining_budget)
            terminal.remaining_budget = 0.0

        return -self._state_cost(terminal)

    def _state_cost(self, state: PlanningState) -> float:
        """System uncertainty cost with permanent-loss penalty."""

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
        actions = list(state.available_actions) or list(state.beliefs.keys())

        if rng.random() < self.rollout_random_action_probability:
            return int(rng.choice(actions))

        return int(max(actions, key=lambda action: self._heuristic_action_score(state, action, drone)))

    def _heuristic_action_score(self, state: PlanningState, action: int, drone: Drone) -> float:
        b = state.beliefs[action]

        if self._belief_is_lost(b):
            return -float("inf")

        uncertainty = b.position_logdet + 50.0 if self.use_logdet_objective else b.position_trace
        uncertainty = max(1e-6, uncertainty)

        travel_time = self._travel_time(state.drone_position, b.position, drone.speed)

        # Discourage immediate revisits to just-seen targets.
        recent_factor = self._recent_revisit_factor(b.time_since_seen)

        # Encourage tracks that have been unseen longer, but cap the effect so it
        # does not dominate uncertainty/travel cost.
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

        travel_time = self._travel_time(next_state.drone_position, selected_before_prediction.position, drone.speed)
        search_time = estimate.expected_find_time if outcome == "find" else estimate.miss_time
        elapsed = min(next_state.remaining_budget, travel_time + search_time)

        # Everyone's covariance grows while the drone travels/searches.
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

        # Remove beliefs that would be considered permanently lost from the
        # action list. They still remain in state.beliefs so the terminal cost
        # can penalize them.
        active_actions = tuple(
            track_id for track_id, belief in next_state.beliefs.items() if not self._belief_is_lost(belief)
        )

        if outcome == "miss" and self.remove_missed_target_from_branch:
            active_actions = tuple(a for a in active_actions if a != action)

        next_state.available_actions = active_actions
        return next_state

    def _estimate_coverage_outcome(self, state: PlanningState, action: int, drone: Drone) -> CoverageEstimate:
        """Estimate P(find), E[T_find], and T_miss for pursuing a target.

        Approximation follows the paper's coverage-estimator spirit:

            covered area ~= sensor_width * drone_speed * t + initial footprint
            normalized area ~= covered_area / (pi * sqrt(det(Sigma_xy)))
            P(find by t) ~= chi2_cdf(normalized area, df=2)

        For df=2: chi2_cdf(x) = 1 - exp(-x/2).
        """

        belief = state.beliefs[action].copy()
        travel_time = self._travel_time(state.drone_position, belief.position, drone.speed)
        remaining_after_travel = max(0.0, state.remaining_budget - travel_time)
        miss_time = min(self.max_search_time, remaining_after_travel)

        if miss_time <= 0.0:
            return CoverageEstimate(p_find=0.0, expected_find_time=0.0, miss_time=0.0)

        # Belief grows while we travel to the target's current predicted center.
        belief.predict(travel_time, self.acceleration_noise_std)

        # Evaluate detection CDF over search time.
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
            # Beliefs already past lost threshold remain in the state for cost,
            # but we do not need to keep increasing them for action selection.
            if self._belief_is_lost(belief):
                continue
            belief.predict(dt, self.acceleration_noise_std)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_root_state(self, tracks: list[Track], drone: Drone) -> PlanningState:
        beliefs = {int(track.track_id): BeliefState.from_track(track) for track in tracks}
        active_actions = tuple(
            track_id for track_id, belief in beliefs.items() if not self._belief_is_lost(belief)
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

    @staticmethod
    def _active_candidate_tracks(tracks: TrackSet) -> list[Track]:
        candidates = []
        for track in tracks.tracks:
            # Compatible with both pre-loss and post-loss Track implementations.
            if getattr(track, "is_lost", False):
                continue
            candidates.append(track)
        return candidates

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
