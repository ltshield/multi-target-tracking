"""Reference-style MCTS planner using expected uncertainty reduction per unit time.

Drop-in location suggestion:
    src/core/planners/mcts_legacy_debug_planner_expected_reduction.py

Planner path for experiments:
    core.planners.mcts_legacy_debug_planner_expected_reduction.LegacyDebugMDPMCTSPlanner

Purpose
-------
This planner removes threshold-style rescue/loss shaping and instead uses a
principled value function:

    expected uncertainty reduction per unit time

The core idea is:

    If I spend time pursuing/searching this target, how much do I expect the
    system uncertainty to decrease compared with doing nothing over that same
    elapsed time?

This avoids hardcoded rescue thresholds such as "go to target i if trace > X".
Instead, high-uncertainty targets become attractive naturally when:

    p_find * reducible_uncertainty / elapsed_time

is large enough to justify the route.

This hypothesis-test variant also includes:
    1. a soft time-since-seen urgency multiplier, and
    2. an optional worst-track terminal penalty,

to test whether MCTS failures are caused by over-dwelling on a small subset of
targets while other tracks become stale.

Planner interface:
    choose_track(planner_input, rng) -> int

Compatibility methods are included for realtime/conditional planner hooks, but
for this debugging planner they simply fall back to choose_track when needed.
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
UncertaintyCostMode = Literal["trace", "sqrt", "log"]


@dataclass(slots=True)
class DebugBelief:
    """Lightweight copy of one track belief used inside MCTS."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    existence_probability: float = 1.0
    time_since_seen: float = 0.0

    @classmethod
    def from_track(cls, track: Track) -> "DebugBelief":
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

    def copy(self) -> "DebugBelief":
        return DebugBelief(
            track_id=int(self.track_id),
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            existence_probability=float(self.existence_probability),
            time_since_seen=float(self.time_since_seen),
        )

    def predict(self, dt: float, acceleration_noise_std: float) -> None:
        """Constant-velocity Kalman-style prediction."""

        if dt <= 0.0:
            return

        F = constant_velocity_F(dt)
        Q = constant_velocity_Q(dt, acceleration_noise_std)

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.time_since_seen += float(dt)

    def reset_to_reference_covariance(self, reference_covariance: np.ndarray) -> None:
        """Reference-style detection update."""

        self.covariance = np.asarray(reference_covariance, dtype=float).copy()
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.time_since_seen = 0.0
        self.existence_probability = 1.0

    def reset_to_reference_covariance_if_better(self, reference_covariance: np.ndarray) -> None:
        current_trace = self.position_trace

        candidate = np.asarray(reference_covariance, dtype=float).copy()
        candidate = 0.5 * (candidate + candidate.T)
        candidate_trace = float(np.trace(candidate[:2, :2]))

        if candidate_trace < current_trace:
            self.covariance = candidate

        self.time_since_seen = 0.0
        self.existence_probability = 1.0


@dataclass(slots=True)
class DebugPlanningState:
    """State payload stored in MCTS state nodes."""

    drone_position: np.ndarray
    remaining_budget: float
    beliefs: dict[int, DebugBelief]
    available_actions: tuple[int, ...]
    depth: int = 0

    # Diagnostics only.
    distance_traveled: float = 0.0
    detections: int = 0
    misses: int = 0

    # Principle-based value bookkeeping.
    cumulative_reward: float = 0.0
    cumulative_action_time: float = 0.0

    def copy(self) -> "DebugPlanningState":
        return DebugPlanningState(
            drone_position=self.drone_position.copy(),
            remaining_budget=float(self.remaining_budget),
            beliefs={int(k): v.copy() for k, v in self.beliefs.items()},
            available_actions=tuple(int(a) for a in self.available_actions),
            depth=int(self.depth),
            distance_traveled=float(self.distance_traveled),
            detections=int(self.detections),
            misses=int(self.misses),
            cumulative_reward=float(self.cumulative_reward),
            cumulative_action_time=float(self.cumulative_action_time),
        )


@dataclass(slots=True)
class ActionEstimate:
    """Estimated low-level outcome for pursuing/searching one target."""

    p_find: float
    travel_time: float
    expected_search_time: float
    miss_search_time: float
    intercept_position: np.ndarray

    @property
    def find_elapsed(self) -> float:
        return float(self.travel_time + self.expected_search_time)

    @property
    def miss_elapsed(self) -> float:
        return float(self.travel_time + self.miss_search_time)


@dataclass(slots=True)
class StateNode:
    state: DebugPlanningState
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
    estimate: ActionEstimate

    outcome_states: dict[Outcome, StateNode] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass(slots=True)
class LegacyDebugMDPMCTSPlanner:
    """Minimal reference-style MDP/MCTS planner with expected-reduction value."""

    # Search settings.
    iterations: int = 500
    max_depth: int = 6
    exploration_weight: float = 2.0

    # Normalized-UCB safeguards.
    min_root_visits_per_action: int = 20
    min_child_visits_per_action: int = 2

    # Principle-based objective settings.
    uncertainty_cost_mode: UncertaintyCostMode = "trace"
    use_reward_rate: bool = True
    min_reward_elapsed: float = 1.0

    # Optional terminal regularizer. Leave 0.0 for the cleanest test.
    terminal_uncertainty_weight: float = 0.0
    include_terminal_cost_in_value: bool = False

    # Internal model settings.
    max_search_time: float = 120.0
    acceleration_noise_std: float = 0.03
    covariance_scale_for_detection: float = 1.0
    use_intercept_time: bool = False

    # Reference-style fixed detection covariance.
    reset_position_variance: float = 400.0
    reset_velocity_variance: float = 0.005

    # If True, after a simulated miss, the selected target is removed from that
    # imagined branch. This matches the reference planner's behavior.
    remove_missed_target_from_branch: bool = False
    rollout_random_action_probability: float = 0.10

    # Keep this False for the pure principle-based test. If True, it adds a
    # structural anti-dwell simplification after a find.
    prevent_immediate_reselect_after_find: bool = False

    # Soft revisit-balance shaping.
    #
    # This does not force switching. It only makes tracks that have not been seen
    # recently more attractive, so MCTS pays an opportunity cost for camping one
    # target while others become stale.
    use_time_since_seen_urgency: bool = True
    time_since_seen_urgency_weight: float = 0.05
    time_since_seen_urgency_tau: float = 120.0

    # Optional worst-track terminal pressure.
    #
    # This helps test whether MCTS failures are caused by ending with one or two
    # badly neglected tracks.
    use_terminal_worst_track_penalty: bool = False
    terminal_worst_track_weight: float = 0.0

    # If True, print root action statistics every time choose_track is called.
    debug_print_root: bool = False

    # Compatibility state for conditional/realtime hooks.
    current_action: int | None = None

    # Internal diagnostic handle.
    _diagnostic_drone: Drone | None = field(default=None, init=False, repr=False)

    @property
    def reset_covariance(self) -> np.ndarray:
        return np.diag(
            [
                float(self.reset_position_variance),
                float(self.reset_position_variance),
                float(self.reset_velocity_variance),
                float(self.reset_velocity_variance),
            ]
        )

    # ------------------------------------------------------------------
    # Public planner API
    # ------------------------------------------------------------------

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        """Choose a valid track id using reference-style MCTS."""

        valid_actions = planner_input.require_valid_actions("LegacyDebugMDPMCTSPlanner")
        root_state = self._make_root_state(planner_input, valid_actions)
        root = StateNode(state=root_state)

        for _ in range(max(1, int(self.iterations))):
            leaf = self._tree_policy(root, planner_input.drone, rng)
            value = self._rollout(leaf.state.copy(), planner_input.drone, rng)
            self._backup(leaf, value)

        if self.debug_print_root:
            self._diagnostic_drone = planner_input.drone
            self._print_root_diagnostics(root)
            self._diagnostic_drone = None

        if not root.action_nodes:
            chosen = int(rng.choice(valid_actions))
        else:
            # Reference-style root selection: choose highest mean value.
            chosen = int(max(root.action_nodes.values(), key=lambda n: n.mean_value).action)

        return validate_action_or_raise(
            chosen,
            valid_actions,
            planner_name="LegacyDebugMDPMCTSPlanner",
        )

    def diagnostics(self) -> dict:
        return {
            "planner": "LegacyDebugMDPMCTSPlanner",
            "objective": "expected_uncertainty_reduction_per_time",
            "mcts_iterations": int(self.iterations),
            "mcts_max_depth": int(self.max_depth),
            "mcts_exploration_weight": float(self.exploration_weight),
            "mcts_min_root_visits_per_action": int(self.min_root_visits_per_action),
            "mcts_min_child_visits_per_action": int(self.min_child_visits_per_action),
            "mcts_ucb_value_normalization": True,
            "mcts_uncertainty_cost_mode": str(self.uncertainty_cost_mode),
            "mcts_use_reward_rate": bool(self.use_reward_rate),
            "mcts_min_reward_elapsed": float(self.min_reward_elapsed),
            "mcts_terminal_uncertainty_weight": float(self.terminal_uncertainty_weight),
            "mcts_include_terminal_cost_in_value": bool(self.include_terminal_cost_in_value),
            "mcts_max_search_time": float(self.max_search_time),
            "mcts_acceleration_noise_std": float(self.acceleration_noise_std),
            "mcts_covariance_scale_for_detection": float(self.covariance_scale_for_detection),
            "mcts_use_intercept_time": bool(self.use_intercept_time),
            "mcts_remove_missed_target_from_branch": bool(self.remove_missed_target_from_branch),
            "mcts_prevent_immediate_reselect_after_find": bool(self.prevent_immediate_reselect_after_find),
            "mcts_use_time_since_seen_urgency": bool(self.use_time_since_seen_urgency),
            "mcts_time_since_seen_urgency_weight": float(self.time_since_seen_urgency_weight),
            "mcts_time_since_seen_urgency_tau": float(self.time_since_seen_urgency_tau),
            "mcts_use_terminal_worst_track_penalty": bool(self.use_terminal_worst_track_penalty),
            "mcts_terminal_worst_track_weight": float(self.terminal_worst_track_weight),
            "mcts_reset_position_variance": float(self.reset_position_variance),
            "mcts_reset_velocity_variance": float(self.reset_velocity_variance),
        }

    # ------------------------------------------------------------------
    # Compatibility hooks
    # ------------------------------------------------------------------

    def start_conditional_planning(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
        current_action: int,
    ) -> None:
        valid_actions = planner_input.require_valid_actions(
            "LegacyDebugMDPMCTSPlanner.start_conditional_planning"
        )
        self.current_action = validate_action_or_raise(
            int(current_action),
            valid_actions,
            planner_name="LegacyDebugMDPMCTSPlanner.start_conditional_planning",
        )

    def plan_during_execution(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
        planning_seconds: float,
    ) -> None:
        return None

    def finish_conditional_planning(
        self,
        outcome: str,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int | None:
        self.current_action = None
        if not planner_input.valid_action_ids:
            return None
        return int(self.choose_track(planner_input, rng))

    # ------------------------------------------------------------------
    # MCTS tree policy
    # ------------------------------------------------------------------

    def _tree_policy(
        self,
        node: StateNode,
        drone: Drone,
        rng: np.random.Generator,
    ) -> StateNode:
        while not self._is_terminal(node.state):
            self._refresh_node_actions(node)

            if not node.state.available_actions:
                return node

            if node.unexpanded_actions:
                return self._expand(node, drone, rng)

            action_node = self._select_ucb_action(node)
            outcome = self._select_tree_outcome(action_node, rng)

            if outcome not in action_node.outcome_states:
                next_state = self._transition(
                    state=node.state,
                    action=action_node.action,
                    outcome=outcome,
                    estimate=action_node.estimate,
                    drone=drone,
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

    def _expand(self, node: StateNode, drone: Drone, rng: np.random.Generator) -> StateNode:
        idx = int(rng.integers(0, len(node.unexpanded_actions)))
        action = int(node.unexpanded_actions.pop(idx))
        estimate = self._estimate_action(node.state, action, drone)

        action_node = ActionNode(
            parent_state=node,
            action=action,
            estimate=estimate,
        )
        node.action_nodes[action] = action_node

        outcome = self._select_tree_outcome(action_node, rng)
        next_state = self._transition(node.state, action, outcome, estimate, drone)
        child = StateNode(
            state=next_state,
            parent_action=action_node,
            outcome_from_parent=outcome,
        )
        action_node.outcome_states[outcome] = child
        return child

    def _select_ucb_action(self, node: StateNode) -> ActionNode:
        """Select action using normalized UCB.

        Raw values may be large depending on the cost mode, so sibling values
        are normalized before adding the exploration term.
        """

        actions = list(node.action_nodes.values())

        if not actions:
            raise RuntimeError("UCB selection requested with no action nodes.")

        unvisited = [action for action in actions if action.visits == 0]
        if unvisited:
            return unvisited[0]

        if node.parent_action is None:
            under_sampled = [
                action for action in actions
                if action.visits < int(self.min_root_visits_per_action)
            ]
            if under_sampled:
                return min(under_sampled, key=lambda action: action.visits)

        elif int(self.min_child_visits_per_action) > 0:
            under_sampled = [
                action for action in actions
                if action.visits < int(self.min_child_visits_per_action)
            ]
            if under_sampled:
                return min(under_sampled, key=lambda action: action.visits)

        values = np.array([action.mean_value for action in actions], dtype=float)
        finite_mask = np.isfinite(values)

        if not np.any(finite_mask):
            return min(actions, key=lambda action: action.visits)

        finite_values = values[finite_mask]
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        value_range = max(1e-9, value_max - value_min)

        parent_visits = max(2, int(node.visits))

        def priority(action_node: ActionNode) -> float:
            normalized_exploit = (
                (float(action_node.mean_value) - value_min) / value_range
                if np.isfinite(action_node.mean_value)
                else 0.0
            )

            exploration = float(self.exploration_weight) * math.sqrt(
                math.log(parent_visits) / max(1, int(action_node.visits))
            )

            return float(normalized_exploit + exploration)

        return max(actions, key=priority)

    def _select_tree_outcome(self, action_node: ActionNode, rng: np.random.Generator) -> Outcome:
        """Reference-style outcome selection for tree growth."""

        has_find = "find" in action_node.outcome_states
        has_miss = "miss" in action_node.outcome_states

        if not has_find and not has_miss:
            return self._sample_outcome(action_node.estimate.p_find, rng)

        if not has_miss:
            return "miss"

        if not has_find:
            return "find"

        return self._sample_outcome(action_node.estimate.p_find, rng)

    @staticmethod
    def _sample_outcome(p_find: float, rng: np.random.Generator) -> Outcome:
        return "find" if rng.random() <= float(np.clip(p_find, 0.0, 1.0)) else "miss"

    def _backup(self, leaf: StateNode, value: float) -> None:
        node: Optional[StateNode] = leaf

        while node is not None:
            node.visits += 1
            node.value_sum += float(value)

            parent_action = node.parent_action
            if parent_action is None:
                break

            parent_action.visits += 1
            parent_action.value_sum += float(value)
            node = parent_action.parent_state

    # ------------------------------------------------------------------
    # Rollout and value
    # ------------------------------------------------------------------

    def _rollout(
        self,
        state: DebugPlanningState,
        drone: Drone,
        rng: np.random.Generator,
    ) -> float:
        """Random rollout to terminal/depth/budget exhaustion."""

        while not self._is_terminal(state):
            actions = tuple(
                int(a) for a in state.available_actions if int(a) in state.beliefs
            )
            if not actions:
                break

            if rng.random() < self.rollout_random_action_probability:
                action = int(rng.choice(actions))
            else:
                action = self._select_rollout_action_by_expected_reduction(state, actions, drone)
            estimate = self._estimate_action(state, action, drone)
            outcome = self._sample_outcome(estimate.p_find, rng)
            state = self._transition(state, action, outcome, estimate, drone)

        if state.remaining_budget > 0.0:
            state = self._carry_state_to_budget_end(state)

        return float(self._rollout_value(state))
    
    def _select_rollout_action_by_expected_reduction(
        self,
        state: DebugPlanningState,
        actions: tuple[int, ...],
        drone: Drone,
    ) -> int:
        best_action = int(actions[0])
        best_score = -float("inf")

        for action in actions:
            estimate = self._estimate_action(state, int(action), drone)
            elapsed = min(float(state.remaining_budget), float(estimate.find_elapsed))

            score = self._expected_uncertainty_reduction_reward(
                state=state,
                action=int(action),
                estimate=estimate,
                elapsed=elapsed,
            )
            score *= self._time_since_seen_urgency_multiplier(
                state.beliefs[int(action)]
            )

            if score > best_score:
                best_score = score
                best_action = int(action)

        return best_action

    def _time_since_seen_urgency_multiplier(self, belief: DebugBelief) -> float:
        """Soft urgency multiplier for stale tracks.

        Returns a multiplier in approximately [1, 1 + weight].

        A recently seen track gets multiplier near 1.
        A stale track approaches 1 + time_since_seen_urgency_weight.

        This does not ban repeated selection. It only makes neglected tracks
        gradually more competitive.
        """

        if not self.use_time_since_seen_urgency:
            return 1.0

        tau = max(1e-9, float(self.time_since_seen_urgency_tau))
        weight = max(0.0, float(self.time_since_seen_urgency_weight))
        stale = max(0.0, float(belief.time_since_seen))

        urgency = 1.0 - math.exp(-stale / tau)

        return float(1.0 + weight * urgency)

    def _rollout_value(self, state: DebugPlanningState) -> float:
        """Return accumulated rollout value.

        The cumulative reward captures useful local uncertainty reductions. The
        optional worst-track terminal penalty is a soft hypothesis-test
        regularizer: it penalizes rollouts that end with one badly neglected or
        high-uncertainty track.
        """

        value = float(state.cumulative_reward)

        if self.include_terminal_cost_in_value and self.terminal_uncertainty_weight != 0.0:
            value -= (
                float(self.terminal_uncertainty_weight)
                * self._system_uncertainty_cost(state)
            )

        if self.use_terminal_worst_track_penalty:
            value -= (
                float(self.terminal_worst_track_weight)
                * self._max_track_uncertainty_cost(state)
            )

        return float(value)

    def _system_uncertainty_cost(self, state: DebugPlanningState) -> float:
        """Scale-stable system uncertainty cost without scenario thresholds."""

        costs: list[float] = []

        for belief in state.beliefs.values():
            trace = max(0.0, float(belief.position_trace))

            if self.uncertainty_cost_mode == "trace":
                costs.append(trace)

            elif self.uncertainty_cost_mode == "sqrt":
                costs.append(math.sqrt(trace))

            elif self.uncertainty_cost_mode == "log":
                costs.append(math.log1p(trace))

            else:
                raise ValueError(
                    f"Unknown uncertainty_cost_mode={self.uncertainty_cost_mode!r}. "
                    "Expected 'trace', 'sqrt', or 'log'."
                )

        return float(sum(costs))

    def _max_track_uncertainty_cost(self, state: DebugPlanningState) -> float:
        """Worst single-track uncertainty cost.

        This penalizes rollouts that end with one badly neglected target,
        without forcing the planner to visit targets in a fixed order.
        """

        if not state.beliefs:
            return 0.0

        costs: list[float] = []

        for belief in state.beliefs.values():
            trace = max(0.0, float(belief.position_trace))

            if self.uncertainty_cost_mode == "trace":
                costs.append(trace)
            elif self.uncertainty_cost_mode == "sqrt":
                costs.append(math.sqrt(trace))
            elif self.uncertainty_cost_mode == "log":
                costs.append(math.log1p(trace))
            else:
                raise ValueError(
                    f"Unknown uncertainty_cost_mode={self.uncertainty_cost_mode!r}. "
                    "Expected 'trace', 'sqrt', or 'log'."
                )

        return float(max(costs))

    def _expected_uncertainty_reduction_reward(
        self,
        state: DebugPlanningState,
        action: int,
        estimate: ActionEstimate,
        elapsed: float,
    ) -> float:
        """Expected uncertainty reduction for one action.

        Baseline is "do nothing useful for the same elapsed time": all beliefs
        propagate, but no target gets reset. The action value is the expected
        improvement relative to that baseline.
        """

        if elapsed <= 0.0 or int(action) not in state.beliefs:
            return 0.0

        baseline_state = state.copy()
        self._predict_all(baseline_state, elapsed)
        baseline_cost = self._system_uncertainty_cost(baseline_state)

        find_state = baseline_state.copy()
        find_state.beliefs[int(action)].reset_to_reference_covariance_if_better(self.reset_covariance)
        find_cost = self._system_uncertainty_cost(find_state)

        miss_cost = baseline_cost

        p_find = float(np.clip(estimate.p_find, 0.0, 1.0))
        expected_post_cost = p_find * find_cost + (1.0 - p_find) * miss_cost

        expected_reduction = baseline_cost - expected_post_cost

        if self.use_reward_rate:
            expected_reduction /= max(float(self.min_reward_elapsed), float(elapsed))

        return float(expected_reduction)

    def _carry_state_to_budget_end(self, state: DebugPlanningState) -> DebugPlanningState:
        next_state = state.copy()
        elapsed = float(next_state.remaining_budget)
        if elapsed > 0.0:
            self._predict_all(next_state, elapsed)
            next_state.remaining_budget = 0.0
        return next_state

    # ------------------------------------------------------------------
    # Transition and action estimation
    # ------------------------------------------------------------------

    def _transition(
        self,
        state: DebugPlanningState,
        action: int,
        outcome: Outcome,
        estimate: ActionEstimate,
        drone: Drone,
    ) -> DebugPlanningState:
        next_state = state.copy()
        action = int(action)

        if action not in next_state.beliefs:
            next_state.available_actions = tuple(
                int(a) for a in next_state.available_actions if int(a) != action
            )
            return next_state

        elapsed_nominal = estimate.find_elapsed if outcome == "find" else estimate.miss_elapsed
        elapsed = min(float(next_state.remaining_budget), float(elapsed_nominal))

        if elapsed <= 0.0:
            if outcome == "miss" and self.remove_missed_target_from_branch:
                next_state.available_actions = tuple(
                    int(a) for a in next_state.available_actions if int(a) != action
                )
            return next_state

        reward = self._expected_uncertainty_reduction_reward(
            state=state,
            action=action,
            estimate=estimate,
            elapsed=elapsed,
        )
        reward *= self._time_since_seen_urgency_multiplier(state.beliefs[action])

        self._predict_all(next_state, elapsed)

        found = outcome == "find" and elapsed >= estimate.travel_time
        if found:
            next_state.beliefs[action].reset_to_reference_covariance_if_better(self.reset_covariance)
            next_state.detections += 1
        else:
            next_state.misses += 1

        new_position = self._position_after_action(
            start=state.drone_position,
            goal=estimate.intercept_position,
            speed=drone.speed,
            elapsed=elapsed,
        )

        next_state.distance_traveled += float(np.linalg.norm(new_position - state.drone_position))
        next_state.drone_position = new_position
        next_state.remaining_budget = max(0.0, next_state.remaining_budget - elapsed)
        next_state.depth += 1

        next_state.cumulative_reward += float(reward)
        next_state.cumulative_action_time += float(elapsed)

        active_actions = [int(a) for a in next_state.available_actions if int(a) in next_state.beliefs]

        if outcome == "miss" and self.remove_missed_target_from_branch:
            active_actions = [a for a in active_actions if int(a) != action]

        if (
            self.prevent_immediate_reselect_after_find
            and found
            and any(int(a) != action for a in active_actions)
        ):
            active_actions = [a for a in active_actions if int(a) != action]

        next_state.available_actions = tuple(active_actions)
        return next_state

    def _estimate_action(
        self,
        state: DebugPlanningState,
        action: int,
        drone: Drone,
    ) -> ActionEstimate:
        belief = state.beliefs[int(action)].copy()

        travel_time, intercept_position = self._intercept_time_and_position(
            drone_position=state.drone_position,
            target_position=belief.position,
            target_velocity=belief.velocity,
            drone_speed=drone.speed,
        )

        remaining_after_travel = max(0.0, float(state.remaining_budget) - travel_time)
        miss_search_time = min(float(self.max_search_time), remaining_after_travel)

        if miss_search_time <= 0.0:
            return ActionEstimate(
                p_find=0.0,
                travel_time=float(travel_time),
                expected_search_time=0.0,
                miss_search_time=0.0,
                intercept_position=intercept_position,
            )

        belief.predict(travel_time, self.acceleration_noise_std)

        num_steps = max(8, int(math.ceil(miss_search_time)))
        times = np.linspace(0.0, miss_search_time, num_steps + 1)
        cdf = np.zeros_like(times)

        for i, search_t in enumerate(times):
            b = belief.copy()
            b.predict(float(search_t), self.acceleration_noise_std)
            cdf[i] = self._coverage_cdf(b, drone, float(search_t))

        cdf = np.maximum.accumulate(cdf)
        p_find = float(np.clip(cdf[-1] * belief.existence_probability, 0.0, 1.0))

        if p_find <= 1e-12:
            expected_search_time = miss_search_time
        else:
            increments = np.diff(cdf, prepend=0.0)
            increments = np.maximum(increments, 0.0)
            if increments.sum() <= 1e-12:
                expected_search_time = miss_search_time
            else:
                expected_search_time = float(np.sum(times * increments) / increments.sum())
                expected_search_time = float(np.clip(expected_search_time, 0.0, miss_search_time))

        return ActionEstimate(
            p_find=p_find,
            travel_time=float(travel_time),
            expected_search_time=float(expected_search_time),
            miss_search_time=float(miss_search_time),
            intercept_position=intercept_position,
        )

    def _coverage_cdf(
        self,
        belief: DebugBelief,
        drone: Drone,
        search_time: float,
    ) -> float:
        """Simple coverage-area / covariance-area approximation."""

        sensor_width = 2.0 * float(drone.sensor_range)
        initial_area = math.pi * float(drone.sensor_range) ** 2
        covered_area = initial_area + sensor_width * float(drone.speed) * max(0.0, float(search_time))

        det = max(float(np.linalg.det(belief.position_covariance)), 1e-12)
        effective_area = (
            math.pi
            * math.sqrt(det)
            * (float(self.covariance_scale_for_detection) ** 2)
        )

        normalized = covered_area / max(effective_area, 1e-12)
        return float(np.clip(1.0 - math.exp(-0.5 * normalized), 0.0, 1.0))

    def _predict_all(self, state: DebugPlanningState, dt: float) -> None:
        for belief in state.beliefs.values():
            belief.predict(float(dt), self.acceleration_noise_std)

    # ------------------------------------------------------------------
    # Root state and helpers
    # ------------------------------------------------------------------

    def _make_root_state(
        self,
        planner_input: PlannerInput,
        valid_action_ids: tuple[int, ...],
    ) -> DebugPlanningState:
        valid_set = set(int(a) for a in valid_action_ids)

        beliefs = {
            int(track.track_id): DebugBelief.from_track(track)
            for track in planner_input.tracks.tracks
            if int(track.track_id) in valid_set
        }

        active_actions = tuple(
            int(a) for a in valid_action_ids if int(a) in beliefs
        )

        return DebugPlanningState(
            drone_position=planner_input.drone.position.copy(),
            remaining_budget=float(planner_input.drone.remaining_budget),
            beliefs=beliefs,
            available_actions=active_actions,
        )

    def _is_terminal(self, state: DebugPlanningState) -> bool:
        return (
            state.remaining_budget <= 0.0
            or state.depth >= int(self.max_depth)
            or not state.available_actions
        )

    def _refresh_node_actions(self, node: StateNode) -> None:
        valid = set(int(a) for a in node.state.available_actions if int(a) in node.state.beliefs)
        node.state.available_actions = tuple(sorted(valid))
        node.unexpanded_actions = [int(a) for a in node.unexpanded_actions if int(a) in valid]
        node.action_nodes = {
            int(a): child for a, child in node.action_nodes.items() if int(a) in valid
        }

    def _intercept_time_and_position(
        self,
        drone_position: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        drone_speed: float,
    ) -> tuple[float, np.ndarray]:
        if drone_speed <= 0.0:
            raise ValueError("drone speed must be positive.")

        rel = np.asarray(target_position, dtype=float) - np.asarray(drone_position, dtype=float)
        vel = np.asarray(target_velocity, dtype=float)

        if not self.use_intercept_time:
            t = float(np.linalg.norm(rel) / drone_speed)
            return t, np.asarray(target_position, dtype=float).copy()

        # Solve ||rel + vel*t|| = drone_speed*t.
        a = float(np.dot(vel, vel) - drone_speed**2)
        b = float(2.0 * np.dot(rel, vel))
        c = float(np.dot(rel, rel))

        roots: list[float] = []
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                roots.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                roots.append((-b - sqrt_disc) / (2.0 * a))
                roots.append((-b + sqrt_disc) / (2.0 * a))

        feasible = [r for r in roots if np.isfinite(r) and r >= 0.0]
        if feasible:
            t = float(min(feasible))
        else:
            t = float(np.linalg.norm(rel) / drone_speed)

        intercept_position = np.asarray(target_position, dtype=float) + vel * t
        return t, intercept_position

    @staticmethod
    def _position_after_action(
        start: np.ndarray,
        goal: np.ndarray,
        speed: float,
        elapsed: float,
    ) -> np.ndarray:
        max_distance = float(speed) * max(0.0, float(elapsed))
        delta = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
        distance = float(np.linalg.norm(delta))

        if distance <= 1e-12:
            return np.asarray(goal, dtype=float).copy()

        if max_distance >= distance:
            return np.asarray(goal, dtype=float).copy()

        return np.asarray(start, dtype=float) + delta / distance * max_distance

    def _print_root_diagnostics(self, root: StateNode) -> None:
        print("\nLegacyDebugMDPMCTSPlanner root diagnostics")
        print("-" * 128)

        actions = [root.action_nodes[a] for a in sorted(root.action_nodes)]
        values = np.array([node.mean_value for node in actions], dtype=float)

        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            value_min = float(np.min(finite_values))
            value_max = float(np.max(finite_values))
            value_range = max(1e-9, value_max - value_min)
        else:
            value_min = 0.0
            value_range = 1.0

        parent_visits = max(2, int(root.visits))

        print(
            f"{'action':>8} {'trace':>10} {'p_find':>8} "
            f"{'travel':>8} {'Esearch':>8} {'miss_t':>8} "
            f"{'immR':>10} {'visits':>8} {'normV':>8} "
            f"{'explore':>8} {'mean_value':>14}"
        )

        for action in sorted(root.action_nodes):
            node = root.action_nodes[action]
            est = node.estimate
            belief = root.state.beliefs.get(int(action))

            trace = float("nan") if belief is None else float(belief.position_trace)

            norm_value = (
                (float(node.mean_value) - value_min) / value_range
                if np.isfinite(node.mean_value)
                else 0.0
            )
            explore = float(self.exploration_weight) * math.sqrt(
                math.log(parent_visits) / max(1, int(node.visits))
            )

            elapsed = min(float(root.state.remaining_budget), float(est.find_elapsed))
            immediate_reward = self._expected_uncertainty_reduction_reward(
                state=root.state,
                action=int(action),
                estimate=est,
                elapsed=elapsed,
            )

            print(
                f"{action:8d} "
                f"{trace:10.1f} "
                f"{est.p_find:8.3f} "
                f"{est.travel_time:8.2f} "
                f"{est.expected_search_time:8.2f} "
                f"{est.miss_search_time:8.2f} "
                f"{immediate_reward:10.3f} "
                f"{node.visits:8d} "
                f"{norm_value:8.3f} "
                f"{explore:8.3f} "
                f"{node.mean_value:14.2f}"
            )

        print("-" * 128)


# Convenience aliases.
ExpectedReductionMDPMCTSPlanner = LegacyDebugMDPMCTSPlanner
DebugMDPMCTSPlanner = LegacyDebugMDPMCTSPlanner
