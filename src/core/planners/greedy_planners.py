"""Greedy planner baselines for multi-target tracking simulations.

All greedy planners consume the standardized PlannerInput object.

This guarantees that greedy planners use the exact same valid action set as
random, MCTS, warm MCTS, and guided MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.planners.action_space import validate_action_or_raise
from core.planners.planner_state import PlannerInput
from core.sim.tracks import Track


def _valid_tracks_or_raise(
    planner_input: PlannerInput,
    planner_name: str,
) -> list[Track]:
    """Return selectable tracks using PlannerInput.valid_action_ids."""

    valid_actions = set(planner_input.require_valid_actions(planner_name))

    valid_tracks = [
        track
        for track in planner_input.tracks.tracks
        if int(track.track_id) in valid_actions
    ]

    if not valid_tracks:
        raise ValueError(f"{planner_name} cannot choose from an empty valid track list.")

    return valid_tracks


@dataclass(slots=True)
class GreedyUncertaintyPlanner:
    """Greedy planner that pursues the most uncertain active track.

    Score:
        position covariance trace = sigma_xx + sigma_yy
    """

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(
            planner_input,
            "GreedyUncertaintyPlanner",
        )

        chosen = max(
            valid_tracks,
            key=lambda track: track.position_variance_trace,
        )

        return validate_action_or_raise(
            int(chosen.track_id),
            planner_input.valid_action_ids,
            planner_name="GreedyUncertaintyPlanner",
        )


@dataclass(slots=True)
class GreedyLogDetPlanner:
    """Greedy planner that pursues the active track with largest log-det uncertainty.

    Score:
        log(det(position covariance))
    """

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(
            planner_input,
            "GreedyLogDetPlanner",
        )

        chosen = max(
            valid_tracks,
            key=lambda track: track.position_uncertainty_logdet,
        )

        return validate_action_or_raise(
            int(chosen.track_id),
            planner_input.valid_action_ids,
            planner_name="GreedyLogDetPlanner",
        )


@dataclass(slots=True)
class GreedyDistanceAwarePlanner:
    """Greedy planner that balances uncertainty against travel time.

    Score:
        uncertainty / (travel_time + travel_time_bias)

    This is not the full shared objective, but it now uses the shared action set.
    """

    travel_time_bias: float = 30.0
    use_logdet: bool = False

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(
            planner_input,
            "GreedyDistanceAwarePlanner",
        )

        drone = planner_input.drone

        def score(track: Track) -> float:
            uncertainty = (
                track.position_uncertainty_logdet
                if self.use_logdet
                else track.position_variance_trace
            )

            # If using log-det, values can be negative for tiny covariance.
            # Shift into a positive range so division behaves sanely.
            if self.use_logdet:
                uncertainty = max(1e-6, uncertainty + 50.0)

            travel_time = drone.time_to(track.position)
            return float(uncertainty / (travel_time + self.travel_time_bias))

        chosen = max(valid_tracks, key=score)

        return validate_action_or_raise(
            int(chosen.track_id),
            planner_input.valid_action_ids,
            planner_name="GreedyDistanceAwarePlanner",
        )


@dataclass(slots=True)
class GreedySharedScorePlanner:
    """Greedy one-step approximation of the shared objective.

    This planner estimates the immediate desirability of a target using:
    - uncertainty benefit
    - travel-time penalty
    - active valid action set

    It is useful as a fair baseline against MCTS because it is closer to the same
    kind of objective MCTS should optimize.
    """

    uncertainty_weight: float = 1.0
    travel_time_weight: float = 1.0
    travel_time_bias: float = 30.0
    use_logdet: bool = False

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(
            planner_input,
            "GreedySharedScorePlanner",
        )

        drone = planner_input.drone

        def score(track: Track) -> float:
            uncertainty = (
                track.position_uncertainty_logdet
                if self.use_logdet
                else track.position_variance_trace
            )

            if self.use_logdet:
                uncertainty = max(1e-6, uncertainty + 50.0)

            travel_time = drone.time_to(track.position)

            return float(
                self.uncertainty_weight * uncertainty
                - self.travel_time_weight * (travel_time + self.travel_time_bias)
            )

        chosen = max(valid_tracks, key=score)

        return validate_action_or_raise(
            int(chosen.track_id),
            planner_input.valid_action_ids,
            planner_name="GreedySharedScorePlanner",
        )