"""Greedy planner baselines for multi-target tracking simulations.

Each planner exposes the same minimal interface expected by simulate_run.py:

    choose_track(tracks, drone, targets, rng) -> int

The planner returns the track_id that the drone should pursue next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from core.sim.drone import Drone
from core.sim.target import TargetSet
from core.sim.tracks import Track, TrackSet


class PlannerProtocol(Protocol):
    """Minimal planner interface used by the simulation runner."""

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        """Return the selected track_id to pursue next."""


def _valid_tracks_or_raise(tracks: TrackSet, planner_name: str) -> list[Track]:
    """Return active selectable tracks for a planner.

    This relies on TrackSet.valid_action_ids() as the shared source of truth.
    """

    valid_actions = set(tracks.valid_action_ids())
    valid_tracks = [
        track
        for track in tracks.tracks
        if int(track.track_id) in valid_actions
    ]

    if not valid_tracks:
        raise ValueError(f"{planner_name} cannot choose from an empty valid action list.")

    return valid_tracks


@dataclass(slots=True)
class GreedyUncertaintyPlanner:
    """Greedy planner that pursues the most uncertain active track.

    Score:
        position covariance trace = sigma_xx + sigma_yy
    """

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(tracks, "GreedyUncertaintyPlanner")

        chosen = max(
            valid_tracks,
            key=lambda track: track.position_variance_trace,
        )
        return int(chosen.track_id)


@dataclass(slots=True)
class GreedyLogDetPlanner:
    """Greedy planner that pursues the active track with largest log-det uncertainty.

    Score:
        log(det(position covariance))
    """

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(tracks, "GreedyLogDetPlanner")

        chosen = max(
            valid_tracks,
            key=lambda track: track.position_uncertainty_logdet,
        )
        return int(chosen.track_id)


@dataclass(slots=True)
class GreedyDistanceAwarePlanner:
    """Greedy planner that balances uncertainty against travel time.

    Score:
        uncertainty / (travel_time + travel_time_bias)
    """

    travel_time_bias: float = 30.0
    use_logdet: bool = False

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        valid_tracks = _valid_tracks_or_raise(tracks, "GreedyDistanceAwarePlanner")

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
        return int(chosen.track_id)