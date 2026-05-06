"""Planner baselines for multi-target tracking simulations.

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


@dataclass(slots=True)
class RandomPlanner:
    """Baseline planner that randomly selects one track.

    This is useful as a sanity-check lower baseline. It should usually perform
    worse than greedy/MCTS over long runs, but it is very helpful for testing the
    simulator pipeline.
    """

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        if len(tracks.tracks) == 0:
            raise ValueError("RandomPlanner cannot choose from an empty TrackSet.")

        return int(rng.choice([track.track_id for track in tracks.tracks]))


@dataclass(slots=True)
class GreedyUncertaintyPlanner:
    """Greedy planner that pursues the most uncertain track.

    Score:
        position covariance trace = sigma_xx + sigma_yy

    This ignores travel time, target speed, and expected probability of detection.
    It simply asks: "Which target's position belief is currently the most spread
    out?" Then it sends the drone there.
    """

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        if len(tracks.tracks) == 0:
            raise ValueError("GreedyUncertaintyPlanner cannot choose from an empty TrackSet.")

        chosen = max(
            tracks.tracks,
            key=lambda track: track.position_variance_trace,
        )
        return int(chosen.track_id)


@dataclass(slots=True)
class GreedyLogDetPlanner:
    """Greedy planner that pursues the track with largest log-det uncertainty.

    Score:
        log(det(position covariance))

    This is closer to the uncertainty metric used in the paper, which measures
    final uncertainty using log(det(Sigma_xy)) across targets.
    """

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        if len(tracks.tracks) == 0:
            raise ValueError("GreedyLogDetPlanner cannot choose from an empty TrackSet.")

        chosen = max(
            tracks.tracks,
            key=lambda track: track.position_uncertainty_logdet,
        )
        return int(chosen.track_id)


@dataclass(slots=True)
class GreedyDistanceAwarePlanner:
    """Greedy planner that balances uncertainty against travel time.

    Score:
        uncertainty / (travel_time + travel_time_bias)

    This prevents the drone from always chasing a very uncertain target that is
    extremely far away if another highly uncertain target is nearby. This is not
    MCTS, but it is often a stronger heuristic baseline than pure uncertainty.
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
        if len(tracks.tracks) == 0:
            raise ValueError("GreedyDistanceAwarePlanner cannot choose from an empty TrackSet.")

        def score(track: Track) -> float:
            uncertainty = (
                track.position_uncertainty_logdet
                if self.use_logdet
                else track.position_variance_trace
            )

            # If using log-det, values can be negative for tiny covariance.
            # Shift into a positive-ish range so division behaves sanely.
            if self.use_logdet:
                uncertainty = max(1e-6, uncertainty + 50.0)

            travel_time = drone.time_to(track.position)
            return float(uncertainty / (travel_time + self.travel_time_bias))

        chosen = max(tracks.tracks, key=score)
        return int(chosen.track_id)
