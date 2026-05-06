"""Planner baselines for multi-target tracking simulations.

Each planner exposes the same minimal interface expected by simulate_run.py:

    choose_track(tracks, drone, targets, rng) -> int

The planner returns the track_id that the drone should pursue next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from drone import Drone
from target import TargetSet
from tracks import Track, TrackSet


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
