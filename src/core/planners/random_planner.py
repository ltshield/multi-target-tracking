"""Random planner baseline for multi-target tracking simulations.

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
from core.sim.tracks import TrackSet


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
    """Baseline planner that randomly selects one valid active track."""

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        valid_actions = tracks.valid_action_ids()

        if not valid_actions:
            raise ValueError("RandomPlanner cannot choose from an empty valid action list.")

        return int(rng.choice(valid_actions))