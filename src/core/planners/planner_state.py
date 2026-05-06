"""Standard planner input object.

The simulator should build exactly one PlannerInput at each decision point and
pass that same object to every planner.

This makes planner comparisons fair because every planner receives:
- the same belief state
- the same UAV/drone state
- the same active tracks
- the same valid action list
- the same lost count
- the same current time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.planners.action_space import valid_target_actions
from core.sim.drone import Drone
from core.sim.target import TargetSet
from core.sim.tracks import Track, TrackSet


@dataclass(frozen=True, slots=True)
class PlannerInput:
    """Immutable decision-time snapshot given to planners."""

    tracks: TrackSet
    drone: Drone
    targets: TargetSet

    valid_action_ids: tuple[int, ...]
    time: float
    lost_count: int
    active_count: int

    @property
    def belief_state(self) -> TrackSet:
        """Alias for readability: tracks are the planner's belief state."""
        return self.tracks

    @property
    def uav_state(self) -> Drone:
        """Alias for readability."""
        return self.drone

    def valid_tracks(self) -> list[Track]:
        """Return active tracks corresponding to valid_action_ids."""

        valid = set(self.valid_action_ids)
        return [
            track
            for track in self.tracks.tracks
            if int(track.track_id) in valid
        ]

    def require_valid_actions(self, planner_name: str) -> tuple[int, ...]:
        """Return valid actions or raise a clear planner-specific error."""

        if not self.valid_action_ids:
            raise ValueError(
                f"{planner_name} cannot choose from an empty valid action list."
            )

        return self.valid_action_ids

    def track_by_id(self, track_id: int) -> Track:
        return self.tracks[int(track_id)]

    def drone_position(self) -> np.ndarray:
        return self.drone.position.copy()


def make_planner_input(
    tracks: TrackSet,
    drone: Drone,
    targets: TargetSet,
    *,
    full_tracks_for_lost_count: TrackSet | None = None,
) -> PlannerInput:
    """Build the standardized planner input.

    Parameters
    ----------
    tracks:
        The TrackSet planners are allowed to reason over. Usually this should be
        active tracks only, but it can also be a full copied TrackSet as long as
        valid_action_ids filters lost tracks.

    drone:
        Current UAV state.

    targets:
        Current target set. Planners generally should not use true target state
        for real decisions, but this remains available for simulator/planning
        compatibility.

    full_tracks_for_lost_count:
        Optional full TrackSet so lost_count reflects the whole simulation even
        if tracks contains only active tracks.
    """

    valid_actions = valid_target_actions(tracks)
    lost_source = full_tracks_for_lost_count if full_tracks_for_lost_count is not None else tracks

    return PlannerInput(
        tracks=tracks,
        drone=drone,
        targets=targets,
        valid_action_ids=valid_actions,
        time=float(drone.elapsed_time),
        lost_count=int(lost_source.num_lost()),
        active_count=int(len(valid_actions)),
    )