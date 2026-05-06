"""Shared action-space utilities for all planners.

Every planner should use this module as the source of truth for which targets
are selectable. This prevents random/greedy/MCTS/guided MCTS from silently using
different action lists.
"""

from __future__ import annotations

from core.sim.tracks import TrackSet


def valid_target_actions(tracks: TrackSet) -> tuple[int, ...]:
    """Return track IDs that planners are allowed to select.

    A valid action is currently defined as any non-lost track.

    Keep this function intentionally small and centralized. If we later add
    constraints like "not already selected", "inside mission budget", or
    "above existence probability threshold", they should go here so all planners
    inherit the same action set.
    """

    return tuple(int(track.track_id) for track in tracks.active_tracks)


def validate_action_or_raise(action: int, valid_action_ids: tuple[int, ...], planner_name: str) -> int:
    """Validate that a planner-selected action is legal."""

    action = int(action)

    if action not in valid_action_ids:
        raise ValueError(
            f"{planner_name} selected invalid action {action}. "
            f"Valid actions are {list(valid_action_ids)}."
        )

    return action