"""Warm-start MCTS planner for multi-target tracking.

Warm MCTS is meant to be a middle baseline between greedy and full MCTS.

Behavior:
    - For the first action, choose using a greedy heuristic.
    - After that, use normal MCTS.
    - Conditional/background planning behavior is inherited from realtime MCTS.

This is useful because the first MCTS decision can be noisy or expensive. A
greedy first action gives MCTS a reasonable starting move, then lets MCTS plan
longer-term after the mission has begun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.planners.action_space import validate_action_or_raise
from core.planners.mcts_planner_realtime import MCTSPlanner
from core.planners.planner_state import PlannerInput
from core.sim.tracks import Track


@dataclass(slots=True)
class WarmMCTSPlanner(MCTSPlanner):
    """MCTS planner with a greedy warm-start first action.

    Parameters
    ----------
    warm_start_mode:
        Which greedy heuristic to use for the first action.

        "uncertainty":
            Pick the active track with highest position covariance trace.

        "distance_aware":
            Pick the active track with the best uncertainty/travel-time ratio.

        "shared_score":
            Pick the active track using a simple one-step score:
                uncertainty - travel_time_weight * travel_time

    use_warm_start_once:
        If True, only the first choose_track call is greedy. Later fresh choices
        use MCTS.

        If False, every fresh choose_track call uses the warm-start heuristic.
        Usually keep this True.

    travel_time_bias:
        Stabilizes the distance-aware denominator.

    travel_time_weight:
        Used by shared_score mode.
    """

    warm_start_mode: Literal[
        "uncertainty",
        "distance_aware",
        "shared_score",
    ] = "distance_aware"

    use_warm_start_once: bool = True
    travel_time_bias: float = 30.0
    travel_time_weight: float = 1.0

    _warm_start_used: bool = False

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        """Choose a target using greedy once, then MCTS afterward."""

        should_use_warm_start = (
            not self._warm_start_used
            or not self.use_warm_start_once
        )

        if should_use_warm_start:
            chosen = self._choose_warm_start_track(planner_input)
            self._warm_start_used = True

            return validate_action_or_raise(
                chosen,
                planner_input.valid_action_ids,
                planner_name="WarmMCTSPlanner",
            )

        return int(
            MCTSPlanner.choose_track(
                self,
                planner_input=planner_input,
                rng=rng,
            )
        )

    def _choose_warm_start_track(self, planner_input: PlannerInput) -> int:
        valid_actions = planner_input.require_valid_actions("WarmMCTSPlanner")
        valid_tracks = self._valid_tracks(planner_input)

        if self.warm_start_mode == "uncertainty":
            chosen = max(
                valid_tracks,
                key=lambda track: track.position_variance_trace,
            )
            return int(chosen.track_id)

        if self.warm_start_mode == "distance_aware":
            chosen = max(
                valid_tracks,
                key=lambda track: self._distance_aware_score(
                    track=track,
                    planner_input=planner_input,
                ),
            )
            return int(chosen.track_id)

        if self.warm_start_mode == "shared_score":
            chosen = max(
                valid_tracks,
                key=lambda track: self._shared_one_step_score(
                    track=track,
                    planner_input=planner_input,
                ),
            )
            return int(chosen.track_id)

        raise ValueError(
            f"Unknown warm_start_mode={self.warm_start_mode!r}. "
            f"Expected one of: uncertainty, distance_aware, shared_score."
        )

    @staticmethod
    def _valid_tracks(planner_input: PlannerInput) -> list[Track]:
        valid_actions = set(planner_input.valid_action_ids)

        valid_tracks = [
            track
            for track in planner_input.tracks.tracks
            if int(track.track_id) in valid_actions
        ]

        if not valid_tracks:
            raise ValueError("WarmMCTSPlanner received no valid active tracks.")

        return valid_tracks

    def _distance_aware_score(
        self,
        track: Track,
        planner_input: PlannerInput,
    ) -> float:
        uncertainty = track.position_variance_trace
        travel_time = planner_input.drone.time_to(track.position)

        return float(
            uncertainty / (travel_time + self.travel_time_bias)
        )

    def _shared_one_step_score(
        self,
        track: Track,
        planner_input: PlannerInput,
    ) -> float:
        uncertainty = track.position_variance_trace
        travel_time = planner_input.drone.time_to(track.position)

        return float(
            uncertainty - self.travel_time_weight * travel_time
        )

    def diagnostics(self) -> dict:
        base = MCTSPlanner.diagnostics(self)
        base.update(
            {
                "warm_mcts_warm_start_mode": self.warm_start_mode,
                "warm_mcts_use_warm_start_once": bool(self.use_warm_start_once),
                "warm_mcts_warm_start_used": bool(self._warm_start_used),
                "warm_mcts_travel_time_bias": float(self.travel_time_bias),
                "warm_mcts_travel_time_weight": float(self.travel_time_weight),
            }
        )
        return base