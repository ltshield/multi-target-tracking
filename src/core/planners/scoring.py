"""Shared scoring function for planner comparisons and MCTS rollouts.

The goal is to make the planner optimize the same kind of objective that appears
in the final results table.

Higher score is better.
Costs are negative.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.sim.drone import Drone
from core.sim.tracks import TrackSet


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Weights for the shared planner/evaluation score."""

    uncertainty_weight: float = 1.0
    lost_target_weight: float = 1_000_000.0
    travel_distance_weight: float = 0.0
    detection_reward: float = 0.0
    use_logdet: bool = False


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Debuggable cost/score components."""

    uncertainty_cost: float
    lost_target_cost: float
    travel_distance_cost: float
    detection_reward: float
    total_cost: float
    score: float


def tracking_score(
    tracks: TrackSet,
    drone: Drone,
    weights: ScoreWeights,
    *,
    detection_count: int = 0,
) -> ScoreBreakdown:
    """Compute a shared planner score.

    Higher score is better. Lower cost is better.

    This should be used by:
    - greedy variants that want a shared objective
    - MCTS terminal value
    - MCTS rollout scoring
    - sanity scenario checks
    - optionally compare_planners.py summary scoring
    """

    if weights.use_logdet:
        uncertainty = tracks.total_position_logdet(active_only=True)
    else:
        uncertainty = tracks.total_position_trace(active_only=True)

    uncertainty_cost = weights.uncertainty_weight * uncertainty
    lost_target_cost = weights.lost_target_weight * tracks.num_lost()
    travel_distance_cost = weights.travel_distance_weight * drone.distance_traveled
    detection_bonus = weights.detection_reward * detection_count

    total_cost = uncertainty_cost + lost_target_cost + travel_distance_cost - detection_bonus
    score = -total_cost

    return ScoreBreakdown(
        uncertainty_cost=float(uncertainty_cost),
        lost_target_cost=float(lost_target_cost),
        travel_distance_cost=float(travel_distance_cost),
        detection_reward=float(detection_bonus),
        total_cost=float(total_cost),
        score=float(score),
    )


def terminal_track_cost(
    tracks: TrackSet,
    *,
    lost_target_penalty: float = 1_000_000.0,
    use_logdet: bool = False,
) -> float:
    """Small helper for MCTS terminal values.

    Lower is better.
    """

    uncertainty = (
        tracks.total_position_logdet(active_only=True)
        if use_logdet
        else tracks.total_position_trace(active_only=True)
    )

    return float(uncertainty + lost_target_penalty * tracks.num_lost())