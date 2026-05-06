"""Blocking/non-realtime MCTS planner for multi-target tracking.

This file intentionally reuses the realtime MCTS implementation so both MCTS
variants share:
- standardized PlannerInput
- identical valid action handling
- identical rollout scoring
- identical loss/distance/detection objective behavior

The only practical difference is the default iteration budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.planners.mcts_planner_realtime import MCTSPlanner as RealtimeMCTSPlanner


@dataclass(slots=True)
class MCTSPlanner(RealtimeMCTSPlanner):
    """Standard blocking MCTS planner.

    This is used when the simulator calls choose_track(...) from scratch.
    The conditional background-planning methods are inherited, so this class can
    still be used in the realtime simulator if desired.
    """

    iterations: int = 1500