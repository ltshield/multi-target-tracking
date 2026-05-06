"""Random planner baseline for multi-target tracking simulations.

Uses the standardized PlannerInput object so it receives the same valid action
set as greedy, MCTS, warm MCTS, and guided MCTS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.planners.action_space import validate_action_or_raise
from core.planners.planner_state import PlannerInput


@dataclass(slots=True)
class RandomPlanner:
    """Baseline planner that randomly selects one valid active track."""

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        valid_actions = planner_input.require_valid_actions("RandomPlanner")

        chosen = int(rng.choice(valid_actions))

        return validate_action_or_raise(
            chosen,
            valid_actions,
            planner_name="RandomPlanner",
        )