"""Drop-in planner for a modular per-track scorer checkpoint.

This planner consumes PlannerInput so it uses the same valid action set as
random, greedy, MCTS, warm MCTS, and guided MCTS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.learning.track_feature_extractors import load_extractor, slot_to_track_id
from core.learning.track_scorer_model import TrackScorerNet
from core.planners.action_space import validate_action_or_raise
from core.planners.planner_state import PlannerInput


class TrackScorerPlanner:
    """Learned one-step target-selection planner.

    The planner:
    1. Builds global + per-track features from PlannerInput.
    2. Runs TrackScorerNet.
    3. Masks invalid/lost/padded tracks.
    4. Selects the highest-scoring valid track.
    """

    def __init__(self, model_path: str = "models/track_scorer_imitation.pt"):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"TrackScorerPlanner checkpoint not found: {model_path}. "
                "Train a model first or update the planner model_path."
            )

        checkpoint = torch.load(model_path, map_location="cpu")

        self.model_path = str(model_path)
        self.extractor = load_extractor(str(checkpoint["extractor"]))
        self.max_tracks = int(checkpoint["max_tracks"])

        self.model = TrackScorerNet(
            global_dim=int(checkpoint["global_dim"]),
            track_dim=int(checkpoint["track_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def choose_track(
        self,
        planner_input: PlannerInput,
        rng: np.random.Generator,
    ) -> int:
        """Choose the highest-scoring valid track."""

        valid_actions = set(
            planner_input.require_valid_actions("TrackScorerPlanner")
        )

        batch = self.extractor.build_batch(
            tracks=planner_input.tracks,
            drone=planner_input.drone,
            max_tracks=self.max_tracks,
        )

        if not np.any(batch.action_mask):
            raise ValueError(
                "TrackScorerPlanner received no active tracks from extractor."
            )

        global_tensor = torch.tensor(
            batch.global_features,
            dtype=torch.float32,
        ).unsqueeze(0)

        track_tensor = torch.tensor(
            batch.track_features,
            dtype=torch.float32,
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            batch.action_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(
                global_tensor,
                track_tensor,
                mask_tensor,
            ).squeeze(0).cpu().numpy()

        # Enforce the simulator-wide valid action rule.
        for slot, track_id in enumerate(batch.track_ids):
            if int(track_id) not in valid_actions:
                scores[slot] = -np.inf

        # Enforce extractor mask too.
        scores[~batch.action_mask] = -np.inf

        if not np.isfinite(scores).any():
            raise ValueError(
                "TrackScorerPlanner found no finite score for any valid action."
            )

        chosen_slot = int(np.argmax(scores))
        chosen_track_id = int(slot_to_track_id(batch.track_ids, chosen_slot))

        return validate_action_or_raise(
            chosen_track_id,
            planner_input.valid_action_ids,
            planner_name="TrackScorerPlanner",
        )

    def score_tracks(
        self,
        planner_input: PlannerInput,
    ) -> dict[int, float]:
        """Return neural scores for valid tracks only.

        This is useful for diagnostics and for guided MCTS.
        """

        valid_actions = set(
            planner_input.require_valid_actions("TrackScorerPlanner")
        )

        batch = self.extractor.build_batch(
            tracks=planner_input.tracks,
            drone=planner_input.drone,
            max_tracks=self.max_tracks,
        )

        global_tensor = torch.tensor(
            batch.global_features,
            dtype=torch.float32,
        ).unsqueeze(0)

        track_tensor = torch.tensor(
            batch.track_features,
            dtype=torch.float32,
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            batch.action_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(
                global_tensor,
                track_tensor,
                mask_tensor,
            ).squeeze(0).cpu().numpy()

        scored_tracks: dict[int, float] = {}

        for slot, track_id in enumerate(batch.track_ids):
            track_id = int(track_id)

            if track_id in valid_actions and bool(batch.action_mask[slot]):
                scored_tracks[track_id] = float(scores[slot])

        return scored_tracks

    def diagnostics(self) -> dict:
        return {
            "track_scorer_model_path": self.model_path,
            "track_scorer_extractor": type(self.extractor).__name__,
            "track_scorer_max_tracks": self.max_tracks,
        }