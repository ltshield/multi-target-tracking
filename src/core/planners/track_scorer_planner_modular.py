"""Drop-in planner for a modular per-track scorer checkpoint."""

from __future__ import annotations

import numpy as np
import torch

from core.learning.track_feature_extractors import load_extractor, slot_to_track_id
from core.learning.track_scorer_model import TrackScorerNet


class TrackScorerPlanner:
    def __init__(self, model_path: str = "models/track_scorer.pt"):
        checkpoint = torch.load(model_path, map_location="cpu")

        self.extractor = load_extractor(str(checkpoint["extractor"]))
        self.max_tracks = int(checkpoint["max_tracks"])

        self.model = TrackScorerNet(
            global_dim=int(checkpoint["global_dim"]),
            track_dim=int(checkpoint["track_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def choose_track(self, tracks, drone, targets, rng) -> int:
        valid_actions = set(tracks.valid_action_ids())

        if not valid_actions:
            raise ValueError("TrackScorerPlanner received no valid active tracks.")

        batch = self.extractor.build_batch(
            tracks=tracks,
            drone=drone,
            max_tracks=self.max_tracks,
        )

        if not np.any(batch.action_mask):
            raise ValueError("TrackScorerPlanner received no active tracks from extractor.")

        g = torch.tensor(batch.global_features, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(batch.track_features, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(batch.action_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(g, t, m).squeeze(0).numpy()

        # Enforce the simulator/planner shared valid-action rule, even if the
        # feature extractor accidentally exposes an invalid slot.
        for i, track_id in enumerate(batch.track_ids):
            if int(track_id) not in valid_actions:
                scores[i] = -np.inf

        scores[~batch.action_mask] = -np.inf

        if not np.isfinite(scores).any():
            raise ValueError("TrackScorerPlanner found no finite score for any valid action.")

        slot = int(np.argmax(scores))
        chosen_track_id = int(slot_to_track_id(batch.track_ids, slot))

        if chosen_track_id not in valid_actions:
            raise RuntimeError(
                f"TrackScorerPlanner selected invalid track_id={chosen_track_id}."
            )

        return chosen_track_id

    def score_tracks(self, tracks, drone) -> dict[int, float]:
        valid_actions = set(tracks.valid_action_ids())

        batch = self.extractor.build_batch(
            tracks=tracks,
            drone=drone,
            max_tracks=self.max_tracks,
        )

        g = torch.tensor(batch.global_features, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(batch.track_features, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(batch.action_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(g, t, m).squeeze(0).numpy()

        return {
            int(track_id): float(scores[i])
            for i, track_id in enumerate(batch.track_ids)
            if int(track_id) in valid_actions and bool(batch.action_mask[i])
        }