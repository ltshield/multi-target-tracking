
"""Shared-weight per-track scoring neural network."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrackScorerNet(nn.Module):
    def __init__(
        self,
        global_dim: int,
        track_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.global_dim = int(global_dim)
        self.track_dim = int(track_dim)
        self.hidden_dim = int(hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(self.global_dim + self.track_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features, track_features, action_mask=None):
        if global_features.dim() != 2:
            raise ValueError("global_features must have shape (B, G).")
        if track_features.dim() != 3:
            raise ValueError("track_features must have shape (B, K, T).")

        batch_size, num_tracks, _ = track_features.shape
        global_repeated = global_features.unsqueeze(1).expand(-1, num_tracks, -1)
        x = torch.cat([global_repeated, track_features], dim=-1)

        scores = self.scorer(x).squeeze(-1)

        if action_mask is not None:
            scores = scores.masked_fill(~action_mask.bool(), -1e9)

        return scores
