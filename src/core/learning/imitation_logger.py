"""Imitation-learning decision logger.

This module records expert planner decisions at simulator decision points.

Each example stores:
- normalized global features
- normalized per-track features
- action mask
- track IDs corresponding to action slots
- expert-selected track ID
- expert-selected action slot
- scenario/episode metadata

The generated tensors are compatible with TrackScorerNet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.learning.track_feature_extractors import (
    FeatureBatch,
    load_extractor,
    track_id_to_slot,
)
from core.planners.planner_state import PlannerInput


@dataclass(slots=True)
class ImitationExample:
    split: str
    episode_id: int
    scenario_seed: int
    decision_index: int
    time: float
    source: str

    global_features: np.ndarray
    track_features: np.ndarray
    action_mask: np.ndarray
    track_ids: list[int]
    valid_action_ids: list[int]

    expert_track_id: int
    expert_action_slot: int

    num_active: int
    num_lost: int


@dataclass(slots=True)
class ImitationLogger:
    """Collect imitation-learning examples during simulation."""

    extractor_name: str = "cartesian"
    max_tracks: int = 6
    examples: list[ImitationExample] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.extractor = load_extractor(self.extractor_name)

    @property
    def global_dim(self) -> int:
        return int(self.extractor.global_dim)

    @property
    def track_dim(self) -> int:
        return int(self.extractor.track_dim)

    def log_decision(
        self,
        *,
        planner_input: PlannerInput,
        selected_track_id: int,
        episode_id: int,
        scenario_seed: int,
        decision_index: int,
        split: str,
        source: str = "expert",
    ) -> None:
        """Log one expert decision.

        Parameters
        ----------
        planner_input:
            The exact standardized decision snapshot given to the expert planner.

        selected_track_id:
            The expert-selected target/track ID.

        episode_id:
            Index of the scenario within the current split.

        scenario_seed:
            Random seed used to generate the scenario.

        decision_index:
            Decision counter before executing the selected action.

        split:
            train, val, or test.

        source:
            Optional label describing whether the action came from direct planner
            selection, cached conditional planning, fallback, etc.
        """

        selected_track_id = int(selected_track_id)
        valid_action_ids = tuple(int(a) for a in planner_input.valid_action_ids)

        if selected_track_id not in valid_action_ids:
            raise ValueError(
                f"Cannot log invalid expert action {selected_track_id}. "
                f"Valid actions are {list(valid_action_ids)}."
            )

        batch: FeatureBatch = self.extractor.build_batch(
            tracks=planner_input.tracks,
            drone=planner_input.drone,
            max_tracks=self.max_tracks,
        )

        expert_action_slot = track_id_to_slot(batch.track_ids, selected_track_id)

        if expert_action_slot < 0 or expert_action_slot >= self.max_tracks:
            raise ValueError(
                f"Expert slot {expert_action_slot} is outside max_tracks={self.max_tracks}."
            )

        if not bool(batch.action_mask[expert_action_slot]):
            raise ValueError(
                f"Expert selected track_id={selected_track_id}, but its action slot "
                f"{expert_action_slot} is masked invalid."
            )

        self.examples.append(
            ImitationExample(
                split=str(split),
                episode_id=int(episode_id),
                scenario_seed=int(scenario_seed),
                decision_index=int(decision_index),
                time=float(planner_input.time),
                source=str(source),
                global_features=batch.global_features.astype(np.float32, copy=True),
                track_features=batch.track_features.astype(np.float32, copy=True),
                action_mask=batch.action_mask.astype(bool, copy=True),
                track_ids=[int(x) for x in batch.track_ids],
                valid_action_ids=[int(x) for x in valid_action_ids],
                expert_track_id=selected_track_id,
                expert_action_slot=int(expert_action_slot),
                num_active=int(planner_input.active_count),
                num_lost=int(planner_input.lost_count),
            )
        )

    def extend(self, other: "ImitationLogger") -> None:
        self.examples.extend(other.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def to_tensor_dict(self) -> dict[str, Any]:
        """Convert logged examples to a PyTorch-friendly dictionary."""

        if not self.examples:
            raise ValueError("Cannot convert empty imitation logger to tensors.")

        global_features = np.stack(
            [ex.global_features for ex in self.examples],
            axis=0,
        ).astype(np.float32)

        track_features = np.stack(
            [ex.track_features for ex in self.examples],
            axis=0,
        ).astype(np.float32)

        action_masks = np.stack(
            [ex.action_mask for ex in self.examples],
            axis=0,
        ).astype(bool)

        labels = np.asarray(
            [ex.expert_action_slot for ex in self.examples],
            dtype=np.int64,
        )

        expert_track_ids = np.asarray(
            [ex.expert_track_id for ex in self.examples],
            dtype=np.int64,
        )

        track_ids = np.asarray(
            [ex.track_ids for ex in self.examples],
            dtype=np.int64,
        )

        scenario_seeds = np.asarray(
            [ex.scenario_seed for ex in self.examples],
            dtype=np.int64,
        )

        episode_ids = np.asarray(
            [ex.episode_id for ex in self.examples],
            dtype=np.int64,
        )

        decision_indices = np.asarray(
            [ex.decision_index for ex in self.examples],
            dtype=np.int64,
        )

        times = np.asarray(
            [ex.time for ex in self.examples],
            dtype=np.float32,
        )

        num_active = np.asarray(
            [ex.num_active for ex in self.examples],
            dtype=np.int64,
        )

        num_lost = np.asarray(
            [ex.num_lost for ex in self.examples],
            dtype=np.int64,
        )

        metadata = [
            {
                "split": ex.split,
                "episode_id": ex.episode_id,
                "scenario_seed": ex.scenario_seed,
                "decision_index": ex.decision_index,
                "time": ex.time,
                "source": ex.source,
                "expert_track_id": ex.expert_track_id,
                "expert_action_slot": ex.expert_action_slot,
                "valid_action_ids": ex.valid_action_ids,
                "track_ids": ex.track_ids,
                "num_active": ex.num_active,
                "num_lost": ex.num_lost,
            }
            for ex in self.examples
        ]

        return {
            "global_features": torch.tensor(global_features, dtype=torch.float32),
            "track_features": torch.tensor(track_features, dtype=torch.float32),
            "action_masks": torch.tensor(action_masks, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
            "expert_track_ids": torch.tensor(expert_track_ids, dtype=torch.long),
            "track_ids": torch.tensor(track_ids, dtype=torch.long),
            "scenario_seeds": torch.tensor(scenario_seeds, dtype=torch.long),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
            "decision_indices": torch.tensor(decision_indices, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float32),
            "num_active": torch.tensor(num_active, dtype=torch.long),
            "num_lost": torch.tensor(num_lost, dtype=torch.long),
            "metadata": metadata,
            "extractor": self.extractor_name,
            "max_tracks": int(self.max_tracks),
            "global_dim": int(self.global_dim),
            "track_dim": int(self.track_dim),
            "num_examples": int(len(self.examples)),
        }

    def save_pt(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_tensor_dict(), path)
        print(f"Saved imitation dataset: {path} ({len(self.examples)} examples)")