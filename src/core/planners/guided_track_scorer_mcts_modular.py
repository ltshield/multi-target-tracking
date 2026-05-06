"""Guided realtime MCTS using the modular per-track neural scorer.

This planner keeps MCTS as the decision-maker, but biases MCTS action heuristics
with a learned per-track neural scorer.

Expected path:
    src/core/planners/guided_track_scorer_mcts_modular.py

Expected planner path:
    core.planners.guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner

Required model:
    models/track_scorer.pt

The model should be created by your track-scorer training script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.learning.track_feature_extractors import load_extractor
from core.learning.track_scorer_model import TrackScorerNet
from core.planners.mcts_planner_realtime import MCTSPlanner, PlanningState


@dataclass(slots=True)
class GuidedTrackScorerMCTSPlanner(MCTSPlanner):
    """Realtime MCTS guided by a modular per-track neural scorer.

    The neural network does not replace MCTS. It only biases the heuristic score
    used inside MCTS rollouts/tree expansion.

    Parameters
    ----------
    model_path:
        Path to a trained modular per-track scorer checkpoint.

    prior_weight:
        How much to blend the neural prior into MCTS's existing heuristic score.

        prior_weight = 0.0:
            identical to unguided MCTS heuristic.

        prior_weight = 1.0:
            heavily follows neural scorer.

    prior_temperature:
        Softmax temperature for converting neural scores into probabilities.
        Lower values make the prior more peaked; higher values make it flatter.
    """

    model_path: str = "models/track_scorer.pt"
    prior_weight: float = 0.30
    prior_temperature: float = 1.0

    # Internal fields must be declared because MCTSPlanner uses dataclass(slots=True).
    max_tracks: int = 0
    extractor_name: str = ""
    extractor: Any = None
    model: Any = None

    _last_prior_cache_key: tuple | None = None
    _last_prior_by_track_id: dict[int, float] | None = None

    def __post_init__(self) -> None:
        checkpoint = torch.load(self.model_path, map_location="cpu")

        self.max_tracks = int(checkpoint["max_tracks"])
        self.extractor_name = str(checkpoint["extractor"])
        self.extractor = load_extractor(self.extractor_name)

        self.model = TrackScorerNet(
            global_dim=int(checkpoint["global_dim"]),
            track_dim=int(checkpoint["track_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self._last_prior_cache_key = None
        self._last_prior_by_track_id = {}

    def _heuristic_action_score(
        self,
        state: PlanningState,
        action: int,
        drone,
    ) -> float:
        """Blend base MCTS heuristic with learned neural prior."""

        # Avoid zero-argument super() with dataclass(slots=True) inheritance.
        base_score = MCTSPlanner._heuristic_action_score(
            self,
            state=state,
            action=action,
            drone=drone,
        )

        if not np.isfinite(base_score):
            return base_score

        prior = self._prior_for_action(state=state, action=action)

        # Neural prior is a probability in [0, 1]. Convert it into a modest
        # multiplier so the neural model nudges MCTS rather than replacing it.
        prior_multiplier = 0.50 + prior
        prior_multiplier = float(np.clip(prior_multiplier, 0.25, 1.50))

        guided_score = (
            (1.0 - self.prior_weight) * base_score
            + self.prior_weight * base_score * prior_multiplier
        )

        return float(guided_score)

    def _prior_for_action(self, state: PlanningState, action: int) -> float:
        prior_by_track_id = self._prior_distribution_for_state(state)
        return float(prior_by_track_id.get(int(action), 0.0))

    def _prior_distribution_for_state(
        self,
        state: PlanningState,
    ) -> dict[int, float]:
        """Return neural softmax prior over currently available track IDs."""

        valid_actions = tuple(sorted(int(a) for a in state.available_actions))

        # Cheap cache key to avoid recomputing several times for exactly the
        # same state/action set during one heuristic sweep.
        cache_key = (
            id(state),
            len(state.beliefs),
            valid_actions,
        )

        if (
            self._last_prior_cache_key == cache_key
            and self._last_prior_by_track_id is not None
        ):
            return self._last_prior_by_track_id

        fake_tracks = _FakeTrackSet(
            beliefs=state.beliefs,
            lost_threshold=self.lost_trace_threshold,
        )
        fake_drone = _FakeDrone(
            position=state.drone_position,
            remaining_budget=state.remaining_budget,
        )

        batch = self.extractor.build_batch(
            tracks=fake_tracks,
            drone=fake_drone,
            max_tracks=self.max_tracks,
        )

        valid_action_set = set(valid_actions)

        # Enforce MCTS's current available action list, even if the feature
        # extractor exposes extra slots.
        for i, track_id in enumerate(batch.track_ids):
            if int(track_id) not in valid_action_set:
                batch.action_mask[i] = False

        if not np.any(batch.action_mask):
            self._last_prior_cache_key = cache_key
            self._last_prior_by_track_id = {}
            return {}

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
            ).squeeze(0)

            # Be robust even if the model itself does not fully mask invalid slots.
            scores = scores.masked_fill(~mask_tensor.squeeze(0), -float("inf"))

            temperature = max(1e-6, float(self.prior_temperature))
            probs = torch.softmax(scores / temperature, dim=0).cpu().numpy()

        prior_by_track_id: dict[int, float] = {}

        for i, track_id in enumerate(batch.track_ids):
            track_id = int(track_id)

            if track_id < 0:
                continue

            if track_id in valid_action_set and bool(batch.action_mask[i]):
                prior_by_track_id[track_id] = float(probs[i])

        self._last_prior_cache_key = cache_key
        self._last_prior_by_track_id = prior_by_track_id

        return prior_by_track_id

    def diagnostics(self) -> dict[str, Any]:
        """Add neural-prior diagnostics on top of base MCTS diagnostics."""

        base = MCTSPlanner.diagnostics(self)

        base.update(
            {
                "guided_model_path": self.model_path,
                "guided_extractor": self.extractor_name,
                "guided_prior_weight": float(self.prior_weight),
                "guided_prior_temperature": float(self.prior_temperature),
                "guided_last_prior": dict(self._last_prior_by_track_id or {}),
            }
        )

        return base


class _FakeDrone:
    """Minimal drone-like object for feature extraction from an MCTS state."""

    def __init__(self, position: np.ndarray, remaining_budget: float):
        self.position = np.asarray(position, dtype=float)
        self.remaining_budget = float(remaining_budget)

        # Some feature extractors may expect these fields.
        self.elapsed_time = 0.0
        self.distance_traveled = 0.0


class _FakeTrackSet:
    """Minimal TrackSet-like object for feature extraction from an MCTS state."""

    def __init__(self, beliefs: dict[int, Any], lost_threshold: float):
        self.tracks = [
            _FakeTrack(
                belief=belief,
                lost_threshold=lost_threshold,
            )
            for belief in beliefs.values()
        ]

    @property
    def active_tracks(self) -> list["_FakeTrack"]:
        return [track for track in self.tracks if not track.is_lost]

    def valid_action_ids(self) -> list[int]:
        return [int(track.track_id) for track in self.active_tracks]


class _FakeTrack:
    """Minimal Track-like object matching the extractor's expected interface."""

    def __init__(self, belief: Any, lost_threshold: float):
        self.track_id = int(belief.track_id)
        self.mean = belief.mean
        self.covariance = belief.covariance
        self.existence_probability = float(belief.existence_probability)
        self.time_since_seen = float(belief.time_since_seen)
        self.age = 0.0

        self.is_lost = self.position_variance_trace >= float(lost_threshold)
        self.lost_reason = None
        self.lost_time = None

    @property
    def position(self) -> np.ndarray:
        return self.mean[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[2:4]

    @property
    def position_covariance(self) -> np.ndarray:
        return self.covariance[:2, :2]

    @property
    def position_variance_trace(self) -> float:
        return float(np.trace(self.position_covariance))

    @property
    def position_uncertainty_logdet(self) -> float:
        det = max(float(np.linalg.det(self.position_covariance)), 1e-12)
        return float(np.log(det))