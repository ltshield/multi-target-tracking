"""Guided realtime MCTS using the modular per-track neural scorer.

Drop this file into:

    src/core/guided_track_scorer_mcts_modular.py

Then compare with:

    python src/core/compare_planners.py ^
      --config configs/basic_3target.yaml ^
      --output-dir runs/track_scorer_compare ^
      --seeds 7 8 9 10 11 ^
      --planners random=random_planner.RandomPlanner greedy=greedy_planners.GreedyDistanceAwarePlanner mcts=mcts_planner_realtime.MCTSPlanner scorer=track_scorer_planner_modular.TrackScorerPlanner guided=guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner

This planner keeps MCTS as the decision-maker, but biases MCTS rollout/action
heuristics with the learned per-track scorer.

Required files in src/core:
    mcts_planner_realtime.py
    track_feature_extractors.py
    track_scorer_model.py

Required model:
    models/track_scorer.pt

That model should be created by train_track_scorer_randomized.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.planners.mcts_planner_realtime import MCTSPlanner, PlanningState
from core.learning.track_feature_extractors import load_extractor
from core.learning.track_scorer_model import TrackScorerNet


@dataclass(slots=True)
class GuidedTrackScorerMCTSPlanner(MCTSPlanner):
    """Realtime MCTS guided by a modular per-track neural scorer.

    The neural network does not replace MCTS. It only biases the heuristic score
    used inside MCTS rollouts/tree expansion. This lets MCTS still reason about
    future uncertainty, find/miss outcomes, and lost-target penalties.

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

        A good starting range is 0.15 to 0.40.

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

    def _heuristic_action_score(self, state: PlanningState, action: int, drone) -> float:
        """Blend base MCTS heuristic with learned neural prior."""

        # Avoid zero-argument super() with dataclass(slots=True) inheritance.
        base_score = MCTSPlanner._heuristic_action_score(self, state, action, drone)

        if not np.isfinite(base_score):
            return base_score

        prior = self._prior_for_action(state=state, action=action)

        # The base score is in arbitrary heuristic units. We convert the neural
        # probability into a multiplier so it nudges MCTS rather than replacing it.
        #
        # Example with prior_weight=0.30:
        #   prior near 1.0   -> score increases
        #   prior near 0.0   -> score decreases
        #   prior near uniform -> modest effect
        #
        # Clamp to keep the learned model from completely overwhelming MCTS if
        # the model is still weak.
        prior_multiplier = 0.50 + prior
        prior_multiplier = float(np.clip(prior_multiplier, 0.25, 1.50))

        guided_score = (1.0 - self.prior_weight) * base_score + (
            self.prior_weight * base_score * prior_multiplier
        )

        return float(guided_score)

    def _prior_for_action(self, state: PlanningState, action: int) -> float:
        prior_by_track_id = self._prior_distribution_for_state(state)
        return float(prior_by_track_id.get(int(action), 0.0))

    def _prior_distribution_for_state(self, state: PlanningState) -> dict[int, float]:
        """Return neural softmax prior over active track IDs."""

        # Cheap cache key: enough to avoid recomputing several times for exactly
        # the same state object/shape during a heuristic sweep.
        cache_key = (
            id(state),
            len(state.beliefs),
            tuple(sorted(state.available_actions)),
        )
        if self._last_prior_cache_key == cache_key and self._last_prior_by_track_id is not None:
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

        valid_actions = set(int(a) for a in state.available_actions)

        for i, track_id in enumerate(batch.track_ids):
            if int(track_id) not in valid_actions:
                batch.action_mask[i] = False

        if not np.any(batch.action_mask):
            self._last_prior_cache_key = cache_key
            self._last_prior_by_track_id = {}
            return {}

        global_tensor = torch.tensor(batch.global_features, dtype=torch.float32).unsqueeze(0)
        track_tensor = torch.tensor(batch.track_features, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(batch.action_mask, dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            scores = self.model(global_tensor, track_tensor, mask_tensor).squeeze(0)

            temperature = max(1e-6, float(self.prior_temperature))
            probs = torch.softmax(scores / temperature, dim=0).cpu().numpy()

        prior_by_track_id = {
            int(track_id): float(probs[i])
            for i, track_id in enumerate(batch.track_ids)
            if int(track_id) >= 0 and bool(batch.action_mask[i])
        }

        self._last_prior_cache_key = cache_key
        self._last_prior_by_track_id = prior_by_track_id
        return prior_by_track_id

    def diagnostics(self) -> dict[str, Any]:
        """Add neural-prior diagnostics on top of base MCTS diagnostics."""

        # Avoid zero-argument super() here because dataclass(slots=True) inheritance
        # can behave oddly after module reloads/copying on Windows/Python 3.13.
        # Calling the base method directly is equivalent and more robust.
        base = MCTSPlanner.diagnostics(self)
        base.update(
            {
                "guided_model_path": self.model_path,
                "guided_extractor": getattr(self, "extractor_name", None),
                "guided_prior_weight": float(self.prior_weight),
                "guided_prior_temperature": float(self.prior_temperature),
                "guided_last_prior": dict(getattr(self, "_last_prior_by_track_id", {})),
            }
        )
        return base


class _FakeDrone:
    """Minimal drone-like object for feature extraction from an MCTS state."""

    def __init__(self, position: np.ndarray, remaining_budget: float):
        self.position = np.asarray(position, dtype=float)
        self.remaining_budget = float(remaining_budget)


class _FakeTrackSet:
    """Minimal TrackSet-like object for feature extraction from an MCTS state."""

    def __init__(self, beliefs: dict[int, Any], lost_threshold: float):
        self.tracks = [
            _FakeTrack(belief=belief, lost_threshold=lost_threshold)
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
        self.is_lost = self.position_variance_trace >= float(lost_threshold)

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
        return float(np.log(max(np.linalg.det(self.position_covariance), 1e-12)))
