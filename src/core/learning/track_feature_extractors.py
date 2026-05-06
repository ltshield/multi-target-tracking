
"""Modular feature extractors for per-track neural scoring.

The neural planner architecture is:

    extractor(state) -> global_features + per_track_features + action_mask
    shared neural scorer(global_features, track_i_features) -> score_i

To try a different input representation, add a new extractor class here, then
train with:

    --extractor track_feature_extractors.PolarTrackFeatureExtractor

and run with the matching planner checkpoint.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class FeatureBatch:
    global_features: np.ndarray
    track_features: np.ndarray
    action_mask: np.ndarray
    track_ids: list[int]


class TrackFeatureExtractor(Protocol):
    global_dim: int
    track_dim: int

    def build_batch(self, tracks, drone, max_tracks: int) -> FeatureBatch:
        ...


def load_extractor(path: str):
    """Load extractor class from module.Class string or common short name."""

    aliases = {
        "cartesian": "core.learning.track_feature_extractors.CartesianTrackFeatureExtractor",
        "polar": "core.learning.track_feature_extractors.PolarTrackFeatureExtractor",

        # Backward compatibility with old checkpoints before package reorganization.
        "track_feature_extractors.CartesianTrackFeatureExtractor": (
            "core.learning.track_feature_extractors.CartesianTrackFeatureExtractor"
        ),
        "track_feature_extractors.PolarTrackFeatureExtractor": (
            "core.learning.track_feature_extractors.PolarTrackFeatureExtractor"
        ),
    }  
    path = aliases.get(path, path)

    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


class CartesianTrackFeatureExtractor:
    """Default feature representation.

    Global:
        [drone_x, drone_y, remaining_budget,
         num_active, num_lost, mean_active_trace, max_active_trace]

    Track:
        [is_active, existence_probability,
         x, y,
         dx, dy,
         vx, vy,
         sigma_xx, sigma_yy, sigma_xy,
         trace, logdet, time_since_seen, distance_to_drone]
    """

    global_dim = 7
    track_dim = 15

    def __init__(
        self,
        position_scale: float = 1000.0,
        velocity_scale: float = 10.0,
        covariance_scale: float = 250000.0,
        time_scale: float = 900.0,
    ):
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.covariance_scale = covariance_scale
        self.time_scale = time_scale

    def build_batch(self, tracks, drone, max_tracks: int) -> FeatureBatch:
        sorted_tracks = sorted(tracks.tracks, key=lambda t: int(t.track_id))
        active_tracks = [t for t in sorted_tracks if not bool(getattr(t, "is_lost", False))]

        active_traces = [float(t.position_variance_trace) for t in active_tracks]
        mean_active_trace = float(np.mean(active_traces)) if active_traces else 0.0
        max_active_trace = float(np.max(active_traces)) if active_traces else 0.0

        global_features = np.array(
            [
                float(drone.position[0]) / self.position_scale,
                float(drone.position[1]) / self.position_scale,
                float(drone.remaining_budget) / self.time_scale,
                float(len(active_tracks)) / max(1.0, float(max_tracks)),
                float(len(sorted_tracks) - len(active_tracks)) / max(1.0, float(max_tracks)),
                mean_active_trace / self.covariance_scale,
                max_active_trace / self.covariance_scale,
            ],
            dtype=np.float32,
        )

        track_features = np.zeros((max_tracks, self.track_dim), dtype=np.float32)
        action_mask = np.zeros(max_tracks, dtype=bool)
        track_ids = [-1 for _ in range(max_tracks)]

        for slot, track in enumerate(sorted_tracks[:max_tracks]):
            is_lost = bool(getattr(track, "is_lost", False))
            is_active = not is_lost

            dx = float(track.position[0] - drone.position[0])
            dy = float(track.position[1] - drone.position[1])
            dist = float(np.linalg.norm(track.position - drone.position))
            trace = float(track.position_variance_trace)
            logdet = float(track.position_uncertainty_logdet)
            cov = track.position_covariance

            track_features[slot] = np.array(
                [
                    1.0 if is_active else 0.0,
                    float(track.existence_probability),

                    float(track.position[0]) / self.position_scale,
                    float(track.position[1]) / self.position_scale,

                    dx / self.position_scale,
                    dy / self.position_scale,

                    float(track.velocity[0]) / self.velocity_scale,
                    float(track.velocity[1]) / self.velocity_scale,

                    float(cov[0, 0]) / self.covariance_scale,
                    float(cov[1, 1]) / self.covariance_scale,
                    float(cov[0, 1]) / self.covariance_scale,

                    trace / self.covariance_scale,
                    logdet / 50.0,
                    float(track.time_since_seen) / self.time_scale,
                    dist / self.position_scale,
                ],
                dtype=np.float32,
            )

            action_mask[slot] = is_active
            track_ids[slot] = int(track.track_id)

        return FeatureBatch(global_features, track_features, action_mask, track_ids)


class PolarTrackFeatureExtractor:
    """Alternative representation using relative polar geometry.

    This is useful for testing whether orientation/rotation generalization
    improves when the network sees range/bearing instead of absolute dx/dy.

    Global:
        [remaining_budget, num_active, num_lost, mean_active_trace, max_active_trace]

    Track:
        [is_active, existence_probability,
         range, cos_bearing, sin_bearing,
         speed, cos_heading, sin_heading,
         sigma_xx, sigma_yy, sigma_xy,
         trace, logdet, time_since_seen]
    """

    global_dim = 5
    track_dim = 14

    def __init__(
        self,
        position_scale: float = 1000.0,
        velocity_scale: float = 10.0,
        covariance_scale: float = 250000.0,
        time_scale: float = 900.0,
    ):
        self.position_scale = position_scale
        self.velocity_scale = velocity_scale
        self.covariance_scale = covariance_scale
        self.time_scale = time_scale

    def build_batch(self, tracks, drone, max_tracks: int) -> FeatureBatch:
        sorted_tracks = sorted(tracks.tracks, key=lambda t: int(t.track_id))
        active_tracks = [t for t in sorted_tracks if not bool(getattr(t, "is_lost", False))]

        active_traces = [float(t.position_variance_trace) for t in active_tracks]
        mean_active_trace = float(np.mean(active_traces)) if active_traces else 0.0
        max_active_trace = float(np.max(active_traces)) if active_traces else 0.0

        global_features = np.array(
            [
                float(drone.remaining_budget) / self.time_scale,
                float(len(active_tracks)) / max(1.0, float(max_tracks)),
                float(len(sorted_tracks) - len(active_tracks)) / max(1.0, float(max_tracks)),
                mean_active_trace / self.covariance_scale,
                max_active_trace / self.covariance_scale,
            ],
            dtype=np.float32,
        )

        track_features = np.zeros((max_tracks, self.track_dim), dtype=np.float32)
        action_mask = np.zeros(max_tracks, dtype=bool)
        track_ids = [-1 for _ in range(max_tracks)]

        for slot, track in enumerate(sorted_tracks[:max_tracks]):
            is_lost = bool(getattr(track, "is_lost", False))
            is_active = not is_lost

            rel = track.position - drone.position
            dist = float(np.linalg.norm(rel))
            bearing = float(np.arctan2(rel[1], rel[0])) if dist > 1e-12 else 0.0

            vel = track.velocity
            speed = float(np.linalg.norm(vel))
            heading = float(np.arctan2(vel[1], vel[0])) if speed > 1e-12 else 0.0

            trace = float(track.position_variance_trace)
            logdet = float(track.position_uncertainty_logdet)
            cov = track.position_covariance

            track_features[slot] = np.array(
                [
                    1.0 if is_active else 0.0,
                    float(track.existence_probability),
                    dist / self.position_scale,
                    np.cos(bearing),
                    np.sin(bearing),
                    speed / self.velocity_scale,
                    np.cos(heading),
                    np.sin(heading),
                    float(cov[0, 0]) / self.covariance_scale,
                    float(cov[1, 1]) / self.covariance_scale,
                    float(cov[0, 1]) / self.covariance_scale,
                    trace / self.covariance_scale,
                    logdet / 50.0,
                    float(track.time_since_seen) / self.time_scale,
                ],
                dtype=np.float32,
            )

            action_mask[slot] = is_active
            track_ids[slot] = int(track.track_id)

        return FeatureBatch(global_features, track_features, action_mask, track_ids)


def track_id_to_slot(track_ids: list[int], track_id: int) -> int:
    for i, tid in enumerate(track_ids):
        if int(tid) == int(track_id):
            return i
    raise ValueError(f"Track ID {track_id} not found in current track batch.")


def slot_to_track_id(track_ids: list[int], slot: int) -> int:
    if slot < 0 or slot >= len(track_ids):
        raise ValueError(f"Invalid slot {slot}.")
    tid = int(track_ids[slot])
    if tid < 0:
        raise ValueError(f"Slot {slot} is padding, not a real track.")
    return tid
