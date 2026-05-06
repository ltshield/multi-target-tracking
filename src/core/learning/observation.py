
"""Observation utilities for neural planners.

The neural policy sees a fixed-length vector:
    [drone_x, drone_y, remaining_budget,
     track_1_features..., track_K_features...]

Each track feature block:
    [is_active, r, dx, dy, vx, vy, sigma_xx, sigma_yy,
     cov_trace, logdet, time_since_seen, is_lost]

Targets/tracks are sorted by track_id. If there are fewer than max_targets,
remaining slots are zero-padded.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


FEATURES_PER_TRACK = 12
DRONE_FEATURES = 3


def observation_dim(max_targets: int) -> int:
    return DRONE_FEATURES + max_targets * FEATURES_PER_TRACK


def build_observation_from_tracks(
    tracks,
    drone,
    max_targets: int,
    normalize: bool = True,
) -> np.ndarray:
    active_tracks = sorted(tracks.tracks, key=lambda t: t.track_id)
    obs = np.zeros(observation_dim(max_targets), dtype=np.float32)

    scale_pos = 1000.0 if normalize else 1.0
    scale_vel = 10.0 if normalize else 1.0
    scale_cov = 250000.0 if normalize else 1.0
    scale_time = 900.0 if normalize else 1.0

    obs[0] = float(drone.position[0]) / scale_pos
    obs[1] = float(drone.position[1]) / scale_pos
    obs[2] = float(drone.remaining_budget) / scale_time

    offset = DRONE_FEATURES
    for slot, track in enumerate(active_tracks[:max_targets]):
        base = offset + slot * FEATURES_PER_TRACK
        is_lost = bool(getattr(track, "is_lost", False))
        is_active = not is_lost

        dx = float(track.position[0] - drone.position[0])
        dy = float(track.position[1] - drone.position[1])
        trace = float(track.position_variance_trace)
        logdet = float(track.position_uncertainty_logdet)

        obs[base + 0] = 1.0 if is_active else 0.0
        obs[base + 1] = float(track.existence_probability)
        obs[base + 2] = dx / scale_pos
        obs[base + 3] = dy / scale_pos
        obs[base + 4] = float(track.velocity[0]) / scale_vel
        obs[base + 5] = float(track.velocity[1]) / scale_vel
        obs[base + 6] = float(track.covariance[0, 0]) / scale_cov
        obs[base + 7] = float(track.covariance[1, 1]) / scale_cov
        obs[base + 8] = trace / scale_cov
        obs[base + 9] = logdet / 50.0
        obs[base + 10] = float(track.time_since_seen) / scale_time
        obs[base + 11] = 1.0 if is_lost else 0.0

    return obs


def active_action_mask(tracks, max_targets: int) -> np.ndarray:
    """Boolean mask over action slots, not track IDs."""
    sorted_tracks = sorted(tracks.tracks, key=lambda t: t.track_id)
    mask = np.zeros(max_targets, dtype=bool)
    for i, track in enumerate(sorted_tracks[:max_targets]):
        mask[i] = not bool(getattr(track, "is_lost", False))
    return mask


def action_index_to_track_id(tracks, action_index: int, max_targets: int) -> int:
    sorted_tracks = sorted(tracks.tracks, key=lambda t: t.track_id)
    if action_index < 0 or action_index >= min(max_targets, len(sorted_tracks)):
        raise ValueError(f"Invalid action index {action_index}.")
    track = sorted_tracks[action_index]
    if bool(getattr(track, "is_lost", False)):
        raise ValueError(f"Action index {action_index} maps to lost track {track.track_id}.")
    return int(track.track_id)


def track_id_to_action_index(tracks, track_id: int, max_targets: int) -> int:
    sorted_tracks = sorted(tracks.tracks, key=lambda t: t.track_id)
    for i, track in enumerate(sorted_tracks[:max_targets]):
        if int(track.track_id) == int(track_id):
            return i
    raise ValueError(f"Track ID {track_id} is not present in first {max_targets} action slots.")


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    scores = np.asarray(logits, dtype=float).copy()
    scores[~mask] = -np.inf
    if not np.any(mask):
        return int(np.argmax(logits))
    return int(np.argmax(scores))
