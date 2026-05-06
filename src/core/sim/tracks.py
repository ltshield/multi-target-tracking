"""Track/belief representation for multi-target tracking.

This file represents the drone/tracker's belief about targets, not the true
ground-truth targets.

State convention:
    mean = [x, y, vx, vy]

Belief propagation:
    mean = F @ mean
    covariance = F @ covariance @ F.T + Q

Measurement update:
    z = [measured_x, measured_y]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ArrayLike2D = Sequence[float] | np.ndarray
ArrayLike4D = Sequence[float] | np.ndarray


def constant_velocity_F(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix."""

    if dt <= 0:
        raise ValueError("dt must be positive.")

    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def constant_velocity_Q(dt: float, acceleration_noise_std: float) -> np.ndarray:
    """Process noise matrix for a 2D constant-velocity model.

    acceleration_noise_std controls how quickly uncertainty grows when a target
    is not seen.

    Larger values mean the target's motion is less predictable.
    """

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if acceleration_noise_std < 0:
        raise ValueError("acceleration_noise_std must be nonnegative.")

    q = acceleration_noise_std**2

    return q * np.array(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
            [dt**3 / 2.0, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2.0, 0.0, dt**2],
        ],
        dtype=float,
    )


@dataclass(slots=True)
class Track:
    """LMB-like track representation.

    This is the belief state used by the planner.

    Parameters
    ----------
    track_id:
        Internal target/track ID.

    mean:
        Estimated state [x, y, vx, vy].

    covariance:
        4x4 state covariance.

    existence_probability:
        Probability that this track corresponds to a real target.

    time_since_seen:
        Seconds since this track was last detected.

    age:
        Total age of the track in seconds.
    """

    track_id: int
    mean: ArrayLike4D
    covariance: np.ndarray
    existence_probability: float = 1.0
    time_since_seen: float = 0.0
    age: float = 0.0

    is_lost: bool = False
    lost_reason: str | None = None
    lost_time: float | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)

        if self.mean.shape != (4,):
            raise ValueError(f"mean must have shape (4,), got {self.mean.shape}.")
        if self.covariance.shape != (4, 4):
            raise ValueError(
                f"covariance must have shape (4, 4), got {self.covariance.shape}."
            )
        if not 0.0 <= self.existence_probability <= 1.0:
            raise ValueError("existence_probability must be in [0, 1].")

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
        """sigma_xx + sigma_yy."""

        return float(np.trace(self.position_covariance))

    @property
    def position_uncertainty_det(self) -> float:
        """determinant of the 2D position covariance."""

        return float(np.linalg.det(self.position_covariance))

    @property
    def position_uncertainty_logdet(self) -> float:
        """log(det(position covariance)), with numerical protection."""

        det = max(self.position_uncertainty_det, 1e-12)
        return float(np.log(det))
    
    def reported_position_variance_trace(self, lost_target_penalty: float) -> float:
        """Return the uncertainty value that should be reported in metrics.

        Active tracks report their actual covariance trace.
        Lost tracks report a fixed max penalty.
        """
        if self.is_lost:
            return float(lost_target_penalty)

        return float(self.position_variance_trace)

    def copy(self) -> "Track":
        copied = Track(
            track_id=self.track_id,
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            existence_probability=self.existence_probability,
            time_since_seen=self.time_since_seen,
            age=self.age,
            is_lost=self.is_lost,
            lost_reason=self.lost_reason,
            lost_time=self.lost_time,
        )
        return copied

    def predict_constant_velocity(
        self,
        dt: float,
        acceleration_noise_std: float = 1.0,
        existence_decay_rate: float = 0.0,
    ) -> None:
        """Predict the track forward using Kalman filter dynamics.

        This is the step where uncertainty grows.

        mean = F mean
        P = F P F.T + Q
        """

        F = constant_velocity_F(dt)
        Q = constant_velocity_Q(dt, acceleration_noise_std)

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q

        # Numerical cleanup: covariance should stay symmetric.
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        self.time_since_seen += dt
        self.age += dt

        if existence_decay_rate > 0.0:
            self.existence_probability *= np.exp(-existence_decay_rate * dt)
            self.existence_probability = float(
                np.clip(self.existence_probability, 0.0, 1.0)
            )

    def update_with_position_measurement(
        self,
        measured_position: ArrayLike2D,
        measurement_noise_std: float = 10.0,
    ) -> None:
        """Kalman update using a noisy position measurement z = [x, y].

        This is the step where uncertainty shrinks after detection.
        """

        z = np.asarray(measured_position, dtype=float)

        if z.shape != (2,):
            raise ValueError(f"measured_position must have shape (2,), got {z.shape}.")
        if measurement_noise_std <= 0.0:
            raise ValueError("measurement_noise_std must be positive.")

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        R = measurement_noise_std**2 * np.eye(2)

        innovation = z - H @ self.mean
        innovation_covariance = H @ self.covariance @ H.T + R

        kalman_gain = (
            self.covariance
            @ H.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.mean = self.mean + kalman_gain @ innovation

        I = np.eye(4)

        # Joseph form is more numerically stable than P = (I - KH)P.
        self.covariance = (
            (I - kalman_gain @ H)
            @ self.covariance
            @ (I - kalman_gain @ H).T
            + kalman_gain @ R @ kalman_gain.T
        )

        self.covariance = 0.5 * (self.covariance + self.covariance.T)

        self.time_since_seen = 0.0
        self.existence_probability = 1.0

    def standardized_belief_vector(self) -> np.ndarray:
        """Return the compact vector useful for planners or neural networks.

        Format:
            [r, x, y, vx, vy, sigma_xx, sigma_yy, time_since_seen]
        """

        return np.array(
            [
                self.existence_probability,
                self.mean[0],
                self.mean[1],
                self.mean[2],
                self.mean[3],
                self.covariance[0, 0],
                self.covariance[1, 1],
                self.time_since_seen,
            ],
            dtype=float,
        )
    
    def mark_lost(self, current_time: float, reason: str) -> None:
        """Mark this track as permanently lost.

        Lost tracks are no longer selectable by planners, no longer updated by
        detections, and no longer propagated by the Kalman prediction step.
        """
        if self.is_lost:
            return

        self.is_lost = True
        self.lost_reason = reason
        self.lost_time = float(current_time)
        self.existence_probability = 0.0
        
    def check_lost(
        self,
        current_time: float,
        max_position_trace: float | None = None,
        max_position_logdet: float | None = None,
        max_time_since_seen: float | None = None,
        min_existence_probability: float | None = None,
    ) -> bool:
        """Mark this track as permanently lost if it exceeds loss thresholds.

        Returns True if the track is lost after this check.
        """

        if self.is_lost:
            return True

        if (
            max_position_trace is not None
            and self.position_variance_trace >= max_position_trace
        ):
            self.mark_lost(
                current_time=current_time,
                reason="position_trace_threshold",
            )
            return True

        if (
            max_position_logdet is not None
            and self.position_uncertainty_logdet >= max_position_logdet
        ):
            self.mark_lost(
                current_time=current_time,
                reason="position_logdet_threshold",
            )
            return True

        if (
            max_time_since_seen is not None
            and self.time_since_seen >= max_time_since_seen
        ):
            self.mark_lost(
                current_time=current_time,
                reason="time_since_seen_threshold",
            )
            return True

        if (
            min_existence_probability is not None
            and self.existence_probability <= min_existence_probability
        ):
            self.mark_lost(
                current_time=current_time,
                reason="existence_probability_threshold",
            )
            return True

        return False


@dataclass(slots=True)
class TrackSet:
    """Collection of tracks/beliefs."""

    tracks: list[Track]

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if not track.is_lost]

    def valid_action_ids(self) -> list[int]:
        """Track IDs that planners are allowed to select."""
        return [int(track.track_id) for track in self.active_tracks]

    def __post_init__(self) -> None:
        ids = [track.track_id for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("Track IDs must be unique.")

    def __iter__(self):
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, track_id: int) -> Track:
        for track in self.tracks:
            if track.track_id == track_id:
                return track

        raise KeyError(f"No track with ID {track_id}.")

    def copy(self) -> "TrackSet":
        return TrackSet([track.copy() for track in self.tracks])

    def predict_all(
    self,
    dt: float,
    acceleration_noise_std: float = 1.0,
    existence_decay_rate: float = 0.0,
    ) -> None:
        for track in self.tracks:
            if track.is_lost:
                continue

            track.predict_constant_velocity(
                dt=dt,
                acceleration_noise_std=acceleration_noise_std,
                existence_decay_rate=existence_decay_rate,
            )

    def update_from_detections(
        self,
        detections,
        measurement_noise_std: float = 10.0,
    ) -> None:
        for detection in detections:
            try:
                track = self[detection.target_id]
            except KeyError:
                continue

            # For this use case, once lost, irretrievable.
            if track.is_lost:
                continue

            track.update_with_position_measurement(
                measured_position=detection.position,
                measurement_noise_std=measurement_noise_std,
            )

    def total_position_trace(self, active_only: bool = True) -> float:
        selected = self.active_tracks if active_only else self.tracks
        return float(sum(track.position_variance_trace for track in selected))

    def total_position_logdet(self, active_only: bool = True) -> float:
        selected = self.active_tracks if active_only else self.tracks
        return float(sum(track.position_uncertainty_logdet for track in selected))

    def num_lost(self) -> int:
        return sum(1 for track in self.tracks if track.is_lost)

    def num_active(self) -> int:
        return sum(1 for track in self.tracks if not track.is_lost)
    
    def check_lost_tracks(
        self,
        current_time: float,
        max_position_trace: float | None = None,
        max_position_logdet: float | None = None,
        max_time_since_seen: float | None = None,
        min_existence_probability: float | None = None,
    ) -> list[int]:
        """Mark tracks as lost if they exceed thresholds.

        Returns the IDs of newly lost tracks.
        """

        newly_lost = []

        for track in self.tracks:
            was_lost = track.is_lost

            track.check_lost(
                current_time=current_time,
                max_position_trace=max_position_trace,
                max_position_logdet=max_position_logdet,
                max_time_since_seen=max_time_since_seen,
                min_existence_probability=min_existence_probability,
            )

            if not was_lost and track.is_lost:
                newly_lost.append(track.track_id)

        return newly_lost