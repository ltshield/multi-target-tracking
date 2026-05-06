"""Simple 2D drone model for multi-target tracking simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


ArrayLike2D = Sequence[float] | np.ndarray


@dataclass(slots=True)
class Detection:
    """A simple detection returned by the drone sensor."""

    target_id: int
    position: np.ndarray
    distance: float
    time: float


@dataclass(slots=True)
class Drone:
    """Simple constant-speed 2D UAV.

    The drone is intentionally simple:
    - It has a 2D position.
    - It flies in straight lines at constant speed.
    - It has a circular sensor footprint.
    - It has a finite mission budget.
    """

    position: ArrayLike2D
    speed: float
    sensor_range: float
    budget: float

    detection_probability: float = 1.0
    false_alarm_rate: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    elapsed_time: float = 0.0
    distance_traveled: float = 0.0

    def __post_init__(self) -> None:
        self.position = self._as_position(self.position)

        if self.speed <= 0:
            raise ValueError("speed must be positive.")
        if self.sensor_range < 0:
            raise ValueError("sensor_range must be nonnegative.")
        if self.budget < 0:
            raise ValueError("budget must be nonnegative.")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1].")

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget - self.elapsed_time)

    @property
    def is_active(self) -> bool:
        return self.remaining_budget > 0.0

    @property
    def sensor_width(self) -> float:
        """Sensor diameter, matching the paper's notation w."""

        return 2.0 * self.sensor_range

    def copy(self) -> "Drone":
        copied = Drone(
            position=self.position.copy(),
            speed=self.speed,
            sensor_range=self.sensor_range,
            budget=self.budget,
            detection_probability=self.detection_probability,
            false_alarm_rate=self.false_alarm_rate,
            rng=self.rng,
        )
        copied.elapsed_time = self.elapsed_time
        copied.distance_traveled = self.distance_traveled
        return copied

    def distance_to(self, waypoint: ArrayLike2D) -> float:
        waypoint = self._as_position(waypoint)
        return float(np.linalg.norm(waypoint - self.position))

    def time_to(self, waypoint: ArrayLike2D) -> float:
        return self.distance_to(waypoint) / self.speed

    def step_toward(self, waypoint: ArrayLike2D, dt: float) -> float:
        """Move toward a waypoint for at most dt seconds.

        Returns the amount of time actually consumed.
        """

        if dt < 0:
            raise ValueError("dt must be nonnegative.")
        if dt == 0.0 or not self.is_active:
            return 0.0

        waypoint = self._as_position(waypoint)
        to_goal = waypoint - self.position
        distance = float(np.linalg.norm(to_goal))

        if distance == 0.0:
            consumed = min(dt, self.remaining_budget)
            self.elapsed_time += consumed
            return consumed

        available_time = min(dt, self.remaining_budget)
        max_distance = self.speed * available_time
        traveled = min(distance, max_distance)

        direction = to_goal / distance
        self.position = self.position + direction * traveled

        consumed = traveled / self.speed
        self.elapsed_time += consumed
        self.distance_traveled += traveled

        return consumed

    def fly_to(self, waypoint: ArrayLike2D, max_duration: Optional[float] = None) -> float:
        """Fly toward a waypoint until arrival, timeout, or budget depletion."""

        duration = self.time_to(waypoint)

        if max_duration is not None:
            if max_duration < 0:
                raise ValueError("max_duration must be nonnegative.")
            duration = min(duration, max_duration)

        return self.step_toward(waypoint, duration)

    def can_detect(self, target_position: ArrayLike2D) -> bool:
        return self.distance_to(target_position) <= self.sensor_range

    def detect_targets(
        self,
        target_positions: dict[int, ArrayLike2D] | Iterable[tuple[int, ArrayLike2D]],
    ) -> list[Detection]:
        """Detect targets inside the circular sensor footprint.

        For early simulation work, this method returns target IDs.
        Later, an LMB-style tracker can consume anonymous raw detections instead.
        """

        items = (
            target_positions.items()
            if isinstance(target_positions, dict)
            else target_positions
        )

        detections: list[Detection] = []

        for target_id, target_position in items:
            target_position = self._as_position(target_position)
            distance = float(np.linalg.norm(target_position - self.position))

            inside_sensor = distance <= self.sensor_range
            detected = inside_sensor and self.rng.random() <= self.detection_probability

            if detected:
                detections.append(
                    Detection(
                        target_id=int(target_id),
                        position=target_position.copy(),
                        distance=distance,
                        time=self.elapsed_time,
                    )
                )

        return detections

    def state_vector(self) -> np.ndarray:
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.speed,
                self.sensor_range,
                self.elapsed_time,
                self.remaining_budget,
                self.distance_traveled,
            ],
            dtype=float,
        )

    @classmethod
    def from_config(cls, config: dict) -> "Drone":
        drone_config = config.get("drone", config)

        return cls(
            position=drone_config.get("initial_position", [0.0, 0.0]),
            speed=float(drone_config["speed"]),
            sensor_range=float(drone_config["sensor_range"]),
            budget=float(drone_config["budget"]),
            detection_probability=float(
                drone_config.get("detection_probability", 1.0)
            ),
            false_alarm_rate=float(drone_config.get("false_alarm_rate", 0.0)),
        )

    @staticmethod
    def _as_position(value: ArrayLike2D) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape != (2,):
            raise ValueError(f"Expected 2D position with shape (2,), got {arr.shape}.")

        return arr