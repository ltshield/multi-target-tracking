"""Elliptic shifting spiral coverage planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


ArrayLike2D = Sequence[float] | np.ndarray


class TrackLike(Protocol):
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(slots=True)
class SpiralState:
    theta: float = 0.0
    elapsed: float = 0.0


@dataclass(slots=True)
class EllipticShiftingSpiralPlanner:
    """Generate elliptic shifting spiral waypoints.

    This planner does not move the drone directly. It only generates waypoints.

    The drone should:
    1. Fly to the target belief center.
    2. Ask this planner for the next waypoint.
    3. Step toward that waypoint.
    4. Check for detections.
    """

    center0: np.ndarray
    velocity: np.ndarray
    position_covariance: np.ndarray
    sensor_width: float
    dt: float

    covariance_scale: float = 2.0
    min_axis_length: float = 1e-6

    # Internal ellipse / spiral parameters computed in __post_init__.
    A: float = field(init=False)
    B: float = field(init=False)
    R: np.ndarray = field(init=False)
    a: float = field(init=False)
    b: float = field(init=False)

    def __post_init__(self) -> None:
        self.center0 = self._as_vector(self.center0, 2, "center0")
        self.velocity = self._as_vector(self.velocity, 2, "velocity")
        self.position_covariance = self._as_matrix(
            self.position_covariance, (2, 2), "position_covariance"
        )

        if self.sensor_width <= 0.0:
            raise ValueError("sensor_width must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.covariance_scale <= 0.0:
            raise ValueError("covariance_scale must be positive.")

        self._setup_ellipse()

        # Archimedean spiral: r(theta) = a + b theta.
        # With this b, each full loop expands by about one sensor width along
        # the larger ellipse axis.
        self.a = 0.0
        self.b = self.sensor_width / (2.0 * np.pi * max(self.A, self.B))

    @classmethod
    def from_track(
        cls,
        track: TrackLike,
        sensor_width: float,
        dt: float,
        covariance_scale: float = 2.0,
    ) -> "EllipticShiftingSpiralPlanner":
        mean = cls._as_vector(track.mean, 4, "track.mean")
        covariance = cls._as_matrix(track.covariance, (4, 4), "track.covariance")

        return cls(
            center0=mean[:2],
            velocity=mean[2:4],
            position_covariance=covariance[:2, :2],
            sensor_width=sensor_width,
            dt=dt,
            covariance_scale=covariance_scale,
        )

    def initial_state(self) -> SpiralState:
        return SpiralState(theta=0.0, elapsed=0.0)

    def center_at(self, elapsed: float) -> np.ndarray:
        """Predicted target belief center after elapsed seconds."""

        return self.center0 + self.velocity * elapsed

    def local_spiral_point(self, theta: float) -> np.ndarray:
        """Spiral point in the ellipse's local coordinate frame."""

        radius = self.a + self.b * theta

        return np.array(
            [
                radius * self.A * np.cos(theta),
                radius * self.B * np.sin(theta),
            ],
            dtype=float,
        )

    def waypoint_at(self, theta: float, elapsed: float) -> np.ndarray:
        """World-frame waypoint at spiral phase theta and time elapsed."""

        local_point = self.local_spiral_point(theta)
        rotated_point = self.R @ local_point
        shifted_center = self.center_at(elapsed)

        return shifted_center + rotated_point

    def next_theta(
        self,
        theta: float,
        elapsed: float,
        drone_speed: float,
        max_expand_iterations: int = 40,
        bisection_iterations: int = 50,
    ) -> float:
        """Find the next theta so the drone moves about speed * dt.

        We solve:

            ||p(theta_next, t + dt) - p(theta, t)|| = drone_speed * dt

        using bracket expansion plus bisection.
        """

        if drone_speed <= 0.0:
            raise ValueError("drone_speed must be positive.")

        current_position = self.waypoint_at(theta, elapsed)
        target_step_distance = drone_speed * self.dt
        next_elapsed = elapsed + self.dt

        def distance_error(candidate_theta: float) -> float:
            candidate_position = self.waypoint_at(candidate_theta, next_elapsed)
            distance = float(np.linalg.norm(candidate_position - current_position))
            return distance - target_step_distance

        low = theta
        high = theta + 0.1

        for _ in range(max_expand_iterations):
            if distance_error(high) >= 0.0:
                break
            high = theta + 2.0 * (high - theta)
        else:
            raise RuntimeError("Could not bracket next spiral theta.")

        for _ in range(bisection_iterations):
            mid = 0.5 * (low + high)

            if distance_error(mid) >= 0.0:
                high = mid
            else:
                low = mid

        return high

    def next_waypoint(
        self,
        state: SpiralState,
        drone_speed: float,
    ) -> tuple[np.ndarray, SpiralState]:
        theta_next = self.next_theta(
            theta=state.theta,
            elapsed=state.elapsed,
            drone_speed=drone_speed,
        )

        elapsed_next = state.elapsed + self.dt
        waypoint = self.waypoint_at(theta_next, elapsed_next)

        return waypoint, SpiralState(theta=theta_next, elapsed=elapsed_next)

    def generate_waypoints(
        self,
        drone_speed: float,
        max_search_time: float,
    ) -> list[np.ndarray]:
        """Generate waypoints for plotting/debugging."""

        if max_search_time < 0.0:
            raise ValueError("max_search_time must be nonnegative.")

        state = self.initial_state()
        waypoints = [self.waypoint_at(state.theta, state.elapsed)]

        while state.elapsed + self.dt <= max_search_time:
            waypoint, state = self.next_waypoint(state, drone_speed)
            waypoints.append(waypoint)

        return waypoints

    def _setup_ellipse(self) -> None:
        """Compute ellipse axes and orientation from covariance."""

        cov = 0.5 * (self.position_covariance + self.position_covariance.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by largest eigenvalue first.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues = np.maximum(eigenvalues, self.min_axis_length**2)

        axes = self.covariance_scale * np.sqrt(eigenvalues)

        self.A = float(max(axes[0], self.min_axis_length))
        self.B = float(max(axes[1], self.min_axis_length))
        self.R = eigenvectors

    @staticmethod
    def _as_vector(value, length: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape != (length,):
            raise ValueError(f"{name} must have shape ({length},), got {arr.shape}.")

        return arr

    @staticmethod
    def _as_matrix(value, shape: tuple[int, int], name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")

        return arr


def execute_spiral_search(
    drone,
    track: TrackLike,
    target_positions: dict[int, ArrayLike2D],
    max_search_time: float,
    dt: float,
    covariance_scale: float = 2.0,
):
    """Send drone to track center, then execute elliptic shifting spiral search.

    Returns
    -------
    list[Detection]
        First non-empty detection list, or empty list if nothing is detected.
    """

    planner = EllipticShiftingSpiralPlanner.from_track(
        track=track,
        sensor_width=drone.sensor_width,
        dt=dt,
        covariance_scale=covariance_scale,
    )

    # Step 1: travel to the target's current belief center / last-known position.
    drone.fly_to(planner.center0)

    state = planner.initial_state()
    search_elapsed = 0.0

    while drone.is_active and search_elapsed < max_search_time:
        waypoint, state = planner.next_waypoint(state, drone.speed)

        consumed = drone.step_toward(waypoint, dt)
        search_elapsed += consumed

        detections = drone.detect_targets(target_positions)
        if detections:
            return detections

        if consumed <= 0.0:
            break

    return []