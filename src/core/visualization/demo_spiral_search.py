"""Small runnable demo for the elliptic shifting spiral search."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core.sim.coverage_spiral import EllipticShiftingSpiralPlanner, execute_spiral_search
from core.sim.drone import Drone
from core.sim.tracks import Track


def main() -> None:
    dt = 1.0

    drone = Drone(
        position=[0.0, 0.0],
        speed=30.0,
        sensor_range=50.0,
        budget=300.0,
        detection_probability=1.0,
    )

    # Target belief:
    # mean = [x, y, vx, vy]
    track = Track(
        track_id=1,
        mean=np.array([500.0, 250.0, 2.0, -1.0]),
        covariance=np.array(
            [
                [9000.0, 2500.0, 0.0, 0.0],
                [2500.0, 4000.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=float,
        ),
        existence_probability=1.0,
    )

    # True target position for this toy demo.
    # In a real simulation, this would be updated by target dynamics.
    true_target_positions = {
        1: np.array([560.0, 235.0], dtype=float),
    }

    detections = execute_spiral_search(
        drone=drone,
        track=track,
        target_positions=true_target_positions,
        max_search_time=120.0,
        dt=dt,
        covariance_scale=2.0,
    )

    print(f"Detections: {detections}")
    print(f"Drone final position: {drone.position}")
    print(f"Elapsed time: {drone.elapsed_time:.2f}")
    print(f"Distance traveled: {drone.distance_traveled:.2f}")

    # Optional plot of the planned spiral waypoints.
    planner = EllipticShiftingSpiralPlanner.from_track(
        track=track,
        sensor_width=drone.sensor_width,
        dt=dt,
        covariance_scale=2.0,
    )

    waypoints = np.array(
        planner.generate_waypoints(
            drone_speed=drone.speed,
            max_search_time=120.0,
        )
    )

    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], marker=".")
    plt.scatter([track.position[0]], [track.position[1]], label="Belief center")
    plt.scatter(
        [true_target_positions[1][0]],
        [true_target_positions[1][1]],
        label="True target",
    )
    plt.scatter([0.0], [0.0], label="Drone start")
    plt.axis("equal")
    plt.legend()
    plt.title("Elliptic Shifting Spiral Search")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()