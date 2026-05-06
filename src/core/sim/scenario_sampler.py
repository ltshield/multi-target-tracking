
"""Random scenario generation for training data.

A single hand-written YAML is useful for debugging, but neural training needs
many randomized scenarios so the network does not overfit to one geometry.

This sampler keeps drone parameters fixed by default while randomizing:
- number of targets
- target initial positions/orientations
- target speeds/headings
- process noise
- optionally initial belief noise

It returns a SimConfig compatible with your existing simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from core.sim.simulate_run import SimConfig


@dataclass(slots=True)
class ScenarioSamplerConfig:
    min_targets: int = 2
    max_targets: int = 6

    # Initial target position sampling in polar coordinates around the drone.
    min_radius: float = 250.0
    max_radius: float = 900.0

    # Target speed distribution.
    min_target_speed: float = 0.25
    max_target_speed: float = 3.0

    # Velocity process noise for true targets.
    min_velocity_noise_std: float = 0.0
    max_velocity_noise_std: float = 0.006

    # Optionally perturb the initial belief estimate away from truth.
    initial_belief_position_noise_std: float = 0.0
    initial_belief_velocity_noise_std: float = 0.0


def sample_random_scenario(
    base_config: SimConfig,
    rng: np.random.Generator,
    sampler_config: ScenarioSamplerConfig,
    seed: int,
) -> SimConfig:
    """Create one randomized SimConfig."""

    config = copy.deepcopy(base_config)
    config.seed = int(seed)

    n_targets = int(rng.integers(sampler_config.min_targets, sampler_config.max_targets + 1))

    # Keep drone information fixed unless the base config changes.
    drone_x, drone_y = config.drone_initial_position

    targets = []
    for target_id in range(1, n_targets + 1):
        radius = float(rng.uniform(sampler_config.min_radius, sampler_config.max_radius))
        angle = float(rng.uniform(-np.pi, np.pi))

        x = drone_x + radius * np.cos(angle)
        y = drone_y + radius * np.sin(angle)

        speed = float(rng.uniform(sampler_config.min_target_speed, sampler_config.max_target_speed))
        heading = float(rng.uniform(-np.pi, np.pi))

        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        velocity_noise = float(
            rng.uniform(
                sampler_config.min_velocity_noise_std,
                sampler_config.max_velocity_noise_std,
            )
        )

        targets.append(
            {
                "target_id": target_id,
                "initial_state": [float(x), float(y), float(vx), float(vy)],
                "process_noise_std": [0.0, 0.0, velocity_noise, velocity_noise],
            }
        )

    config.targets = targets
    return config
