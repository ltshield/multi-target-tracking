"""Target / ground-truth dynamics for multi-target tracking simulations.

This file represents the *true* simulated targets, not the drone's belief about
those targets. The tracker/belief files should estimate target state; this file
advances the hidden ground truth that the sensor may or may not observe.

State convention:
    [x, y, vx, vy]

Default dynamics:
    constant velocity with optional Gaussian process noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


ArrayLike2D = Sequence[float] | np.ndarray
ArrayLike4D = Sequence[float] | np.ndarray


@dataclass(slots=True)
class Target:
    """Ground-truth target used by the simulator.

    Parameters
    ----------
    target_id:
        Unique integer ID used internally by the simulator.
    state:
        True target state `[x, y, vx, vy]`.
    process_noise_std:
        Standard deviation of additive Gaussian noise applied during each step.
        This can either be:
        - a scalar, applied to all 4 state entries
        - a length-4 vector, one value for each state entry
        Use 0.0 for deterministic constant-velocity motion.
    rng:
        Optional NumPy random generator for reproducibility.
    is_active:
        Whether this target should continue moving and being detectable.
    """

    target_id: int
    state: ArrayLike4D
    process_noise_std: float | ArrayLike4D = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    is_active: bool = True

    age: float = 0.0

    def __post_init__(self) -> None:
        self.state = self._as_state(self.state)
        self.process_noise_std = self._as_noise_std(self.process_noise_std)

        if self.target_id < 0:
            raise ValueError("target_id must be nonnegative.")

    @property
    def position(self) -> np.ndarray:
        """Current true 2D position `[x, y]`."""

        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        """Current true 2D velocity `[vx, vy]`."""

        return self.state[2:4]

    @property
    def speed(self) -> float:
        """Current true scalar speed."""

        return float(np.linalg.norm(self.velocity))

    def copy(self) -> "Target":
        """Return a simulation-safe copy of this target."""

        copied = Target(
            target_id=self.target_id,
            state=self.state.copy(),
            process_noise_std=self.process_noise_std.copy(),
            rng=self.rng,
            is_active=self.is_active,
        )
        copied.age = self.age
        return copied

    def predict_state(self, dt: float) -> np.ndarray:
        """Return deterministic constant-velocity prediction without mutation."""

        self._validate_dt(dt)
        F = self.transition_matrix(dt)
        return F @ self.state

    def step(self, dt: float, add_noise: bool = True) -> np.ndarray:
        """Advance the target by one time step.

        Returns
        -------
        np.ndarray
            The updated true state.
        """

        self._validate_dt(dt)

        if not self.is_active:
            return self.state

        self.state = self.predict_state(dt)

        if add_noise and np.any(self.process_noise_std > 0.0):
            self.state = self.state + self.rng.normal(
                loc=0.0,
                scale=self.process_noise_std,
                size=4,
            )

        self.age += dt
        return self.state

    def as_detection_position(self, measurement_noise_std: float | ArrayLike2D = 0.0) -> np.ndarray:
        """Return a noisy measured position for this target.

        The drone currently detects exact target positions. When you want a more
        realistic sensor, call this after a detection and feed the noisy position
        into the tracker.
        """

        noise_std = self._as_position_noise_std(measurement_noise_std)

        if np.any(noise_std > 0.0):
            return self.position + self.rng.normal(loc=0.0, scale=noise_std, size=2)

        return self.position.copy()

    def to_position_tuple(self) -> tuple[int, np.ndarray]:
        """Return `(target_id, position)` for Drone.detect_targets(...)."""

        return self.target_id, self.position.copy()

    def to_dict(self) -> dict:
        """Return a YAML/JSON-friendly representation."""

        return {
            "target_id": self.target_id,
            "state": self.state.tolist(),
            "process_noise_std": self.process_noise_std.tolist(),
            "is_active": self.is_active,
            "age": self.age,
        }

    @classmethod
    def from_config(cls, config: dict, rng: np.random.Generator | None = None) -> "Target":
        """Build a Target from a YAML-style dictionary.

        Expected shape:

        target_id: 1
        initial_state: [500.0, 250.0, 2.0, -1.0]
        process_noise_std: [0.0, 0.0, 0.0, 0.0]
        """

        target_id = int(config.get("target_id", config.get("id")))
        state = config.get("initial_state", config.get("state"))

        if state is None:
            raise ValueError("Target config must include 'initial_state' or 'state'.")

        return cls(
            target_id=target_id,
            state=state,
            process_noise_std=config.get("process_noise_std", 0.0),
            rng=np.random.default_rng() if rng is None else rng,
            is_active=bool(config.get("is_active", True)),
        )

    @staticmethod
    def transition_matrix(dt: float) -> np.ndarray:
        """Constant-velocity state transition matrix."""

        Target._validate_dt(dt)
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _as_state(value: ArrayLike4D) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Target state must have shape (4,), got {arr.shape}.")
        return arr

    @staticmethod
    def _as_noise_std(value: float | ArrayLike4D) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape == ():
            if float(arr) < 0.0:
                raise ValueError("process_noise_std must be nonnegative.")
            return np.full(4, float(arr), dtype=float)

        if arr.shape != (4,):
            raise ValueError(
                f"process_noise_std must be scalar or shape (4,), got {arr.shape}."
            )
        if np.any(arr < 0.0):
            raise ValueError("process_noise_std entries must be nonnegative.")

        return arr

    @staticmethod
    def _as_position_noise_std(value: float | ArrayLike2D) -> np.ndarray:
        arr = np.asarray(value, dtype=float)

        if arr.shape == ():
            if float(arr) < 0.0:
                raise ValueError("measurement_noise_std must be nonnegative.")
            return np.full(2, float(arr), dtype=float)

        if arr.shape != (2,):
            raise ValueError(
                f"measurement_noise_std must be scalar or shape (2,), got {arr.shape}."
            )
        if np.any(arr < 0.0):
            raise ValueError("measurement_noise_std entries must be nonnegative.")

        return arr

    @staticmethod
    def _validate_dt(dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")


@dataclass(slots=True)
class TargetSet:
    """Small convenience wrapper for a collection of ground-truth targets."""

    targets: list[Target]

    def __post_init__(self) -> None:
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("Target IDs must be unique.")

    def __iter__(self) -> Iterable[Target]:
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, target_id: int) -> Target:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise KeyError(f"No target with ID {target_id}.")

    @property
    def active_targets(self) -> list[Target]:
        return [target for target in self.targets if target.is_active]

    def copy(self) -> "TargetSet":
        return TargetSet([target.copy() for target in self.targets])

    def step(self, dt: float, add_noise: bool = True) -> None:
        """Advance all active targets."""

        for target in self.targets:
            target.step(dt, add_noise=add_noise)

    def positions_dict(self, active_only: bool = True) -> dict[int, np.ndarray]:
        """Return `{target_id: position}` for Drone.detect_targets(...)."""

        selected = self.active_targets if active_only else self.targets
        return {target.target_id: target.position.copy() for target in selected}

    def states_dict(self, active_only: bool = True) -> dict[int, np.ndarray]:
        """Return `{target_id: state}` for logging/debugging."""

        selected = self.active_targets if active_only else self.targets
        return {target.target_id: target.state.copy() for target in selected}

    @classmethod
    def from_config(
        cls,
        config: dict,
        rng: np.random.Generator | None = None,
    ) -> "TargetSet":
        """Build a TargetSet from a YAML-style dictionary.

        Expected shape:

        targets:
          - target_id: 1
            initial_state: [500.0, 250.0, 2.0, -1.0]
            process_noise_std: 0.0
          - target_id: 2
            initial_state: [-300.0, 100.0, 1.0, 2.0]
            process_noise_std: [0.1, 0.1, 0.01, 0.01]
        """

        target_configs = config.get("targets", config)
        if not isinstance(target_configs, list):
            raise ValueError("TargetSet config must include a list under 'targets'.")

        base_rng = np.random.default_rng() if rng is None else rng
        targets: list[Target] = []

        for target_config in target_configs:
            # Give each target its own RNG stream while remaining reproducible
            # if base_rng was seeded.
            child_seed = int(base_rng.integers(0, 2**32 - 1))
            child_rng = np.random.default_rng(child_seed)
            targets.append(Target.from_config(target_config, rng=child_rng))

        return cls(targets)
