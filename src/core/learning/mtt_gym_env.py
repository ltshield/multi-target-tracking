
"""Gymnasium environment for target-selection training with Stable-Baselines3.

Action:
    Discrete(max_targets), where each action slot maps to the sorted active
    track list by track_id.

Step:
    One action means "choose this target and execute the current pursuit process":
    fly to the belief center, spiral-search until selected target is found,
    selected target is lost, search times out, all targets lost, or budget ends.

Reward:
    reduction in total uncertainty minus lost-target penalties.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install gymnasium: pip install gymnasium") from exc

from core.sim.coverage_spiral import EllipticShiftingSpiralPlanner
from core.sim.simulate_run import SimConfig, make_initial_targets, make_initial_tracks_from_targets
from core.sim.drone import Drone
from core.sim.tracks import TrackSet
from core.learning.observation import (
    observation_dim,
    build_observation_from_tracks,
    active_action_mask,
    action_index_to_track_id,
)


class MTTTargetSelectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig, max_targets: int = 6):
        super().__init__()
        self.config = config
        self.max_targets = max_targets

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(max_targets),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max_targets)

        self.rng = np.random.default_rng(config.seed)
        self.drone = None
        self.targets = None
        self.tracks = None
        self.decision_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.drone = Drone(
            position=self.config.drone_initial_position,
            speed=self.config.drone_speed,
            sensor_range=self.config.sensor_range,
            budget=self.config.mission_budget,
            detection_probability=self.config.detection_probability,
            rng=self.rng,
        )
        self.targets = make_initial_targets(self.config)
        self.tracks = make_initial_tracks_from_targets(self.targets, self.config)
        self.decision_count = 0

        return self._obs(), self._info(event="reset")

    def step(self, action):
        action = int(action)
        before_metric = self._objective_metric()

        mask = active_action_mask(self.tracks, self.max_targets)
        invalid_action = action >= len(mask) or not mask[action]

        if invalid_action:
            # Strong penalty for choosing a nonexistent/lost track.
            reward = -self.config.lost_target_penalty
            terminated = self._terminated()
            return self._obs(), reward, terminated, False, self._info(event="invalid_action")

        track_id = action_index_to_track_id(self.tracks, action, self.max_targets)
        event = self._execute_pursuit(track_id)

        after_metric = self._objective_metric()
        reward = before_metric - after_metric
        self.decision_count += 1

        terminated = self._terminated()
        return self._obs(), float(reward), terminated, False, self._info(event=event)

    def _execute_pursuit(self, track_id: int) -> str:
        selected_track = self.tracks[track_id]
        spiral = EllipticShiftingSpiralPlanner.from_track(
            track=selected_track,
            sensor_width=self.drone.sensor_width,
            dt=self.config.dt,
            covariance_scale=self.config.covariance_scale_for_search,
        )
        spiral_state = spiral.initial_state()

        # Fly to belief center.
        while self.drone.is_active and self.drone.distance_to(spiral.center0) > 1e-6:
            consumed = self.drone.step_toward(spiral.center0, self.config.dt)
            self._advance_world(consumed)
            if self._track_is_lost(track_id):
                return "selected_target_lost"

        # Spiral search.
        search_elapsed = 0.0
        while self.drone.is_active and search_elapsed < self.config.max_search_time_per_decision:
            waypoint, spiral_state = spiral.next_waypoint(spiral_state, self.drone.speed)
            consumed = self.drone.step_toward(waypoint, self.config.dt)
            search_elapsed += consumed
            self._advance_world(consumed)

            detections = self._detect_and_update()
            if any(int(d.target_id) == int(track_id) for d in detections):
                return "selected_target_found"

            if self._track_is_lost(track_id):
                return "selected_target_lost"

        return "search_timeout"

    def _advance_world(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.targets.step(dt, add_noise=True)
        self.tracks.predict_all(
            dt=dt,
            acceleration_noise_std=self.config.acceleration_noise_std,
            existence_decay_rate=0.0,
        )
        self.tracks.check_lost_tracks(
            current_time=self.drone.elapsed_time,
            max_position_trace=self.config.max_position_trace_before_lost,
            max_position_logdet=self.config.max_position_logdet_before_lost,
            max_time_since_seen=self.config.max_time_since_seen_before_lost,
        )

    def _detect_and_update(self):
        raw = self.drone.detect_targets(self.targets.positions_dict())
        detections = []
        for det in raw:
            try:
                track = self.tracks[det.target_id]
            except KeyError:
                continue
            if getattr(track, "is_lost", False):
                continue
            detections.append(det)

        if detections:
            self.tracks.update_from_detections(
                detections,
                measurement_noise_std=self.config.measurement_noise_std,
            )
        return detections

    def _track_is_lost(self, track_id: int) -> bool:
        try:
            return bool(self.tracks[track_id].is_lost)
        except KeyError:
            return True

    def _objective_metric(self) -> float:
        return float(
            self.tracks.total_position_trace(active_only=True)
            + self.config.lost_target_penalty * self.tracks.num_lost()
        )

    def _terminated(self) -> bool:
        return (
            not self.drone.is_active
            or self.tracks.num_active() == 0
            or self.decision_count >= self.config.max_steps
        )

    def _obs(self):
        return build_observation_from_tracks(self.tracks, self.drone, self.max_targets)

    def _info(self, event: str):
        return {
            "event": event,
            "time": float(self.drone.elapsed_time),
            "remaining_budget": float(self.drone.remaining_budget),
            "num_active": int(self.tracks.num_active()),
            "num_lost": int(self.tracks.num_lost()),
            "objective_metric": self._objective_metric(),
        }
