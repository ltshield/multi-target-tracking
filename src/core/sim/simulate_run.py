"""Run and save a multi-target tracking simulation with background conditional MCTS.

This version separates execution from planning more realistically:

- The first action is chosen normally.
- Once the drone begins flying/searching for the selected target, a planner that
  supports the background interface is allowed to plan during each elapsed
  simulation step.
- For MCTS, that background planning fixes the current action and builds
  conditional next-action recommendations for:
    outcome == "find"
    outcome == "miss"
- When the selected target is found, times out, or is considered lost, the
  simulation uses the cached conditional recommendation if available.

Planner interfaces
------------------
Simple planners only need:

    choose_track(tracks, drone, targets, rng) -> int

Realtime/conditional planners may additionally implement:

    start_conditional_planning(tracks, drone, targets, rng, current_action) -> None
    plan_during_execution(tracks, drone, targets, rng, planning_seconds) -> None
    finish_conditional_planning(outcome, tracks, drone, targets, rng) -> int | None

where outcome is "find" or "miss".
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from core.sim.coverage_spiral import EllipticShiftingSpiralPlanner
from core.sim.drone import Drone
from core.sim.target import Target, TargetSet
from core.sim.tracks import Track, TrackSet


class PlannerProtocol(Protocol):
    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        ...


class RandomPlanner:
    """Baseline planner that randomly selects one active track."""

    def choose_track(
        self,
        tracks: TrackSet,
        drone: Drone,
        targets: TargetSet,
        rng: np.random.Generator,
    ) -> int:
        valid_actions = tracks.valid_action_ids()

        if not valid_actions:
            raise RuntimeError("RandomPlanner received no valid actions.")

        return int(rng.choice(valid_actions))


class Mode(Enum):
    CHOOSE_TARGET = auto()
    FLY_TO_CENTER = auto()
    SPIRAL_SEARCH = auto()
    DONE = auto()


@dataclass
class SimConfig:
    seed: int = 7
    dt: float = 0.5
    mission_budget: float = 900.0
    max_steps: int = 10000

    opportunistic_detections: bool = True

    # Track loss thresholds.
    max_position_trace_before_lost: float | None = 250000.0
    max_position_logdet_before_lost: float | None = None
    max_time_since_seen_before_lost: float | None = None
    min_existence_probability_before_lost: float | None = None
    lost_target_penalty: float = 1000000.0

    drone_initial_position: list[float] = field(default_factory=lambda: [0.0, 0.0])
    drone_speed: float = 30.0
    sensor_range: float = 50.0
    detection_probability: float = 1.0

    max_search_time_per_decision: float = 65.0

    acceleration_noise_std: float = 0.03
    measurement_noise_std: float = 20.0
    initial_position_std: float = 75.0
    initial_velocity_std: float = 3.0
    covariance_scale_for_search: float = 2.0

    targets: list[dict[str, Any]] = field(default_factory=list)


def load_config_from_yaml(path: Path) -> SimConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    drone = raw.get("drone", {})
    belief = raw.get("belief", {})
    loss = raw.get("loss", {})
    search = raw.get("search", {})

    return SimConfig(
        seed=int(raw.get("seed", 7)),
        dt=float(raw.get("dt", 0.5)),
        mission_budget=float(raw.get("mission_budget", 900.0)),
        max_steps=int(raw.get("max_steps", 10000)),
        opportunistic_detections=bool(raw.get("opportunistic_detections", True)),
        drone_initial_position=drone.get("initial_position", [0.0, 0.0]),
        drone_speed=float(drone.get("speed", 30.0)),
        sensor_range=float(drone.get("sensor_range", 50.0)),
        detection_probability=float(drone.get("detection_probability", 1.0)),
        max_search_time_per_decision=float(
            search.get("max_search_time_per_decision", 65.0)
        ),
        acceleration_noise_std=float(belief.get("acceleration_noise_std", 0.03)),
        measurement_noise_std=float(belief.get("measurement_noise_std", 20.0)),
        initial_position_std=float(belief.get("initial_position_std", 75.0)),
        initial_velocity_std=float(belief.get("initial_velocity_std", 3.0)),
        covariance_scale_for_search=float(
            belief.get("covariance_scale_for_search", 2.0)
        ),
        max_position_trace_before_lost=loss.get(
            "max_position_trace_before_lost", 250000.0
        ),
        max_position_logdet_before_lost=loss.get(
            "max_position_logdet_before_lost", None
        ),
        max_time_since_seen_before_lost=loss.get(
            "max_time_since_seen_before_lost", None
        ),
        min_existence_probability_before_lost=loss.get(
            "min_existence_probability_before_lost", None
        ),
        lost_target_penalty=float(loss.get("lost_target_penalty", 1000000.0)),
        targets=raw.get("targets", []),
    )


@dataclass
class SimulationState:
    rng: np.random.Generator
    config: SimConfig
    planner: PlannerProtocol
    drone: Drone
    targets: TargetSet
    tracks: TrackSet

    mode: Mode = Mode.CHOOSE_TARGET
    selected_track_id: int | None = None
    spiral_planner: EllipticShiftingSpiralPlanner | None = None
    spiral_state: object | None = None
    search_elapsed: float = 0.0
    decision_count: int = 0

    # If background conditional planning produced the next action, it is staged
    # here and consumed by choose_target().
    planned_next_track_id: int | None = None
    planned_next_outcome: str | None = None

    newly_lost_tracks: list[int] = field(default_factory=list)
    background_planning_seconds: float = 0.0
    background_planning_calls: int = 0

    history: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        self.log_frame(event="start")

        step_idx = 0
        while self.mode != Mode.DONE and step_idx < self.config.max_steps:
            self.step()
            step_idx += 1

        if step_idx >= self.config.max_steps:
            self.mode = Mode.DONE
            self.log_frame(event="max_steps_reached")

        self.log_frame(event="end")

        return {
            "metadata": {
                "sim_type": "multi_target_tracking",
                "planner": type(self.planner).__name__,
                "seed": self.config.seed,
            },
            "config": self.config.__dict__,
            "history": self.history,
        }

    def step(self) -> None:
        self.newly_lost_tracks = []

        if not self.drone.is_active:
            self.mode = Mode.DONE
            self.log_frame(event="budget_depleted")
            return

        if self.mode == Mode.CHOOSE_TARGET:
            self.choose_target()
            self.log_frame(event="choose_target")
            return

        if self.mode == Mode.FLY_TO_CENTER:
            self.step_fly_to_center()
            return

        if self.mode == Mode.SPIRAL_SEARCH:
            self.step_spiral_search()
            return

    # ------------------------------------------------------------------
    # Planning / action lifecycle
    # ------------------------------------------------------------------

    def choose_target(self) -> None:
        """Choose the next target.

        If the current planner has been planning conditionally in the background,
        use that staged recommendation. Otherwise ask the planner from scratch.
        """

        available_tracks = self.available_tracks_for_planning()
        valid_actions = available_tracks.valid_action_ids()

        if not valid_actions:
            self.mode = Mode.DONE
            self.log_frame(event="all_targets_lost")
            return

        selected_id: int | None = None

        if (
            self.planned_next_track_id is not None
            and int(self.planned_next_track_id) in valid_actions
        ):
            selected_id = int(self.planned_next_track_id)

        # Consume the staged recommendation whether valid or invalid.
        self.planned_next_track_id = None
        self.planned_next_outcome = None

        if selected_id is None:
            selected_id = int(
                self.planner.choose_track(
                    tracks=available_tracks,
                    drone=self.drone,
                    targets=self.targets,
                    rng=self.rng,
                )
            )

        if selected_id not in valid_actions:
            # Hard safety fallback: no planner is allowed to select a lost target.
            selected_id = int(valid_actions[0])

        self.selected_track_id = int(selected_id)
        selected_track = self.tracks[self.selected_track_id]

        self.spiral_planner = EllipticShiftingSpiralPlanner.from_track(
            track=selected_track,
            sensor_width=self.drone.sensor_width,
            dt=self.config.dt,
            covariance_scale=self.config.covariance_scale_for_search,
        )
        self.spiral_state = self.spiral_planner.initial_state()
        self.search_elapsed = 0.0
        self.decision_count += 1

        self.start_background_planning_for_current_action()
        self.mode = Mode.FLY_TO_CENTER

    def start_background_planning_for_current_action(self) -> None:
        if self.selected_track_id is None:
            return

        method = getattr(self.planner, "start_conditional_planning", None)
        if callable(method):
            method(
                tracks=self.available_tracks_for_planning(),
                drone=self.drone,
                targets=self.targets,
                rng=self.rng,
                current_action=int(self.selected_track_id),
            )

    def run_background_planning(self, planning_seconds: float) -> None:
        """Let a realtime-capable planner compute during execution time."""

        if planning_seconds <= 0.0:
            return

        method = getattr(self.planner, "plan_during_execution", None)
        if not callable(method):
            return

        method(
            tracks=self.available_tracks_for_planning(),
            drone=self.drone,
            targets=self.targets,
            rng=self.rng,
            planning_seconds=float(planning_seconds),
        )
        self.background_planning_seconds += float(planning_seconds)
        self.background_planning_calls += 1

    def finish_current_action(self, outcome: str) -> None:
        """Finish current pursuit and stage a conditional next action if possible."""

        method = getattr(self.planner, "finish_conditional_planning", None)
        next_id = None

        if callable(method):
            next_id = method(
                outcome=outcome,
                tracks=self.available_tracks_for_planning(),
                drone=self.drone,
                targets=self.targets,
                rng=self.rng,
            )

        if next_id is not None and self.track_is_active(int(next_id)):
            self.planned_next_track_id = int(next_id)
            self.planned_next_outcome = outcome
        else:
            self.planned_next_track_id = None
            self.planned_next_outcome = outcome

        self.mode = Mode.CHOOSE_TARGET

    # ------------------------------------------------------------------
    # Execution modes
    # ------------------------------------------------------------------

    def step_fly_to_center(self) -> None:
        assert self.spiral_planner is not None
        assert self.selected_track_id is not None

        consumed = self.drone.step_toward(self.spiral_planner.center0, self.config.dt)

        self.advance_world_and_beliefs(consumed)

        detections = []
        if self.config.opportunistic_detections:
            detections = self.detect_and_update_tracks()

        # Planning is allowed for exactly the real execution time consumed.
        self.run_background_planning(consumed)

        selected_found = self.detections_include_selected(detections)
        selected_lost = self.selected_track_id in self.newly_lost_tracks

        event = "fly_to_center"
        if selected_found:
            event = "detection"
            self.finish_current_action(outcome="find")
        elif selected_lost:
            event = "selected_target_lost"
            self.finish_current_action(outcome="miss")
        elif self.drone.distance_to(self.spiral_planner.center0) <= 1e-6:
            self.mode = Mode.SPIRAL_SEARCH

        self.log_frame(event=event, detections=detection_ids(detections))

    def step_spiral_search(self) -> None:
        assert self.spiral_planner is not None
        assert self.spiral_state is not None
        assert self.selected_track_id is not None

        waypoint, self.spiral_state = self.spiral_planner.next_waypoint(
            self.spiral_state,
            self.drone.speed,
        )

        consumed = self.drone.step_toward(waypoint, self.config.dt)
        self.search_elapsed += consumed

        self.advance_world_and_beliefs(consumed)
        detections = self.detect_and_update_tracks()

        # Planning is allowed while the drone is executing the spiral.
        self.run_background_planning(consumed)

        selected_found = self.detections_include_selected(detections)
        selected_lost = self.selected_track_id in self.newly_lost_tracks

        event = "spiral_search"
        if selected_found:
            event = "detection"
            self.finish_current_action(outcome="find")
        elif selected_lost:
            event = "selected_target_lost"
            self.finish_current_action(outcome="miss")
        elif self.search_elapsed >= self.config.max_search_time_per_decision:
            event = "search_timeout"
            self.finish_current_action(outcome="miss")
        elif not self.drone.is_active:
            event = "budget_depleted"
            self.mode = Mode.DONE

        self.log_frame(
            event=event,
            detections=detection_ids(detections),
            spiral_waypoint=waypoint,
        )

    # ------------------------------------------------------------------
    # World, belief, and detection updates
    # ------------------------------------------------------------------

    def advance_world_and_beliefs(self, dt: float) -> None:
        if dt <= 0.0:
            return

        # Move only active ground-truth targets.
        # Target.step() handles the active/lost check internally.
        self.targets.step(dt, add_noise=True)

        # Predict only active belief tracks.
        # TrackSet.predict_all() skips lost tracks internally.
        self.tracks.predict_all(
            dt=dt,
            acceleration_noise_std=self.config.acceleration_noise_std,
            existence_decay_rate=0.0,
        )

        newly_lost = self.tracks.check_lost_tracks(
            current_time=self.drone.elapsed_time,
            max_position_trace=self.config.max_position_trace_before_lost,
            max_position_logdet=self.config.max_position_logdet_before_lost,
            max_time_since_seen=self.config.max_time_since_seen_before_lost,
            min_existence_probability=self.config.min_existence_probability_before_lost,
        )

        if newly_lost:
            self.newly_lost_tracks = newly_lost

            # Important: when the belief track is lost, deactivate the corresponding
            # true target so it stops moving and stops being detectable.
            for track_id in newly_lost:
                try:
                    self.targets[track_id].mark_lost()
                except KeyError:
                    pass

    def detect_and_update_tracks(self):
        raw_detections = self.drone.detect_targets(self.targets.positions_dict())

        detections = []
        for detection in raw_detections:
            try:
                track = self.tracks[detection.target_id]
            except KeyError:
                continue

            # Once lost, irretrievable in this use case.
            if track.is_lost:
                continue

            detections.append(detection)

        if detections:
            self.tracks.update_from_detections(
                detections,
                measurement_noise_std=self.config.measurement_noise_std,
            )

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def available_tracks_for_planning(self) -> TrackSet:
        return TrackSet([track.copy() for track in self.tracks.active_tracks])

    def track_is_active(self, track_id: int) -> bool:
        try:
            return not self.tracks[track_id].is_lost
        except KeyError:
            return False

    def detections_include_selected(self, detections) -> bool:
        if self.selected_track_id is None:
            return False
        return any(int(d.target_id) == int(self.selected_track_id) for d in detections)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        event: str,
        detections: list[int] | None = None,
        spiral_waypoint: np.ndarray | None = None,
    ) -> None:
        frame = {
            "event": event,
            "mode": self.mode.name,
            "time": float(self.drone.elapsed_time),
            "remaining_budget": float(self.drone.remaining_budget),
            "decision_count": int(self.decision_count),
            "selected_track_id": self.selected_track_id,
            "planned_next_track_id": self.planned_next_track_id,
            "planned_next_outcome": self.planned_next_outcome,
            "search_elapsed": float(self.search_elapsed),
            "detections": detections or [],
            "background_planning_seconds": float(self.background_planning_seconds),
            "background_planning_calls": int(self.background_planning_calls),
            "planner_diagnostics": self.get_planner_diagnostics(),
            "drone": {
                "position": arr(self.drone.position),
                "speed": float(self.drone.speed),
                "sensor_range": float(self.drone.sensor_range),
                "distance_traveled": float(self.drone.distance_traveled),
            },
            "targets": {
                str(target.target_id): {
                    "state": arr(target.state),
                    "position": arr(target.position),
                    "velocity": arr(target.velocity),
                    "is_active": bool(target.is_active),
                }
                for target in self.targets
            },
            "tracks": {
                str(track.track_id): {
                    "mean": arr(track.mean),
                    "position": arr(track.position),
                    "velocity": arr(track.velocity),
                    "covariance": matrix(track.covariance),
                    "position_covariance": matrix(track.position_covariance),
                    "existence_probability": float(track.existence_probability),
                    "time_since_seen": float(track.time_since_seen),
                    "is_lost": bool(track.is_lost),
                    "lost_reason": track.lost_reason,
                    "lost_time": track.lost_time,
                    "position_variance_trace": float(track.position_variance_trace),
                    "reported_position_variance_trace": float(
                        track.reported_position_variance_trace(
                            self.config.lost_target_penalty
                        )
                    ),
                    "position_uncertainty_logdet": float(
                        track.position_uncertainty_logdet
                    ),
                }
                for track in self.tracks
            },
            "metrics": {
                "total_position_trace": float(
                    self.tracks.total_position_trace(active_only=True)
                    + self.config.lost_target_penalty * self.tracks.num_lost()
                ),
                "active_position_trace": float(
                    self.tracks.total_position_trace(active_only=True)
                ),
                "total_position_logdet": float(
                    self.tracks.total_position_logdet(active_only=True)
                ),
                "num_lost": int(self.tracks.num_lost()),
                "num_active": int(self.tracks.num_active()),
            },
            "newly_lost_tracks": self.newly_lost_tracks,
        }

        if spiral_waypoint is not None:
            frame["spiral_waypoint"] = arr(spiral_waypoint)

        self.history.append(frame)

    def get_planner_diagnostics(self) -> dict[str, Any]:
        method = getattr(self.planner, "diagnostics", None)
        if callable(method):
            return method()
        return {}


def arr(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=float).tolist()]


def matrix(x: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(x, dtype=float).tolist()]


def detection_ids(detections) -> list[int]:
    return [int(detection.target_id) for detection in detections]


def make_initial_targets(config: SimConfig) -> TargetSet:
    if not config.targets:
        raise ValueError(
            "No targets were provided. Add a 'targets:' list to your config file."
        )

    targets: list[Target] = []
    for target_cfg in config.targets:
        target_id = int(target_cfg["target_id"])
        child_rng = np.random.default_rng(config.seed + 1000 + target_id)
        targets.append(
            Target(
                target_id=target_id,
                state=target_cfg["initial_state"],
                process_noise_std=target_cfg.get("process_noise_std", 0.0),
                rng=child_rng,
            )
        )

    return TargetSet(targets)


def make_initial_tracks_from_targets(targets: TargetSet, config: SimConfig) -> TrackSet:
    tracks: list[Track] = []

    for target in targets:
        covariance = np.diag(
            [
                config.initial_position_std**2,
                config.initial_position_std**2,
                config.initial_velocity_std**2,
                config.initial_velocity_std**2,
            ]
        )
        tracks.append(
            Track(
                track_id=target.target_id,
                mean=target.state.copy(),
                covariance=covariance,
                existence_probability=1.0,
            )
        )

    return TrackSet(tracks)


def make_simulation(config: SimConfig, planner: PlannerProtocol) -> SimulationState:
    rng = np.random.default_rng(config.seed)

    drone = Drone(
        position=config.drone_initial_position,
        speed=config.drone_speed,
        sensor_range=config.sensor_range,
        budget=config.mission_budget,
        detection_probability=config.detection_probability,
        rng=rng,
    )

    targets = make_initial_targets(config)
    tracks = make_initial_tracks_from_targets(targets, config)

    return SimulationState(
        rng=rng,
        config=config,
        planner=planner,
        drone=drone,
        targets=targets,
        tracks=tracks,
    )


def load_planner(planner_path: str | None) -> PlannerProtocol:
    if planner_path is None or planner_path.lower() == "random":
        return RandomPlanner()

    module_name, class_name = planner_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    planner_cls = getattr(module, class_name)
    return planner_cls()


def save_run(run: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)

    print(f"Saved run to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="runs/random_run.json")
    parser.add_argument("--planner", type=str, default="planners.RandomPlanner")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)

    parser.add_argument(
        "--opportunistic-detections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable detections while flying to a belief center.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        config = load_config_from_yaml(Path(args.config))
    else:
        raise ValueError(
            "Please provide a config file with --config configs/basic_3target.yaml"
        )

    if args.seed is not None:
        config.seed = args.seed
    if args.budget is not None:
        config.mission_budget = args.budget
    if args.dt is not None:
        config.dt = args.dt
    if args.opportunistic_detections is not None:
        config.opportunistic_detections = args.opportunistic_detections

    planner = load_planner(args.planner)

    sim = make_simulation(config=config, planner=planner)
    run = sim.run()

    save_run(run, Path(args.output))


if __name__ == "__main__":
    main()
