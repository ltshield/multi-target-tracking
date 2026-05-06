"""Generate imitation-learning data from an expert planner.

This script samples randomized scenarios, runs the real simulator with an expert
planner, logs every target-selection decision, and saves train/val/test datasets.

Recommended example:

    python src/core/learning/generate_imitation_data.py ^
        --base-config configs/basic_3target.yaml ^
        --output-dir data/imitation ^
        --train-episodes 800 ^
        --val-episodes 200 ^
        --test-episodes 200 ^
        --max-tracks 6 ^
        --extractor cartesian ^
        --teacher-iterations 1000

The split is by scenario seed, not timestep, so timesteps from the same run
cannot leak between train/validation/test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core.learning.imitation_logger import ImitationLogger
from core.planners.mcts_planner_realtime import MCTSPlanner
from core.sim.scenario_sampler import ScenarioSamplerConfig, sample_random_scenario
from core.sim.simulate_run import load_config_from_yaml, make_simulation


def make_teacher(args: argparse.Namespace) -> MCTSPlanner:
    """Create the MCTS expert planner."""

    return MCTSPlanner(
        iterations=args.teacher_iterations,
        iterations_per_second=args.teacher_iterations_per_second,
        max_background_iterations_per_call=args.max_background_iterations_per_call,
        max_depth=args.teacher_max_depth,
        exploration_weight=args.teacher_exploration_weight,
        max_search_time=args.max_search_time,
        acceleration_noise_std=args.acceleration_noise_std,
        measurement_noise_std=args.measurement_noise_std,
        covariance_scale_for_detection=args.covariance_scale_for_detection,
        use_logdet_objective=args.use_logdet_objective,
        lost_trace_threshold=args.lost_trace_threshold,
        lost_target_penalty=args.lost_target_penalty,
        uncertainty_weight=args.uncertainty_weight,
        travel_distance_weight=args.travel_distance_weight,
        detection_reward=args.detection_reward,
        rollout_random_action_probability=args.rollout_random_action_probability,
        distance_bias_seconds=args.distance_bias_seconds,
        root_selection=args.root_selection,
    )


def final_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    history = run.get("history", [])
    if not history:
        return {
            "num_lost": 999,
            "num_active": 0,
            "final_metric": float("inf"),
            "num_decisions": 0,
            "num_detections": 0,
            "final_time": 0.0,
        }

    final = history[-1]
    return {
        "num_lost": int(final["metrics"]["num_lost"]),
        "num_active": int(final["metrics"]["num_active"]),
        "final_metric": float(final["metrics"]["total_position_trace"]),
        "num_decisions": int(final.get("decision_count", 0)),
        "num_detections": int(
            sum(len(frame.get("detections", [])) for frame in history)
        ),
        "final_time": float(final.get("time", 0.0)),
    }


def should_keep_episode(
    *,
    episode_logger: ImitationLogger,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if len(episode_logger) < args.min_samples_per_episode:
        return False, (
            f"samples {len(episode_logger)} < "
            f"{args.min_samples_per_episode}"
        )

    if metrics["num_lost"] > args.max_lost_for_keep:
        return False, (
            f"num_lost {metrics['num_lost']} > "
            f"{args.max_lost_for_keep}"
        )

    if metrics["final_metric"] > args.max_final_metric_for_keep:
        return False, (
            f"final_metric {metrics['final_metric']:.2f} > "
            f"{args.max_final_metric_for_keep:.2f}"
        )

    return True, "kept"


def collect_split(
    *,
    split: str,
    num_episodes: int,
    seed_start: int,
    base_config,
    sampler_config: ScenarioSamplerConfig,
    args: argparse.Namespace,
) -> tuple[ImitationLogger, list[dict[str, Any]]]:
    """Collect one dataset split."""

    split_logger = ImitationLogger(
        extractor_name=args.extractor,
        max_tracks=args.max_tracks,
    )

    episode_summaries: list[dict[str, Any]] = []

    # Separate sampler RNG per split for reproducibility.
    sampler_rng = np.random.default_rng(seed_start + args.sampler_seed_offset)

    kept = 0
    discarded = 0

    for episode_id in range(num_episodes):
        scenario_seed = int(seed_start + episode_id)

        config = sample_random_scenario(
            base_config=base_config,
            rng=sampler_rng,
            sampler_config=sampler_config,
            seed=scenario_seed,
        )

        # Keep teacher parameters aligned with the sampled config unless the user
        # explicitly overrides them.
        teacher = make_teacher(args)

        episode_logger = ImitationLogger(
            extractor_name=args.extractor,
            max_tracks=args.max_tracks,
        )

        sim = make_simulation(
            config=config,
            planner=teacher,
            imitation_logger=episode_logger,
            episode_id=episode_id,
            split=split,
        )

        try:
            run = sim.run()
            metrics = final_run_metrics(run)
            keep, reason = should_keep_episode(
                episode_logger=episode_logger,
                metrics=metrics,
                args=args,
            )
        except Exception as exc:
            metrics = {
                "num_lost": 999,
                "num_active": 0,
                "final_metric": float("inf"),
                "num_decisions": 0,
                "num_detections": 0,
                "final_time": 0.0,
            }
            keep = False
            reason = f"simulation_error: {exc}"

        if keep:
            split_logger.extend(episode_logger)
            kept += 1
            status = "KEPT"
        else:
            discarded += 1
            status = "DISCARDED"

        summary = {
            "split": split,
            "episode_id": int(episode_id),
            "scenario_seed": int(scenario_seed),
            "num_targets": int(len(config.targets)),
            "num_examples": int(len(episode_logger)),
            "status": status,
            "reason": reason,
            **metrics,
        }
        episode_summaries.append(summary)

        print(
            f"{split:>5} episode={episode_id:04d} "
            f"seed={scenario_seed} "
            f"targets={len(config.targets)} "
            f"samples={len(episode_logger)} "
            f"kept_total={len(split_logger)} "
            f"lost={metrics['num_lost']} "
            f"active={metrics['num_active']} "
            f"final_metric={metrics['final_metric']:.2f} "
            f"{status}"
            + (f" ({reason})" if status == "DISCARDED" else "")
        )

    print(
        f"\n{split.upper()} summary: "
        f"kept_episodes={kept}, "
        f"discarded_episodes={discarded}, "
        f"examples={len(split_logger)}\n"
    )

    return split_logger, episode_summaries


def save_episode_summaries(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved episode summaries: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/imitation")

    parser.add_argument("--extractor", type=str, default="cartesian")
    parser.add_argument("--max-tracks", type=int, default=6)

    parser.add_argument("--train-episodes", type=int, default=800)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=200)

    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--val-seed-start", type=int, default=2000)
    parser.add_argument("--test-seed-start", type=int, default=3000)
    parser.add_argument("--sampler-seed-offset", type=int, default=99173)

    # Random scenario distribution.
    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=6)
    parser.add_argument("--min-radius", type=float, default=250.0)
    parser.add_argument("--max-radius", type=float, default=900.0)
    parser.add_argument("--min-target-speed", type=float, default=0.25)
    parser.add_argument("--max-target-speed", type=float, default=3.0)
    parser.add_argument("--min-velocity-noise-std", type=float, default=0.0)
    parser.add_argument("--max-velocity-noise-std", type=float, default=0.006)

    # Expert MCTS parameters.
    parser.add_argument("--teacher-iterations", type=int, default=1000)
    parser.add_argument("--teacher-iterations-per-second", type=float, default=80.0)
    parser.add_argument("--max-background-iterations-per-call", type=int, default=250)
    parser.add_argument("--teacher-max-depth", type=int, default=5)
    parser.add_argument("--teacher-exploration-weight", type=float, default=1.4)
    parser.add_argument("--max-search-time", type=float, default=65.0)

    parser.add_argument("--acceleration-noise-std", type=float, default=0.03)
    parser.add_argument("--measurement-noise-std", type=float, default=20.0)
    parser.add_argument("--covariance-scale-for-detection", type=float, default=3.0)

    parser.add_argument("--use-logdet-objective", action="store_true")
    parser.add_argument("--lost-trace-threshold", type=float, default=250000.0)
    parser.add_argument("--lost-target-penalty", type=float, default=1000000.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--travel-distance-weight", type=float, default=0.0)
    parser.add_argument("--detection-reward", type=float, default=0.0)
    parser.add_argument("--rollout-random-action-probability", type=float, default=0.10)
    parser.add_argument("--distance-bias-seconds", type=float, default=30.0)
    parser.add_argument(
        "--root-selection",
        type=str,
        choices=["value", "visits"],
        default="value",
    )

    # Episode quality filters.
    parser.add_argument("--max-lost-for-keep", type=int, default=0)
    parser.add_argument(
        "--max-final-metric-for-keep",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("--min-samples-per-episode", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config_from_yaml(Path(args.base_config))

    sampler_config = ScenarioSamplerConfig(
        min_targets=args.min_targets,
        max_targets=args.max_targets,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_target_speed=args.min_target_speed,
        max_target_speed=args.max_target_speed,
        min_velocity_noise_std=args.min_velocity_noise_std,
        max_velocity_noise_std=args.max_velocity_noise_std,
    )

    config_summary = {
        "base_config": args.base_config,
        "output_dir": str(output_dir),
        "extractor": args.extractor,
        "max_tracks": args.max_tracks,
        "sampler_config": asdict(sampler_config),
        "args": vars(args),
    }

    with (output_dir / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    all_summaries: list[dict[str, Any]] = []

    split_specs = [
        ("train", args.train_episodes, args.train_seed_start),
        ("val", args.val_episodes, args.val_seed_start),
        ("test", args.test_episodes, args.test_seed_start),
    ]

    for split, num_episodes, seed_start in split_specs:
        logger, summaries = collect_split(
            split=split,
            num_episodes=num_episodes,
            seed_start=seed_start,
            base_config=base_config,
            sampler_config=sampler_config,
            args=args,
        )

        if len(logger) == 0:
            raise ValueError(
                f"No examples collected for split={split}. "
                "Try relaxing filters or increasing scenario count."
            )

        logger.save_pt(output_dir / f"{split}.pt")
        all_summaries.extend(summaries)

    save_episode_summaries(
        all_summaries,
        output_dir / "episode_summaries.json",
    )

    print("Done generating imitation-learning data.")


if __name__ == "__main__":
    main()