"""Run multiple planners, save each run, and compare performance metrics.

This script is meant to sit beside:
    simulate_run.py
    planners.py
    mcts_planner.py
    visualize_run_pygame.py

Example from project root:
    python src/core/compare_planners.py \
        --config configs/basic_3target.yaml \
        --output-dir runs/comparison_basic \
        --seeds 7 8 9

Example from src/core:
    python compare_planners.py \
        --config ../../configs/basic_3target.yaml \
        --output-dir ../../runs/comparison_basic \
        --seeds 7 8 9

Outputs:
    output_dir/
      random_seed7.json
      greedy_uncertainty_seed7.json
      greedy_distance_seed7.json
      mcts_seed7.json
      summary_metrics.csv
      final_uncertainty_trace.png
      average_uncertainty_trace.png
      detections.png
      distance_traveled.png
      uncertainty_over_time_seed7.png
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from core.sim.simulate_run import (
    SimConfig,
    load_config_from_yaml,
    load_planner,
    make_simulation,
    save_run,
)


DEFAULT_PLANNERS = {
    "random": "random_planner.RandomPlanner",
    "greedy_uncertainty": "greedy_planners.GreedyUncertaintyPlanner",
    # "greedy_logdet": "greedy_planners.GreedyLogDetPlanner",
    # "greedy_distance": "greedy_planners.GreedyDistanceAwarePlanner",
    "mcts": "mcts_planner.MCTSPlanner",
}


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def run_planner_once(
    base_config: SimConfig,
    planner_name: str,
    planner_path: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one planner on one seed, save JSON, and return metrics."""

    config = replace(base_config, seed=seed)
    planner = load_planner(planner_path)

    sim = make_simulation(config=config, planner=planner)
    run = sim.run()

    run["metadata"]["planner_key"] = planner_name
    run["metadata"]["planner_path"] = planner_path
    run["metadata"]["seed"] = seed

    output_path = output_dir / f"{planner_name}_seed{seed}.json"
    save_run(run, output_path)

    metrics = summarize_run(run)
    metrics.update(
        {
            "planner": planner_name,
            "planner_path": planner_path,
            "seed": seed,
            "run_file": str(output_path),
        }
    )
    return metrics


def run_comparison(
    config: SimConfig,
    planners: dict[str, str],
    seeds: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run all planners across all seeds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []

    total_runs = len(planners) * len(seeds)
    run_idx = 0

    for seed in seeds:
        for planner_name, planner_path in planners.items():
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {planner_name} seed={seed}...")
            metrics = run_planner_once(
                base_config=config,
                planner_name=planner_name,
                planner_path=planner_path,
                seed=seed,
                output_dir=output_dir,
            )
            all_metrics.append(metrics)
            print(
                f"    final_trace={metrics['final_uncertainty_trace']:.2f} "
                f"avg_trace={metrics['avg_uncertainty_trace']:.2f} "
                f"detections={metrics['num_detections']} "
                f"distance={metrics['distance_traveled']:.2f}"
            )

    return all_metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Compute scalar metrics from one saved run dictionary."""

    history = run["history"]
    if not history:
        raise ValueError("Run has empty history.")

    times = np.array([frame["time"] for frame in history], dtype=float)
    traces = np.array(
        [frame["metrics"]["total_position_trace"] for frame in history],
        dtype=float,
    )
    logdets = np.array(
        [frame["metrics"]["total_position_logdet"] for frame in history],
        dtype=float,
    )

    detection_events = [frame for frame in history if frame.get("detections")]
    selected_tracks = [
        frame.get("selected_track_id")
        for frame in history
        if frame.get("event") == "choose_target"
    ]

    final_frame = history[-1]
    final_drone = final_frame["drone"]
    final_metrics = final_frame["metrics"]

    return {
        "final_time": float(final_frame["time"]),
        "num_frames": int(len(history)),
        "num_decisions": int(final_frame.get("decision_count", 0)),
        "num_detections": int(sum(len(frame.get("detections", [])) for frame in history)),
        "num_detection_events": int(len(detection_events)),
        "unique_detected_targets": int(
            len(
                {
                    target_id
                    for frame in history
                    for target_id in frame.get("detections", [])
                }
            )
        ),
        "final_uncertainty_trace": float(traces[-1]),
        "avg_uncertainty_trace": float(np.mean(traces)),
        "min_uncertainty_trace": float(np.min(traces)),
        "max_uncertainty_trace": float(np.max(traces)),
        "auc_uncertainty_trace": area_under_curve(times, traces),
        "final_uncertainty_logdet": float(logdets[-1]),
        "avg_uncertainty_logdet": float(np.mean(logdets)),
        "auc_uncertainty_logdet": area_under_curve(times, logdets),
        "final_num_lost": int(final_metrics.get("num_lost", 0)),
        "final_num_active": int(final_metrics.get("num_active", 0)),
        "distance_traveled": float(final_drone["distance_traveled"]),
        "selected_track_sequence": "-".join(str(x) for x in selected_tracks if x is not None),
    }


def area_under_curve(times: np.ndarray, values: np.ndarray) -> float:
    """Compute trapezoidal AUC with guards for repeated time stamps."""

    if len(times) < 2:
        return 0.0

    # Some frames, like choose_target, can have the same timestamp as the prior
    # frame. np.trapz handles this fine; this just keeps the intent explicit.
    return float(np.trapz(values, times))


def save_metrics_csv(metrics: list[dict[str, Any]], output_path: Path) -> None:
    if not metrics:
        return

    fieldnames = list(metrics[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Saved summary metrics to: {output_path}")


def print_aggregate_table(metrics: list[dict[str, Any]]) -> None:
    """Print mean/std summary by planner."""

    planners = sorted({row["planner"] for row in metrics})
    metric_names = [
        "final_uncertainty_trace",
        "avg_uncertainty_trace",
        "auc_uncertainty_trace",
        "num_detections",
        "final_num_lost",
        "distance_traveled",
    ]

    print("\nAggregate summary across seeds")
    print("=" * 96)
    header = f"{'planner':<22}" + "".join(f"{name:<28}" for name in metric_names)
    print(header)
    print("-" * 96)

    for planner in planners:
        rows = [row for row in metrics if row["planner"] == planner]
        line = f"{planner:<22}"
        for name in metric_names:
            vals = np.array([row[name] for row in rows], dtype=float)
            line += f"{np.mean(vals):.2f} ± {np.std(vals):.2f}".ljust(28)
        print(line)

    print("=" * 96)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_bar_plot(
    metrics: list[dict[str, Any]],
    metric_name: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    planners = sorted({row["planner"] for row in metrics})
    means = []
    stds = []

    for planner in planners:
        vals = np.array([row[metric_name] for row in metrics if row["planner"] == planner], dtype=float)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    plt.figure(figsize=(10, 5))
    x = np.arange(len(planners))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, planners, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_uncertainty_time_plot(
    output_dir: Path,
    planners: dict[str, str],
    seed: int,
    output_path: Path,
) -> None:
    """Plot total uncertainty over time for one representative seed."""

    plt.figure(figsize=(10, 5))

    for planner_name in planners:
        run_path = output_dir / f"{planner_name}_seed{seed}.json"
        if not run_path.exists():
            continue

        with run_path.open("r", encoding="utf-8") as f:
            run = json.load(f)

        history = run["history"]
        times = [frame["time"] for frame in history]
        traces = [frame["metrics"]["total_position_trace"] for frame in history]
        plt.plot(times, traces, label=planner_name)

    plt.xlabel("Mission time")
    plt.ylabel("Total position covariance trace")
    plt.title(f"Uncertainty over time, seed={seed}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_all_plots(metrics: list[dict[str, Any]], planners: dict[str, str], seeds: list[int], output_dir: Path) -> None:
    make_bar_plot(
        metrics,
        metric_name="final_uncertainty_trace",
        ylabel="Final total covariance trace",
        title="Final uncertainty by planner",
        output_path=output_dir / "final_uncertainty_trace.png",
    )
    make_bar_plot(
        metrics,
        metric_name="avg_uncertainty_trace",
        ylabel="Average total covariance trace",
        title="Average uncertainty by planner",
        output_path=output_dir / "average_uncertainty_trace.png",
    )
    make_bar_plot(
        metrics,
        metric_name="auc_uncertainty_trace",
        ylabel="AUC of total covariance trace",
        title="Time-integrated uncertainty by planner",
        output_path=output_dir / "auc_uncertainty_trace.png",
    )
    make_bar_plot(
        metrics,
        metric_name="num_detections",
        ylabel="Number of detections",
        title="Detections by planner",
        output_path=output_dir / "detections.png",
    )
    make_bar_plot(
        metrics,
        metric_name="distance_traveled",
        ylabel="Distance traveled",
        title="Distance traveled by planner",
        output_path=output_dir / "distance_traveled.png",
    )

    if seeds:
        make_uncertainty_time_plot(
            output_dir=output_dir,
            planners=planners,
            seed=seeds[0],
            output_path=output_dir / f"uncertainty_over_time_seed{seeds[0]}.png",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_planner_args(planner_args: list[str] | None) -> dict[str, str]:
    """Parse planner CLI values.

    Accepts values like:
        random=planners.RandomPlanner
        greedy=planners.GreedyUncertaintyPlanner
        mcts=mcts_planner.MCTSPlanner

    If omitted, DEFAULT_PLANNERS is used.
    """

    if not planner_args:
        return DEFAULT_PLANNERS.copy()

    planners: dict[str, str] = {}
    for item in planner_args:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = path.rsplit(".", 1)[-1]
        planners[name.strip()] = path.strip()

    return planners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument(
        "--planners",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional planner specs like name=module.Class. "
            "If omitted, random, greedy, and MCTS baselines are run."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run simulations and save CSV, but do not create PNG plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config_from_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    planners = parse_planner_args(args.planners)

    metrics = run_comparison(
        config=config,
        planners=planners,
        seeds=args.seeds,
        output_dir=output_dir,
    )

    save_metrics_csv(metrics, output_dir / "summary_metrics.csv")
    print_aggregate_table(metrics)

    if not args.no_plots:
        make_all_plots(metrics, planners, args.seeds, output_dir)


if __name__ == "__main__":
    main()
