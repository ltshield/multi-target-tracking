"""Run multiple planners and compare them using objective-aligned metrics.

This script is intended for comparing greedy, MCTS, warm MCTS, guided MCTS,
legacy/reference-style MCTS, and related planners across the same seeds.

Why this version exists
-----------------------
The reference/professor-style MCTS backs up negative terminal uncertainty. If we
judge it only by raw AUC or by a simple frame-average uncertainty, we may be
answering the wrong question. This script reports separate metric families:

1. Reference-replication metrics
   These answer: "Did this planner behave like the reference MCTS objective?"
   Primary metric: final_uncertainty_trace.

2. Tracking-quality metrics
   These answer: "Was tracking quality better over time?"
   Primary metric: normalized_auc_uncertainty_trace, which is
   integral(trace dt) / mission_duration.

3. Balanced metrics
   These answer: "Did it improve final uncertainty and time-average uncertainty
   without losing more targets?"

Important distinction
---------------------
- avg_uncertainty_trace is a frame average. It can be biased if planners create
  different numbers/timing of logged frames.
- normalized_auc_uncertainty_trace is the fair continuous-time average. Prefer
  this for comparing average tracking quality.

Example:
    python src/core/evaluation/compare_planners.py \
        --config configs/basic_3target.yaml \
        --output-dir runs/comparison_basic \
        --seeds 7 8 9 \
        --planners \
            greedy=core.planners.greedy_planners.GreedyDistanceAwarePlanner \
            legacy_mcts=core.planners.mcts_legacy_dwell_planner.LegacyDwellMDPMCTSPlanner
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
    "random": "core.planners.random_planner.RandomPlanner",
    "greedy": "core.planners.greedy_planners.GreedyDistanceAwarePlanner",
    "mcts": "core.planners.mcts_planner.MCTSPlanner",
    "warm_mcts": "core.planners.warm_mcts_planner.WarmMCTSPlanner",
    "guided_mcts": "core.planners.guided_track_scorer_mcts_modular.GuidedTrackScorerMCTSPlanner",
}


MCTS_LIKE_PLANNERS = {
    "mcts",
    "warm_mcts",
    "guided_mcts",
    "realtime_mcts",
    "legacy_mcts",
    "legacy_dwell_mcts",
    "reference_mcts",
    "mdp_mcts",
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
    """Run one planner on one seed, save JSON, and return scalar metrics."""

    config = replace(base_config, seed=seed)
    planner = load_planner(planner_path)

    start = time.perf_counter()
    sim = make_simulation(config=config, planner=planner)
    run = sim.run()
    runtime_seconds = time.perf_counter() - start

    run["metadata"]["planner_key"] = planner_name
    run["metadata"]["planner_path"] = planner_path
    run["metadata"]["seed"] = seed
    run["metadata"]["runtime_seconds"] = float(runtime_seconds)

    output_path = output_dir / f"{planner_name}_seed{seed}.json"
    save_run(run, output_path)

    metrics = summarize_run(run)
    metrics.update(
        {
            "planner": planner_name,
            "planner_path": planner_path,
            "seed": seed,
            "run_file": str(output_path),
            "runtime_seconds": float(runtime_seconds),
        }
    )

    return metrics


def run_comparison(
    config: SimConfig,
    planners: dict[str, str],
    seeds: list[int],
    output_dir: Path,
    continue_on_error: bool = False,
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

            try:
                metrics = run_planner_once(
                    base_config=config,
                    planner_name=planner_name,
                    planner_path=planner_path,
                    seed=seed,
                    output_dir=output_dir,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise

                print(f"    ERROR: {planner_name} seed={seed} failed: {exc}")
                all_metrics.append(
                    {
                        "planner": planner_name,
                        "planner_path": planner_path,
                        "seed": seed,
                        "run_file": "",
                        "error": str(exc),
                        "runtime_seconds": math.nan,
                    }
                )
                continue

            all_metrics.append(metrics)
            print(
                f"    final_trace={metrics['final_uncertainty_trace']:.2f} "
                f"norm_auc_trace={metrics['normalized_auc_uncertainty_trace']:.2f} "
                f"frame_avg_trace={metrics['avg_uncertainty_trace']:.2f} "
                f"lost={metrics['final_num_lost']} "
                f"norm_lost={metrics['normalized_auc_num_lost']:.2f} "
                f"detections={metrics['num_detections']} "
                f"decisions={metrics['num_decisions']} "
                f"distance={metrics['distance_traveled']:.2f} "
                f"runtime={metrics['runtime_seconds']:.2f}s"
            )

    return all_metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Compute scalar metrics from one saved run dictionary.

    The two main uncertainty metrics are:

    - final_uncertainty_trace: matches a terminal-uncertainty MCTS objective.
    - normalized_auc_uncertainty_trace: fair continuous-time average uncertainty.

    avg_uncertainty_trace is retained as a diagnostic only because it is a mean
    over frames rather than a time-weighted mean.
    """

    history = run["history"]
    if not history:
        raise ValueError("Run has empty history.")

    times = np.array([frame["time"] for frame in history], dtype=float)
    mission_duration = float(max(1e-9, times[-1] - times[0]))

    traces = np.array(
        [frame["metrics"]["total_position_trace"] for frame in history],
        dtype=float,
    )
    active_traces = np.array(
        [frame["metrics"].get("active_position_trace", np.nan) for frame in history],
        dtype=float,
    )
    logdets = np.array(
        [frame["metrics"]["total_position_logdet"] for frame in history],
        dtype=float,
    )
    num_lost_series = np.array(
        [frame["metrics"].get("num_lost", 0) for frame in history],
        dtype=float,
    )
    num_active_series = np.array(
        [frame["metrics"].get("num_active", 0) for frame in history],
        dtype=float,
    )

    detection_events = [frame for frame in history if frame.get("detections")]
    selected_tracks = [
        frame.get("selected_track_id")
        for frame in history
        if frame.get("event") == "choose_target"
    ]
    selected_track_sequence = "-".join(str(x) for x in selected_tracks if x is not None)

    final_frame = history[-1]
    first_frame = history[0]
    final_drone = final_frame["drone"]
    final_metrics = final_frame["metrics"]
    first_metrics = first_frame.get("metrics", {})

    unique_selected_targets = {int(x) for x in selected_tracks if x is not None}
    unique_detected_targets = {
        int(target_id)
        for frame in history
        for target_id in frame.get("detections", [])
    }
    target_ids_seen = {
        int(target_id)
        for frame in history
        for target_id in frame.get("targets", {}).keys()
    }

    initial_target_count = int(
        max(
            1,
            len(target_ids_seen),
            int(first_metrics.get("num_active", 0)) + int(first_metrics.get("num_lost", 0)),
            int(final_metrics.get("num_active", 0)) + int(final_metrics.get("num_lost", 0)),
        )
    )

    auc_trace = area_under_curve(times, traces)
    auc_active_trace = area_under_curve(times, active_traces)
    auc_logdet = area_under_curve(times, logdets)
    auc_lost = area_under_curve(times, num_lost_series)
    auc_active_count = area_under_curve(times, num_active_series)

    normalized_auc_trace = auc_trace / mission_duration
    normalized_auc_active_trace = auc_active_trace / mission_duration
    normalized_auc_logdet = auc_logdet / mission_duration
    normalized_auc_lost = auc_lost / mission_duration
    normalized_auc_active_count = auc_active_count / mission_duration

    num_detections = int(sum(len(frame.get("detections", [])) for frame in history))

    return {
        "final_time": float(final_frame["time"]),
        "mission_duration": float(mission_duration),
        "num_frames": int(len(history)),
        "initial_target_count": int(initial_target_count),
        "num_decisions": int(final_frame.get("decision_count", 0)),
        "num_detections": num_detections,
        "num_detection_events": int(len(detection_events)),
        "unique_detected_targets": int(len(unique_detected_targets)),
        "unique_selected_targets": int(len(unique_selected_targets)),

        # Reference-replication metrics.
        "final_uncertainty_trace": float(traces[-1]),
        "final_uncertainty_trace_per_target": float(traces[-1] / initial_target_count),
        "final_active_uncertainty_trace": float(active_traces[-1]),
        "final_uncertainty_logdet": float(logdets[-1]),

        # Fair tracking-quality metrics.
        "auc_uncertainty_trace": float(auc_trace),
        "normalized_auc_uncertainty_trace": float(normalized_auc_trace),
        "normalized_auc_uncertainty_trace_per_target": float(
            normalized_auc_trace / initial_target_count
        ),
        "auc_active_uncertainty_trace": float(auc_active_trace),
        "normalized_auc_active_uncertainty_trace": float(normalized_auc_active_trace),
        "normalized_auc_active_uncertainty_trace_per_target": float(
            normalized_auc_active_trace / initial_target_count
        ),
        "auc_uncertainty_logdet": float(auc_logdet),
        "normalized_auc_uncertainty_logdet": float(normalized_auc_logdet),

        # Frame-average diagnostics.
        "avg_uncertainty_trace": float(np.mean(traces)),
        "avg_active_uncertainty_trace": float(np.nanmean(active_traces)),
        "avg_uncertainty_logdet": float(np.mean(logdets)),
        "min_uncertainty_trace": float(np.min(traces)),
        "max_uncertainty_trace": float(np.max(traces)),

        # Lost/active target metrics.
        "final_num_lost": int(final_metrics.get("num_lost", 0)),
        "max_num_lost": int(np.max(num_lost_series)),
        "avg_num_lost": float(np.mean(num_lost_series)),
        "auc_num_lost": float(auc_lost),
        "normalized_auc_num_lost": float(normalized_auc_lost),
        "final_num_active": int(final_metrics.get("num_active", 0)),
        "min_num_active": int(np.min(num_active_series)),
        "avg_num_active": float(np.mean(num_active_series)),
        "auc_num_active": float(auc_active_count),
        "normalized_auc_num_active": float(normalized_auc_active_count),

        # Operational metrics.
        "distance_traveled": float(final_drone["distance_traveled"]),
        "distance_per_detection": float(final_drone["distance_traveled"] / max(1, num_detections)),
        "selected_track_sequence": selected_track_sequence,
    }


def area_under_curve(times: np.ndarray, values: np.ndarray) -> float:
    """Compute trapezoidal AUC with guards for nonfinite values."""

    if len(times) < 2:
        return 0.0
    if len(values) != len(times):
        raise ValueError("times and values must have the same length.")

    if np.any(~np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # np.trapezoid is newer; np.trapz keeps compatibility with older NumPy.
    return float(np.trapz(values, times))


def save_metrics_csv(metrics: list[dict[str, Any]], output_path: Path) -> None:
    if not metrics:
        return

    fieldnames = sorted({key for row in metrics for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Saved summary metrics to: {output_path}")


def print_aggregate_table(metrics: list[dict[str, Any]]) -> None:
    """Print mean/std summary by planner."""

    valid_rows = [row for row in metrics if "error" not in row]
    if not valid_rows:
        print("No successful runs to summarize.")
        return

    planners = sorted({row["planner"] for row in valid_rows})
    metric_names = [
        "final_uncertainty_trace",
        "normalized_auc_uncertainty_trace",
        "avg_uncertainty_trace",
        "final_num_lost",
        "normalized_auc_num_lost",
        "num_detections",
        "distance_traveled",
        "num_decisions",
        "runtime_seconds",
    ]

    print("\nAggregate summary across seeds")
    print("=" * 170)
    header = f"{'planner':<22}" + "".join(f"{name:<28}" for name in metric_names)
    print(header)
    print("-" * 170)

    for planner in planners:
        rows = [row for row in valid_rows if row["planner"] == planner]
        line = f"{planner:<22}"
        for name in metric_names:
            vals = np.array([row.get(name, np.nan) for row in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            cell = "n/a" if len(vals) == 0 else f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
            line += cell.ljust(28)
        print(line)

    print("=" * 170)


# ---------------------------------------------------------------------------
# MCTS-vs-greedy analysis
# ---------------------------------------------------------------------------

def analyze_mcts_vs_greedy(
    metrics: list[dict[str, Any]],
    output_path: Path,
    greedy_name: str = "greedy",
    terminal_tolerance: float = 0.0,
    auc_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Compare MCTS-style planners against greedy on each seed.

    Three win conditions are reported:

    - reference_terminal_win:
        final uncertainty improves and final lost count does not worsen.

    - tracking_quality_win:
        normalized AUC uncertainty improves, final uncertainty does not worsen,
        and final lost count does not worsen.

    - balanced_win:
        final uncertainty and normalized AUC both improve, with no worse final
        lost count.
    """

    valid_rows = [row for row in metrics if "error" not in row]
    by_seed_planner = {
        (int(row["seed"]), str(row["planner"])): row
        for row in valid_rows
    }

    seeds = sorted({int(row["seed"]) for row in valid_rows})
    planner_names = sorted({str(row["planner"]) for row in valid_rows})
    candidate_mcts_names = [
        name
        for name in planner_names
        if name in MCTS_LIKE_PLANNERS or "mcts" in name.lower()
    ]

    comparisons: list[dict[str, Any]] = []

    for seed in seeds:
        greedy = by_seed_planner.get((seed, greedy_name))
        if greedy is None:
            continue

        for planner_name in candidate_mcts_names:
            if planner_name == greedy_name:
                continue
            candidate = by_seed_planner.get((seed, planner_name))
            if candidate is None:
                continue

            final_delta = float(candidate["final_uncertainty_trace"]) - float(greedy["final_uncertainty_trace"])
            final_per_target_delta = float(candidate["final_uncertainty_trace_per_target"]) - float(
                greedy["final_uncertainty_trace_per_target"]
            )
            norm_auc_delta = float(candidate["normalized_auc_uncertainty_trace"]) - float(
                greedy["normalized_auc_uncertainty_trace"]
            )
            norm_auc_per_target_delta = float(candidate["normalized_auc_uncertainty_trace_per_target"]) - float(
                greedy["normalized_auc_uncertainty_trace_per_target"]
            )
            raw_auc_delta = float(candidate["auc_uncertainty_trace"]) - float(greedy["auc_uncertainty_trace"])
            frame_avg_delta = float(candidate["avg_uncertainty_trace"]) - float(greedy["avg_uncertainty_trace"])
            active_norm_auc_delta = float(candidate["normalized_auc_active_uncertainty_trace"]) - float(
                greedy["normalized_auc_active_uncertainty_trace"]
            )
            lost_delta = int(candidate["final_num_lost"]) - int(greedy["final_num_lost"])
            norm_lost_auc_delta = float(candidate["normalized_auc_num_lost"]) - float(
                greedy["normalized_auc_num_lost"]
            )
            detection_delta = int(candidate["num_detections"]) - int(greedy["num_detections"])
            distance_delta = float(candidate["distance_traveled"]) - float(greedy["distance_traveled"])

            reference_terminal_win = final_delta < -terminal_tolerance and lost_delta <= 0
            tracking_quality_win = (
                norm_auc_delta < -auc_tolerance
                and final_delta <= terminal_tolerance
                and lost_delta <= 0
            )
            balanced_win = (
                final_delta < -terminal_tolerance
                and norm_auc_delta < -auc_tolerance
                and lost_delta <= 0
            )
            terminal_only_tradeoff = (
                reference_terminal_win
                and norm_auc_delta >= -auc_tolerance
            )
            tracking_only_tradeoff = (
                tracking_quality_win
                and final_delta >= -terminal_tolerance
            )

            comparisons.append(
                {
                    "seed": seed,
                    "planner": planner_name,
                    "baseline": greedy_name,
                    "planner_final_uncertainty": candidate["final_uncertainty_trace"],
                    "greedy_final_uncertainty": greedy["final_uncertainty_trace"],
                    "delta_final_uncertainty": final_delta,
                    "delta_final_uncertainty_per_target": final_per_target_delta,
                    "planner_normalized_auc_uncertainty": candidate["normalized_auc_uncertainty_trace"],
                    "greedy_normalized_auc_uncertainty": greedy["normalized_auc_uncertainty_trace"],
                    "delta_normalized_auc_uncertainty": norm_auc_delta,
                    "delta_normalized_auc_uncertainty_per_target": norm_auc_per_target_delta,
                    "planner_raw_auc_uncertainty": candidate["auc_uncertainty_trace"],
                    "greedy_raw_auc_uncertainty": greedy["auc_uncertainty_trace"],
                    "delta_raw_auc_uncertainty": raw_auc_delta,
                    "planner_frame_avg_uncertainty": candidate["avg_uncertainty_trace"],
                    "greedy_frame_avg_uncertainty": greedy["avg_uncertainty_trace"],
                    "delta_frame_avg_uncertainty": frame_avg_delta,
                    "delta_normalized_auc_active_uncertainty": active_norm_auc_delta,
                    "planner_lost": candidate["final_num_lost"],
                    "greedy_lost": greedy["final_num_lost"],
                    "delta_lost": lost_delta,
                    "delta_normalized_auc_lost": norm_lost_auc_delta,
                    "planner_detections": candidate["num_detections"],
                    "greedy_detections": greedy["num_detections"],
                    "delta_detections": detection_delta,
                    "planner_distance": candidate["distance_traveled"],
                    "greedy_distance": greedy["distance_traveled"],
                    "delta_distance": distance_delta,
                    "reference_terminal_win": reference_terminal_win,
                    "tracking_quality_win": tracking_quality_win,
                    "balanced_win": balanced_win,
                    "terminal_only_tradeoff": terminal_only_tradeoff,
                    "tracking_only_tradeoff": tracking_only_tradeoff,
                    # Backward-compatible alias for older scripts.
                    "beats_greedy_primary": tracking_quality_win,
                    "possible_long_term_planning_win": terminal_only_tradeoff,
                }
            )

    save_dicts_csv(comparisons, output_path)
    print_mcts_vs_greedy_summary(comparisons)
    return comparisons


def print_mcts_vs_greedy_summary(comparisons: list[dict[str, Any]]) -> None:
    if not comparisons:
        print("\nNo MCTS-vs-greedy comparisons were produced.")
        return

    planners = sorted({row["planner"] for row in comparisons})

    print("\nMCTS-style planners vs greedy")
    print("=" * 132)
    print(
        "terminal = reference-style final uncertainty win | "
        "tracking = normalized-AUC win | balanced = both"
    )

    for planner in planners:
        rows = [row for row in comparisons if row["planner"] == planner]
        terminal_wins = sum(1 for row in rows if row["reference_terminal_win"])
        tracking_wins = sum(1 for row in rows if row["tracking_quality_win"])
        balanced_wins = sum(1 for row in rows if row["balanced_win"])
        terminal_only = sum(1 for row in rows if row["terminal_only_tradeoff"])
        tracking_only = sum(1 for row in rows if row["tracking_only_tradeoff"])

        norm_auc_deltas = np.array([row["delta_normalized_auc_uncertainty"] for row in rows], dtype=float)
        final_deltas = np.array([row["delta_final_uncertainty"] for row in rows], dtype=float)
        lost_deltas = np.array([row["delta_lost"] for row in rows], dtype=float)

        print(
            f"{planner:<24} "
            f"terminal={terminal_wins}/{len(rows)} | "
            f"tracking={tracking_wins}/{len(rows)} | "
            f"balanced={balanced_wins}/{len(rows)} | "
            f"terminal_only={terminal_only}/{len(rows)} | "
            f"tracking_only={tracking_only}/{len(rows)} | "
            f"mean_delta_final={np.mean(final_deltas):.2f} | "
            f"mean_delta_norm_auc={np.mean(norm_auc_deltas):.2f} | "
            f"mean_delta_lost={np.mean(lost_deltas):.2f}"
        )

    print("=" * 132)


def save_dicts_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved analysis CSV to: {output_path}")


# ---------------------------------------------------------------------------
# Plot loading helpers
# ---------------------------------------------------------------------------

def load_run(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_path_for(output_dir: Path, planner_name: str, seed: int) -> Path:
    return output_dir / f"{planner_name}_seed{seed}.json"


def available_successful_planners(output_dir: Path, planners: dict[str, str], seed: int) -> list[str]:
    return [
        planner_name
        for planner_name in planners
        if run_path_for(output_dir, planner_name, seed).exists()
    ]


def extract_series(run: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract common time series from a run."""

    history = run["history"]
    times = np.array([frame["time"] for frame in history], dtype=float)

    if key == "total_uncertainty":
        values = np.array([frame["metrics"]["total_position_trace"] for frame in history], dtype=float)
    elif key == "active_uncertainty":
        values = np.array([frame["metrics"].get("active_position_trace", np.nan) for frame in history], dtype=float)
    elif key == "num_lost":
        values = np.array([frame["metrics"].get("num_lost", 0) for frame in history], dtype=float)
    elif key == "num_active":
        values = np.array([frame["metrics"].get("num_active", 0) for frame in history], dtype=float)
    elif key == "selected_track":
        values = np.array(
            [
                np.nan if frame.get("selected_track_id") is None else float(frame.get("selected_track_id"))
                for frame in history
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown series key: {key}")

    return times, values


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
    valid_rows = [row for row in metrics if "error" not in row]
    if not valid_rows:
        return

    planners = sorted({row["planner"] for row in valid_rows})
    means: list[float] = []
    stds: list[float] = []

    for planner in planners:
        vals = np.array(
            [row[metric_name] for row in valid_rows if row["planner"] == planner and metric_name in row],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)) if len(vals) else 0.0)
        stds.append(float(np.std(vals)) if len(vals) else 0.0)

    plt.figure(figsize=(11, 5))
    x = np.arange(len(planners))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, planners, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_final_metric_bar_charts(metrics: list[dict[str, Any]], output_dir: Path) -> None:
    specs = [
        (
            "final_uncertainty_trace",
            "Final total covariance trace",
            "Reference-style terminal uncertainty by planner",
            "final_uncertainty_trace.png",
        ),
        (
            "normalized_auc_uncertainty_trace",
            "Time-normalized AUC covariance trace",
            "Fair continuous-time average uncertainty by planner",
            "normalized_auc_uncertainty_trace.png",
        ),
        (
            "avg_uncertainty_trace",
            "Frame-average total covariance trace",
            "Frame-average uncertainty diagnostic by planner",
            "average_uncertainty_trace.png",
        ),
        (
            "auc_uncertainty_trace",
            "Raw AUC of total covariance trace",
            "Raw time-integrated uncertainty by planner",
            "auc_uncertainty_trace.png",
        ),
        (
            "final_num_lost",
            "Final number of lost targets",
            "Final lost targets by planner",
            "lost_targets.png",
        ),
        (
            "normalized_auc_num_lost",
            "Time-normalized lost target count",
            "Continuous-time average lost targets by planner",
            "normalized_lost_targets.png",
        ),
        (
            "num_detections",
            "Number of detections",
            "Detections by planner",
            "detections.png",
        ),
        (
            "distance_traveled",
            "Distance traveled",
            "Distance traveled by planner",
            "distance_traveled.png",
        ),
        (
            "num_decisions",
            "Number of decisions",
            "Decisions by planner",
            "num_decisions.png",
        ),
        (
            "runtime_seconds",
            "Runtime seconds",
            "Runtime by planner",
            "runtime_seconds.png",
        ),
    ]

    for metric_name, ylabel, title, filename in specs:
        make_bar_plot(
            metrics=metrics,
            metric_name=metric_name,
            ylabel=ylabel,
            title=title,
            output_path=output_dir / filename,
        )


def make_uncertainty_time_plot(
    output_dir: Path,
    planners: dict[str, str],
    seed: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5))
    for planner_name in available_successful_planners(output_dir, planners, seed):
        run = load_run(run_path_for(output_dir, planner_name, seed))
        times, traces = extract_series(run, "total_uncertainty")
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


def make_active_lost_time_plot(
    output_dir: Path,
    planners: dict[str, str],
    seed: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5))
    for planner_name in available_successful_planners(output_dir, planners, seed):
        run = load_run(run_path_for(output_dir, planner_name, seed))
        times, active = extract_series(run, "num_active")
        _, lost = extract_series(run, "num_lost")
        plt.plot(times, active, label=f"{planner_name} active")
        plt.plot(times, lost, linestyle="--", label=f"{planner_name} lost")

    plt.xlabel("Mission time")
    plt.ylabel("Target count")
    plt.title(f"Active/lost targets over time, seed={seed}")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_selected_target_time_plot(
    output_dir: Path,
    planners: dict[str, str],
    seed: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 5))
    for planner_name in available_successful_planners(output_dir, planners, seed):
        run = load_run(run_path_for(output_dir, planner_name, seed))
        times, selected = extract_series(run, "selected_track")
        mask = np.isfinite(selected)
        if np.any(mask):
            plt.step(times[mask], selected[mask], where="post", label=planner_name)

    plt.xlabel("Mission time")
    plt.ylabel("Selected track ID")
    plt.title(f"Selected target over time, seed={seed}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_trajectory_plot_for_run(run: dict[str, Any], title: str, output_path: Path) -> None:
    history = run["history"]
    if not history:
        return

    drone_xy = np.array([frame["drone"]["position"] for frame in history], dtype=float)
    target_ids = sorted({int(target_id) for frame in history for target_id in frame.get("targets", {}).keys()})

    plt.figure(figsize=(7, 7))
    plt.plot(drone_xy[:, 0], drone_xy[:, 1], label="UAV")
    plt.scatter(drone_xy[0, 0], drone_xy[0, 1], marker="o", label="UAV start")
    plt.scatter(drone_xy[-1, 0], drone_xy[-1, 1], marker="x", label="UAV end")

    for target_id in target_ids:
        xy = []
        for frame in history:
            target = frame.get("targets", {}).get(str(target_id))
            if target is not None:
                xy.append(target["position"])
        if xy:
            xy_arr = np.array(xy, dtype=float)
            plt.plot(xy_arr[:, 0], xy_arr[:, 1], linestyle="--", label=f"target {target_id}")
            plt.scatter(xy_arr[0, 0], xy_arr[0, 1], marker="o")
            plt.scatter(xy_arr[-1, 0], xy_arr[-1, 1], marker="x")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot: {output_path}")


def make_trajectory_plots(output_dir: Path, planners: dict[str, str], seed: int) -> None:
    traj_dir = output_dir / f"trajectories_seed{seed}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for planner_name in available_successful_planners(output_dir, planners, seed):
        run = load_run(run_path_for(output_dir, planner_name, seed))
        make_trajectory_plot_for_run(
            run=run,
            title=f"UAV and target trajectories: {planner_name}, seed={seed}",
            output_path=traj_dir / f"{planner_name}_trajectory_seed{seed}.png",
        )


def make_all_plots(
    metrics: list[dict[str, Any]],
    planners: dict[str, str],
    seeds: list[int],
    output_dir: Path,
) -> None:
    make_final_metric_bar_charts(metrics, output_dir)
    if not seeds:
        return

    representative_seed = seeds[0]
    make_uncertainty_time_plot(
        output_dir=output_dir,
        planners=planners,
        seed=representative_seed,
        output_path=output_dir / f"uncertainty_over_time_seed{representative_seed}.png",
    )
    make_active_lost_time_plot(
        output_dir=output_dir,
        planners=planners,
        seed=representative_seed,
        output_path=output_dir / f"active_lost_over_time_seed{representative_seed}.png",
    )
    make_selected_target_time_plot(
        output_dir=output_dir,
        planners=planners,
        seed=representative_seed,
        output_path=output_dir / f"selected_target_over_time_seed{representative_seed}.png",
    )
    make_trajectory_plots(output_dir=output_dir, planners=planners, seed=representative_seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_planner_args(planner_args: list[str] | None) -> dict[str, str]:
    """Parse planner CLI values.

    Accepts values like:
        greedy=core.planners.greedy_planners.GreedyDistanceAwarePlanner
        legacy_mcts=core.planners.mcts_legacy_dwell_planner.LegacyDwellMDPMCTSPlanner
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

        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid planner spec with empty name: {item}")
        if not path:
            raise ValueError(f"Invalid planner spec with empty path: {item}")
        planners[name] = path

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
        help="Optional planner specs like name=module.Class.",
    )
    parser.add_argument(
        "--greedy-name",
        type=str,
        default="greedy",
        help="Planner name to use as the greedy baseline for MCTS-vs-greedy analysis.",
    )
    parser.add_argument(
        "--terminal-tolerance",
        type=float,
        default=0.0,
        help="Small tolerance for final-uncertainty win comparisons.",
    )
    parser.add_argument(
        "--auc-tolerance",
        type=float,
        default=0.0,
        help="Small tolerance for normalized-AUC win comparisons.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run simulations and save CSVs, but do not create PNG plots.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other planners/seeds if one planner fails.",
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
        continue_on_error=args.continue_on_error,
    )

    save_metrics_csv(metrics, output_dir / "summary_metrics.csv")
    print_aggregate_table(metrics)

    analyze_mcts_vs_greedy(
        metrics=metrics,
        output_path=output_dir / "mcts_vs_greedy_analysis.csv",
        greedy_name=args.greedy_name,
        terminal_tolerance=float(args.terminal_tolerance),
        auc_tolerance=float(args.auc_tolerance),
    )

    if not args.no_plots:
        make_all_plots(metrics=metrics, planners=planners, seeds=args.seeds, output_dir=output_dir)


if __name__ == "__main__":
    main()
