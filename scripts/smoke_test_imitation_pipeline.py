"""Smoke test for imitation-learning pipeline.

Run from project root:

    python scripts/smoke_test_imitation_pipeline.py

This test:
1. Generates a tiny imitation dataset from MCTS.
2. Checks train/val/test files exist.
3. Checks tensor shapes.
4. Trains TrackScorerNet for a few epochs.
5. Loads the trained checkpoint.
6. Runs one simulation with the learned planner.
"""

from __future__ import annotations
from datetime import datetime

import shutil
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG = PROJECT_ROOT / "configs" / "basic_3target.yaml"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = PROJECT_ROOT / "data" / f"smoke_imitation_{RUN_ID}"
MODEL_PATH = PROJECT_ROOT / "models" / f"smoke_track_scorer_{RUN_ID}.pt"

def run(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100)

    subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
    )


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Expected file does not exist: {path}")


def inspect_dataset(path: Path) -> None:
    print(f"\nInspecting dataset: {path}")
    data = torch.load(path, map_location="cpu")

    required = [
        "global_features",
        "track_features",
        "action_masks",
        "labels",
        "track_ids",
        "scenario_seeds",
        "extractor",
        "max_tracks",
        "global_dim",
        "track_dim",
        "num_examples",
    ]

    for key in required:
        if key not in data:
            raise AssertionError(f"{path} is missing key: {key}")

    global_features = data["global_features"]
    track_features = data["track_features"]
    action_masks = data["action_masks"]
    labels = data["labels"]

    print(f"  global_features: {tuple(global_features.shape)}")
    print(f"  track_features:  {tuple(track_features.shape)}")
    print(f"  action_masks:    {tuple(action_masks.shape)}")
    print(f"  labels:          {tuple(labels.shape)}")
    print(f"  extractor:       {data['extractor']}")
    print(f"  max_tracks:      {data['max_tracks']}")
    print(f"  global_dim:      {data['global_dim']}")
    print(f"  track_dim:       {data['track_dim']}")
    print(f"  num_examples:    {data['num_examples']}")

    if global_features.ndim != 2:
        raise AssertionError("global_features should have shape [N, G].")
    if track_features.ndim != 3:
        raise AssertionError("track_features should have shape [N, K, T].")
    if action_masks.ndim != 2:
        raise AssertionError("action_masks should have shape [N, K].")
    if labels.ndim != 1:
        raise AssertionError("labels should have shape [N].")

    n = global_features.shape[0]

    if n == 0:
        raise AssertionError("Dataset has zero examples.")
    if track_features.shape[0] != n:
        raise AssertionError("track_features N does not match global_features N.")
    if action_masks.shape[0] != n:
        raise AssertionError("action_masks N does not match global_features N.")
    if labels.shape[0] != n:
        raise AssertionError("labels N does not match global_features N.")

    rows = torch.arange(n)
    valid_labels = action_masks[rows, labels].bool()

    if not bool(valid_labels.all()):
        bad = int((~valid_labels).sum().item())
        raise AssertionError(f"{bad} labels point to masked/invalid actions.")

    print("  label validity:  OK")


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"Could not find {BASE_CONFIG}. "
            "Update BASE_CONFIG in this script if your config path is different."
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    # 1. Generate tiny imitation data.
    #
    # Notes:
    # - We relax --max-lost-for-keep to 6 for smoke testing so the test checks
    #   plumbing rather than expert quality.
    # - Use tiny episode counts and low MCTS iterations so this finishes quickly.
    run(
        [
            sys.executable,
            "src/core/learning/generate_imitation_data.py",
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(OUTPUT_DIR),
            "--train-episodes",
            "3",
            "--val-episodes",
            "2",
            "--test-episodes",
            "2",
            "--min-targets",
            "2",
            "--max-targets",
            "3",
            "--max-tracks",
            "6",
            "--extractor",
            "cartesian",
            "--teacher-iterations",
            "20",
            "--teacher-iterations-per-second",
            "20",
            "--max-background-iterations-per-call",
            "20",
            "--teacher-max-depth",
            "2",
            "--max-lost-for-keep",
            "6",
            "--min-samples-per-episode",
            "1",
        ]
    )

    train_pt = OUTPUT_DIR / "train.pt"
    val_pt = OUTPUT_DIR / "val.pt"
    test_pt = OUTPUT_DIR / "test.pt"

    assert_exists(train_pt)
    assert_exists(val_pt)
    assert_exists(test_pt)

    inspect_dataset(train_pt)
    inspect_dataset(val_pt)
    inspect_dataset(test_pt)

    # 2. Train for just a few epochs.
    run(
        [
            sys.executable,
            "src/core/learning/train_track_scorer_imitation.py",
            "--train",
            str(train_pt),
            "--val",
            str(val_pt),
            "--model-output",
            str(MODEL_PATH),
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--hidden-dim",
            "32",
            "--lr",
            "1e-3",
            "--cpu",
        ]
    )

    assert_exists(MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    print("\nLoaded checkpoint:")
    print(f"  extractor:          {checkpoint['extractor']}")
    print(f"  max_tracks:         {checkpoint['max_tracks']}")
    print(f"  global_dim:         {checkpoint['global_dim']}")
    print(f"  track_dim:          {checkpoint['track_dim']}")
    print(f"  num_train_examples: {checkpoint['num_train_examples']}")
    print(f"  num_val_examples:   {checkpoint['num_val_examples']}")

    # 3. Quick learned-planner simulation.
    #
    # This requires TrackScorerPlanner to accept model_path or default to the
    # smoke model. If your load_planner only constructs with no args, temporarily
    # copy this smoke model to models/track_scorer.pt.
    default_model = PROJECT_ROOT / "models" / "track_scorer_imitation.pt"
    backup_model = PROJECT_ROOT / "models" / "track_scorer_imitation.pt.bak"

    restored_backup = False

    if default_model.exists():
        shutil.copy2(default_model, backup_model)
        restored_backup = True

    shutil.copy2(MODEL_PATH, default_model)

    try:
        run(
            [
                sys.executable,
                "src/core/evaluation/compare_planners.py",
                "--config",
                str(BASE_CONFIG),
                "--output-dir",
                str(PROJECT_ROOT / "runs" / "smoke_imitation_compare"),
                "--seeds",
                "7",
                "--planners",
                "learned=core.planners.track_scorer_planner_modular.TrackScorerPlanner",
                "--no-plots",
            ]
        )
    finally:
        if restored_backup:
            shutil.move(backup_model, default_model)
        elif default_model.exists():
            default_model.unlink()

    print("\n" + "=" * 100)
    print("SMOKE TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()