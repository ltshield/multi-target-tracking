"""Train a per-track neural scorer on randomized MCTS demonstrations.

This is the recommended training-data path.

It samples many randomized worlds from a base config, uses MCTS as the teacher,
filters out bad teacher episodes, and trains a shared-weight network:

    global belief context + candidate track -> score

Example:
    python src/core/train_track_scorer_randomized.py ^
      --base-config configs/basic_3target.yaml ^
      --dataset data/randomized_track_scorer_demos.npz ^
      --model-output models/track_scorer.pt ^
      --episodes 500 ^
      --extractor polar ^
      --max-lost-for-keep 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.sim.simulate_run import load_config_from_yaml
from core.learning.mtt_gym_env import MTTTargetSelectionEnv
from planners.mcts_planner_realtime import MCTSPlanner
from core.sim.scenario_sampler import ScenarioSamplerConfig, sample_random_scenario
from core.learning.track_feature_extractors import load_extractor, track_id_to_slot
from core.learning.track_scorer_model import TrackScorerNet

def collect_dataset(args):
    """Collect MCTS teacher demonstrations from randomized scenarios.

    Important:
        Samples are first stored in temporary per-episode lists. The episode's
        samples are only added to the final dataset if the episode passes the
        quality filters:
            - num_lost <= max_lost_for_keep
            - final objective metric <= max_final_metric_for_keep
            - episode has at least min_samples_per_episode samples

    This prevents the neural scorer from imitating teacher trajectories that
    lost too many targets or otherwise performed poorly.
    """

    base_config = load_config_from_yaml(Path(args.base_config))
    rng = np.random.default_rng(args.seed)
    extractor = load_extractor(args.extractor)

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

    global_features = []
    track_features = []
    action_masks = []
    labels = []
    metadata = []

    kept_episodes = 0
    discarded_episodes = 0

    for ep in range(args.episodes):
        scenario_seed = args.seed + ep
        config = sample_random_scenario(
            base_config=base_config,
            rng=rng,
            sampler_config=sampler_config,
            seed=scenario_seed,
        )

        env = MTTTargetSelectionEnv(config=config, max_targets=args.max_targets)
        obs, info = env.reset(seed=scenario_seed)

        teacher = MCTSPlanner(iterations=args.teacher_iterations)

        # Episode-local buffers. Only moved into final dataset if the episode
        # passes the quality filters after it finishes.
        episode_global_features = []
        episode_track_features = []
        episode_action_masks = []
        episode_labels = []
        episode_metadata = []

        done = False
        steps = 0
        skipped_episode = False
        skip_reason = ""

        while not done:
            batch = extractor.build_batch(
                env.tracks,
                env.drone,
                max_tracks=args.max_targets,
            )

            try:
                teacher_track_id = teacher.choose_track(
                    tracks=env.tracks,
                    drone=env.drone,
                    targets=env.targets,
                    rng=env.rng,
                )
                action_slot = track_id_to_slot(batch.track_ids, teacher_track_id)
            except Exception as exc:
                skipped_episode = True
                skip_reason = f"teacher_error: {exc}"
                break

            episode_global_features.append(batch.global_features)
            episode_track_features.append(batch.track_features)
            episode_action_masks.append(batch.action_mask)
            episode_labels.append(action_slot)
            episode_metadata.append([ep, scenario_seed, steps, teacher_track_id])

            obs, reward, terminated, truncated, info = env.step(action_slot)
            done = terminated or truncated
            steps += 1

        num_lost = int(info.get("num_lost", 999))
        final_metric = float(info.get("objective_metric", float("inf")))

        keep_episode = (
            not skipped_episode
            and num_lost <= args.max_lost_for_keep
            and final_metric <= args.max_final_metric_for_keep
            and len(episode_labels) >= args.min_samples_per_episode
        )

        if keep_episode:
            global_features.extend(episode_global_features)
            track_features.extend(episode_track_features)
            action_masks.extend(episode_action_masks)
            labels.extend(episode_labels)
            metadata.extend(episode_metadata)
            kept_episodes += 1
            status = "KEPT"
        else:
            discarded_episodes += 1
            status = "DISCARDED"

            if skipped_episode and not skip_reason:
                skip_reason = "skipped_episode"
            elif num_lost > args.max_lost_for_keep:
                skip_reason = f"num_lost {num_lost} > {args.max_lost_for_keep}"
            elif final_metric > args.max_final_metric_for_keep:
                skip_reason = (
                    f"final_metric {final_metric:.2f} > "
                    f"{args.max_final_metric_for_keep:.2f}"
                )
            elif len(episode_labels) < args.min_samples_per_episode:
                skip_reason = (
                    f"samples {len(episode_labels)} < "
                    f"{args.min_samples_per_episode}"
                )

        print(
            f"episode={ep:04d} targets={len(config.targets)} "
            f"episode_samples={len(episode_labels)} "
            f"total_kept_samples={len(labels)} "
            f"final_metric={final_metric:.2f} "
            f"active={info.get('num_active')} lost={num_lost} "
            f"{status}"
            + (f" ({skip_reason})" if status == "DISCARDED" else "")
        )

    print(
        "\nDataset collection summary: "
        f"kept_episodes={kept_episodes} "
        f"discarded_episodes={discarded_episodes} "
        f"kept_samples={len(labels)}"
    )

    if len(labels) == 0:
        raise ValueError(
            "No training samples were kept. Try relaxing the filters, e.g. "
            "--max-lost-for-keep 1, lowering scenario difficulty, or increasing "
            "teacher iterations."
        )

    dataset = {
        "global_features": np.asarray(global_features, dtype=np.float32),
        "track_features": np.asarray(track_features, dtype=np.float32),
        "action_masks": np.asarray(action_masks, dtype=bool),
        "labels": np.asarray(labels, dtype=np.int64),
        "metadata": np.asarray(metadata, dtype=np.float32),
        "max_tracks": np.asarray([args.max_targets], dtype=np.int64),
        "global_dim": np.asarray([extractor.global_dim], dtype=np.int64),
        "track_dim": np.asarray([extractor.track_dim], dtype=np.int64),
        "extractor": np.asarray([args.extractor]),
        "kept_episodes": np.asarray([kept_episodes], dtype=np.int64),
        "discarded_episodes": np.asarray([discarded_episodes], dtype=np.int64),
        "max_lost_for_keep": np.asarray([args.max_lost_for_keep], dtype=np.int64),
        "max_final_metric_for_keep": np.asarray(
            [args.max_final_metric_for_keep],
            dtype=np.float32,
        ),
    }

    return dataset


def train(dataset, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = torch.tensor(dataset["global_features"], dtype=torch.float32)
    T = torch.tensor(dataset["track_features"], dtype=torch.float32)
    M = torch.tensor(dataset["action_masks"], dtype=torch.bool)
    y = torch.tensor(dataset["labels"], dtype=torch.long)

    ds = TensorDataset(G, T, M, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = TrackScorerNet(
        global_dim=int(dataset["global_dim"][0]),
        track_dim=int(dataset["track_dim"][0]),
        hidden_dim=args.hidden_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for gb, tb, mb, yb in loader:
            gb = gb.to(device)
            tb = tb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)

            scores = model(gb, tb, mb)
            loss = loss_fn(scores, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(yb)
            correct += int((scores.argmax(dim=1) == yb).sum().item())
            total += len(yb)

        print(
            f"epoch={epoch:03d} "
            f"loss={total_loss / max(1, total):.4f} "
            f"acc={correct / max(1, total):.3f}"
        )

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "max_tracks": int(dataset["max_tracks"][0]),
            "global_dim": int(dataset["global_dim"][0]),
            "track_dim": int(dataset["track_dim"][0]),
            "extractor": str(dataset["extractor"][0]),
            "kept_episodes": int(dataset["kept_episodes"][0]),
            "discarded_episodes": int(dataset["discarded_episodes"][0]),
            "max_lost_for_keep": int(dataset["max_lost_for_keep"][0]),
            "max_final_metric_for_keep": float(dataset["max_final_metric_for_keep"][0]),
        },
        model_output,
    )
    print(f"Saved model to {model_output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/randomized_track_scorer_demos.npz",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/track_scorer.pt",
    )

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--teacher-iterations", type=int, default=750)
    parser.add_argument("--extractor", type=str, default="polar")

    parser.add_argument("--min-targets", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=6)
    parser.add_argument("--min-radius", type=float, default=250.0)
    parser.add_argument("--max-radius", type=float, default=900.0)
    parser.add_argument("--min-target-speed", type=float, default=0.25)
    parser.add_argument("--max-target-speed", type=float, default=3.0)
    parser.add_argument("--min-velocity-noise-std", type=float, default=0.0)
    parser.add_argument("--max-velocity-noise-std", type=float, default=0.006)

    # Episode-quality filters. Bad MCTS teacher trajectories are discarded
    # instead of being imitated by the neural scorer.
    parser.add_argument("--max-lost-for-keep", type=int, default=0)
    parser.add_argument(
        "--max-final-metric-for-keep",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("--min-samples-per-episode", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)

    args = parser.parse_args()

    dataset = collect_dataset(args)

    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **dataset)
    print(f"Saved dataset to {dataset_path}")

    train(dataset, args)


if __name__ == "__main__":
    main()
