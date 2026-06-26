#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data import Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.dataset.memory_replay import read_memory_replay_jsonl  # noqa: E402
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ACTION_KEY  # noqa: E402
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ROBOT_KEY  # noqa: E402
from himem_bridge_vla.dataset.rmbench_progress_warmup import rmbench_action_normalizer_from_stats  # noqa: E402
from himem_bridge_vla.model.planner import ActionSegmentAutoencoder  # noqa: E402
from himem_bridge_vla.model.planner import ActionSegmentAutoencoderConfig  # noqa: E402
from himem_bridge_vla.model.planner import action_segment_autoencoder_loss  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RMBench 14-dim action-intent autoencoder.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <AUTODL_TMP>/benchmarks/RMBench.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--robot-key", default=DEFAULT_RMBENCH_ROBOT_KEY)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--samples-per-epoch", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--max-val-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--ckpt-interval", type=int, default=0)
    parser.add_argument("--distance-loss-weight", type=float, default=0.1)
    parser.add_argument("--endpoint-distance-weight", type=float, default=0.5)
    parser.add_argument("--gripper-distance-weight", type=float, default=0.25)
    parser.add_argument("--gripper-loss-weight", type=float, default=2.0)
    return parser.parse_args(argv)


class RMBenchActionSegmentDataset(Dataset):
    def __init__(
        self,
        *,
        rmbench_root: str | Path,
        index_path: str | Path,
        chunk_size: int,
        action_normalizer,
    ) -> None:
        self.rmbench_root = Path(rmbench_root).expanduser()
        self.chunk_size = int(chunk_size)
        self.action_normalizer = action_normalizer
        rows = read_memory_replay_jsonl(index_path)
        self.rows = [
            row
            for row in rows
            if str(row.get("benchmark", "")).upper() == "RMBENCH"
            and int(row.get("action_valid_count", 0)) >= self.chunk_size
        ]
        if not self.rows:
            raise ValueError(f"no valid RMBench action chunks found in {index_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        actions = read_rmbench_action_slice(
            self.rmbench_root,
            row,
            start=int(row["action_start"]),
            end=int(row["action_start"]) + self.chunk_size,
        )
        tensor = self.action_normalizer(torch.as_tensor(actions, dtype=torch.float32)).float()
        if tensor.shape != (self.chunk_size, 14):
            raise ValueError(f"RMBench action chunk shape {tuple(tensor.shape)} != ({self.chunk_size}, 14)")
        return {
            "actions": tensor,
            "episode_id": str(row["episode_id"]),
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    rmbench_root = resolve_rmbench_root(args.rmbench_root)
    normalizer = rmbench_action_normalizer_from_stats(args.norm_stats, robot_key=args.robot_key)
    dataset = RMBenchActionSegmentDataset(
        rmbench_root=rmbench_root,
        index_path=args.index,
        chunk_size=args.chunk_size,
        action_normalizer=normalizer,
    )
    config = ActionSegmentAutoencoderConfig(
        action_dim=14,
        chunk_size=int(args.chunk_size),
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_dim=int(args.ffn_dim),
        dropout=float(args.dropout),
        gripper_dim=1,
    )
    train_indices, val_indices = split_indices_by_episode(dataset, val_fraction=float(args.val_fraction), seed=int(args.seed))
    train_subset = Subset(dataset, train_indices)
    train_sampler = RandomSampler(
        train_subset,
        replacement=True,
        num_samples=max(int(args.samples_per_epoch), int(args.batch_size)),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_action_segments,
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_action_segments,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionSegmentAutoencoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gripper_indices = (6, 13)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "rmbench_root": str(rmbench_root),
                "dataset_size": len(dataset),
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "segment_autoencoder_config": asdict(config),
                "gripper_indices": list(gripper_indices),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    data_iter = iter(train_loader)
    for step in range(1, int(args.max_steps) + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        row = train_step(model, optimizer, batch, args, gripper_indices, device=device)
        row["step"] = step
        if val_loader is not None and int(args.eval_interval) > 0 and step % int(args.eval_interval) == 0:
            row.update({f"val_{key}": value for key, value in evaluate(model, val_loader, args, gripper_indices, device=device).items()})
        history.append(row)
        if val_loader is None or "val_loss" in row:
            selection_loss = float(row.get("val_loss", row["loss"]))
            if selection_loss < best_loss:
                best_loss = selection_loss
                save_checkpoint(output_dir / "best.pt", model, optimizer, config, row)
        if int(args.ckpt_interval) > 0 and step % int(args.ckpt_interval) == 0:
            save_checkpoint(output_dir / f"step_{step:06d}.pt", model, optimizer, config, row)
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            print(format_metrics(row), flush=True)

    save_checkpoint(output_dir / "last.pt", model, optimizer, config, history[-1])
    (output_dir / "train_history.json").write_text(
        json.dumps({"steps": history, "best_loss": best_loss, "final_loss": history[-1]["loss"]}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


def train_step(model, optimizer, batch, args, gripper_indices: tuple[int, ...], *, device: torch.device) -> dict[str, float | int]:
    model.train()
    actions = batch["actions"].to(device)
    segments = actions.unsqueeze(1)
    mask = torch.ones(actions.shape[0], 1, device=device, dtype=actions.dtype)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = action_segment_autoencoder_loss(
        model,
        segments,
        mask,
        gripper_indices=gripper_indices,
        gripper_loss_weight=float(args.gripper_loss_weight),
        distance_loss_weight=float(args.distance_loss_weight),
        endpoint_distance_weight=float(args.endpoint_distance_weight),
        gripper_distance_weight=float(args.gripper_distance_weight),
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
        **{key: float(value.detach().cpu().item()) for key, value in metrics.items()},
    }


@torch.no_grad()
def evaluate(model, loader, args, gripper_indices: tuple[int, ...], *, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total = 0
    for batch_index, batch in enumerate(loader):
        if int(args.max_val_batches) > 0 and batch_index >= int(args.max_val_batches):
            break
        actions = batch["actions"].to(device)
        segments = actions.unsqueeze(1)
        mask = torch.ones(actions.shape[0], 1, device=device, dtype=actions.dtype)
        loss, metrics = action_segment_autoencoder_loss(
            model,
            segments,
            mask,
            gripper_indices=gripper_indices,
            gripper_loss_weight=float(args.gripper_loss_weight),
            distance_loss_weight=float(args.distance_loss_weight),
            endpoint_distance_weight=float(args.endpoint_distance_weight),
            gripper_distance_weight=float(args.gripper_distance_weight),
        )
        weight = int(actions.shape[0])
        total += weight
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu().item()) * weight
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item()) * weight
    if total <= 0:
        raise ValueError("validation loader produced no batches")
    return {key: value / total for key, value in totals.items()}


def read_rmbench_action_slice(rmbench_root: Path, row: dict[str, Any], *, start: int, end: int):
    import h5py
    import numpy as np

    with h5py.File(rmbench_root / str(row["source_path"]), "r") as handle:
        return np.asarray(handle[DEFAULT_RMBENCH_ACTION_KEY][int(start) : int(end)], dtype="float32")


def collate_action_segments(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "actions": torch.stack([item["actions"] for item in batch], dim=0),
        "episode_id": [str(item["episode_id"]) for item in batch],
    }


def split_indices_by_episode(dataset: RMBenchActionSegmentDataset, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    all_indices = list(range(len(dataset)))
    if val_fraction == 0.0:
        return all_indices, []
    by_episode: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        by_episode.setdefault(str(row["episode_id"]), []).append(index)
    episodes = sorted(by_episode)
    rng = random.Random(int(seed))
    rng.shuffle(episodes)
    val_count = max(1, min(len(episodes) - 1, int(round(len(episodes) * float(val_fraction)))))
    val_episodes = set(episodes[:val_count])
    val_indices = sorted(index for episode in val_episodes for index in by_episode[episode])
    val_set = set(val_indices)
    train_indices = [index for index in all_indices if index not in val_set]
    return train_indices, val_indices


def save_checkpoint(path: Path, model, optimizer, config: ActionSegmentAutoencoderConfig, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "action_segment_autoencoder",
            "version": 1,
            "segment_autoencoder_config": asdict(config),
            "segment_autoencoder_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
        },
        path,
    )


def resolve_rmbench_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    autodl_tmp = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return autodl_tmp / "benchmarks" / "RMBench"


def format_metrics(row: dict[str, Any]) -> str:
    keys = ["step", "loss", "segment_ae_rec_loss", "segment_ae_dist_loss", "val_loss", "val_segment_ae_rec_loss", "grad_norm"]
    parts = []
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        parts.append(f"{key}={value}" if isinstance(value, int) else f"{key}={float(value):.6f}")
    return "rmbench_action_segment_ae " + " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
