from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from coarse_planner.config import load_config, write_resolved_config
from coarse_planner.data import build_datasets
from himem_bridge_vla.model.planner import (
    ActionSegmentAutoencoder,
    ActionSegmentAutoencoderConfig,
    action_segment_autoencoder_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train action-only segment autoencoder for CoarsePlanner intent latents.")
    parser.add_argument("--config", default="coarse_planner/configs/default.yaml")
    parser.add_argument("--run-dir", default=None, help="Override outputs.segment_ae_run_dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override training.lr.")
    parser.add_argument("--resume-from", default=None, help="Resume model weights from an AE checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.run_dir:
        config.setdefault("outputs", {})["segment_ae_run_dir"] = args.run_dir
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        config["training"]["lr"] = float(args.learning_rate)
    run_dir = Path(config["outputs"].get("segment_ae_run_dir") or f"{config['outputs']['run_dir']}_segment_ae").expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir / "resolved_config.yaml")
    set_seed(int(config.get("seed", 42)))

    train_dataset, val_dataset = build_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=str(args.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    ae_config = resolve_autoencoder_config(config, train_dataset.sample_shapes)
    model = ActionSegmentAutoencoder(ae_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    best_loss = float("inf")
    history = []
    start_epoch = 1
    history_path = run_dir / "train_history.json"
    if args.resume_from:
        checkpoint = torch.load(Path(args.resume_from).expanduser(), map_location=args.device)
        model.load_state_dict(checkpoint["segment_autoencoder_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if history_path.exists():
            history = json.loads(history_path.read_text())
            start_epoch = len(history) + 1
            best_loss = min((float(row["val_loss"]) for row in history), default=float("inf"))
        else:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_loss = float(checkpoint.get("best_loss", checkpoint.get("val_metrics", {}).get("loss", float("inf"))))

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        epoch_loss = 0.0
        steps = 0
        start = time.time()
        for batch in train_loader:
            segments = batch["action_segments"].to(args.device)
            mask = batch["action_segment_mask"].to(args.device)
            loss, _ = action_segment_autoencoder_loss(
                model,
                segments,
                mask,
                gripper_indices=config["loss"].get("gripper_indices", [-1]),
                gripper_loss_weight=float(config["loss"].get("gripper_loss_weight", 1.0)),
                distance_loss_weight=float(config["segment_autoencoder"].get("distance_loss_weight", 0.0)),
                dct_low_frequency=int(config["segment_autoencoder"].get("dct_low_frequency", 4)),
                endpoint_distance_weight=float(config["segment_autoencoder"].get("endpoint_distance_weight", 1.0)),
                gripper_distance_weight=float(config["segment_autoencoder"].get("gripper_distance_weight", 1.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        val_metrics = evaluate_autoencoder(model, val_loader, config, device=args.device)
        summary = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "val_loss": val_metrics["loss"],
            "val_rec_loss": val_metrics["rec_loss"],
            "val_dist_loss": val_metrics["dist_loss"],
            "seconds": round(time.time() - start, 2),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            **cuda_memory_summary(args.device),
        }
        history.append(summary)
        print(json.dumps(summary, sort_keys=True))
        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "segment_autoencoder_state_dict": model.state_dict(),
            "segment_autoencoder_config": ae_config.__dict__,
            "config": config,
            "val_metrics": val_metrics,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, run_dir / "best.pt")

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return 0


def resolve_autoencoder_config(config: dict[str, Any], sample_shapes: dict[str, tuple[int, ...]]) -> ActionSegmentAutoencoderConfig:
    segment_shape = sample_shapes["action_segments"]
    ae_config = config["segment_autoencoder"]
    return ActionSegmentAutoencoderConfig(
        action_dim=int(segment_shape[-1]),
        chunk_size=int(segment_shape[-2]),
        latent_dim=int(ae_config.get("latent_dim", 64)),
        hidden_dim=int(ae_config.get("hidden_dim", 256)),
        num_layers=int(ae_config.get("num_layers", 2)),
        num_heads=int(ae_config.get("num_heads", 4)),
        ffn_dim=ae_config.get("ffn_dim"),
        dropout=float(ae_config.get("dropout", 0.0)),
        gripper_dim=int(ae_config.get("gripper_dim", 1)),
    )


@torch.no_grad()
def evaluate_autoencoder(
    model: ActionSegmentAutoencoder,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_rec = 0.0
    total_dist = 0.0
    batches = 0
    for batch in loader:
        loss, metrics = action_segment_autoencoder_loss(
            model,
            batch["action_segments"].to(device),
            batch["action_segment_mask"].to(device),
            gripper_indices=config["loss"].get("gripper_indices", [-1]),
            gripper_loss_weight=float(config["loss"].get("gripper_loss_weight", 1.0)),
            distance_loss_weight=float(config["segment_autoencoder"].get("distance_loss_weight", 0.0)),
            dct_low_frequency=int(config["segment_autoencoder"].get("dct_low_frequency", 4)),
            endpoint_distance_weight=float(config["segment_autoencoder"].get("endpoint_distance_weight", 1.0)),
            gripper_distance_weight=float(config["segment_autoencoder"].get("gripper_distance_weight", 1.0)),
        )
        total_loss += float(loss.detach().cpu().item())
        total_rec += float(metrics["segment_ae_rec_loss"].detach().cpu().item())
        total_dist += float(metrics["segment_ae_dist_loss"].detach().cpu().item())
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "rec_loss": total_rec / max(batches, 1),
        "dist_loss": total_dist / max(batches, 1),
        "batches": float(batches),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_memory_summary(device: str | torch.device) -> dict[str, float]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 3),
        "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / (1024**3), 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
