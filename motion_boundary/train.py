from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_boundary.config import load_config, write_resolved_config
from motion_boundary.data import build_datasets, make_training_sampler
from motion_boundary.evaluate import evaluate_model
from motion_boundary.losses import boundary_loss, resolve_pos_weight
from motion_boundary.model import MotionStateBoundaryHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a motion-state boundary detector.")
    parser.add_argument("--config", default="motion_boundary/configs/default.yaml")
    parser.add_argument("--run-dir", default=None, help="Override outputs.run_dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.run_dir:
        config["outputs"]["run_dir"] = args.run_dir
    run_dir = Path(config["outputs"]["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir / "resolved_config.yaml")
    set_seed(int(config.get("seed", 42)))

    train_dataset, val_dataset = build_datasets(config)
    sampler, shuffle = make_training_sampler(train_dataset, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    model = MotionStateBoundaryHead(input_dim=train_dataset.input_dim, **config["model"]).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    labels, valid_mask = train_dataset.labels_and_mask()
    pos_weight = resolve_pos_weight(labels, valid_mask, config["training"].get("pos_weight", "sqrt_neg_pos"))
    best_scores = {
        "best_auprc": -1.0,
        "best_event_f1": -1.0,
        "best_replan": -1.0,
        "best_memory_write": -1.0,
    }
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        start = time.time()
        for batch in train_loader:
            features = batch["features"].to(args.device)
            labels = batch["label"].to(args.device).unsqueeze(1)
            valid = batch["valid"].to(args.device).unsqueeze(1)
            logits = model(features)
            loss = boundary_loss(logits, labels, valid, config["training"], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        val_metrics = evaluate_model(model, val_loader, config, device=args.device)
        threshold_summary = val_metrics.get("thresholds", {})
        summary = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "val_auprc": val_metrics["auprc"],
            "val_best_event_f1": _nested_metric(threshold_summary, "best_f1_metrics", "f1"),
            "val_replan_threshold": threshold_summary.get("replan_threshold"),
            "val_memory_write_threshold": threshold_summary.get("memory_write_threshold"),
            "seconds": round(time.time() - start, 2),
            "pos_weight": pos_weight,
            "sampler": config["training"].get("sampler", "balanced"),
            "loss": config["training"].get("loss", "bce"),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
        history.append(summary)
        print(json.dumps(summary, sort_keys=True))
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "input_dim": train_dataset.input_dim,
            "val_metrics": val_metrics,
        }
        _maybe_save_checkpoint(
            checkpoint,
            run_dir,
            "best_auprc",
            float(val_metrics["auprc"]),
            best_scores,
            also_write_best=True,
        )
        _maybe_save_checkpoint(
            checkpoint,
            run_dir,
            "best_event_f1",
            _nested_metric(threshold_summary, "best_f1_metrics", "f1"),
            best_scores,
        )
        _maybe_save_checkpoint(
            checkpoint,
            run_dir,
            "best_replan",
            _nested_metric(threshold_summary, "replan_metrics", "f1"),
            best_scores,
        )
        _maybe_save_checkpoint(
            checkpoint,
            run_dir,
            "best_memory_write",
            _nested_metric(threshold_summary, "memory_write_metrics", "recall"),
            best_scores,
        )

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return 0


def _nested_metric(container: dict, section: str, key: str) -> float:
    value = container.get(section)
    if not isinstance(value, dict) or value.get(key) is None:
        return -1.0
    return float(value[key])


def _maybe_save_checkpoint(
    checkpoint: dict,
    run_dir: Path,
    name: str,
    score: float,
    best_scores: dict[str, float],
    *,
    also_write_best: bool = False,
) -> None:
    if score <= best_scores[name]:
        return
    best_scores[name] = score
    path = run_dir / f"{name}.pt"
    torch.save(checkpoint, path)
    if also_write_best:
        torch.save(checkpoint, run_dir / "best.pt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
