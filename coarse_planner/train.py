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
from coarse_planner.evaluate import evaluate_planner
from himem_bridge_vla.model.planner import CoarsePlanner, CoarsePlannerConfig
from himem_bridge_vla.training_loss import coarse_planner_smooth_l1_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone CoarsePlanner on precomputed VLM feature cache.")
    parser.add_argument("--config", default="coarse_planner/configs/default.yaml")
    parser.add_argument("--run-dir", default=None, help="Override outputs.run_dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.run_dir:
        config["outputs"]["run_dir"] = args.run_dir
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    run_dir = Path(config["outputs"]["run_dir"]).expanduser()
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

    model_config = resolve_model_config(config, train_dataset.sample_shapes)
    model = CoarsePlanner(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    loss_config = config.get("loss", {})
    best_loss = float("inf")
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        epoch_loss = 0.0
        steps = 0
        start = time.time()
        for batch in train_loader:
            output = model(batch["vlm_tokens"].to(args.device), batch["state"].to(args.device))
            loss = coarse_planner_smooth_l1_loss(
                output.coarse_actions,
                batch["coarse_actions"].to(args.device),
                batch["coarse_action_mask"].to(args.device),
                gripper_indices=loss_config.get("gripper_indices", [-1]),
                gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
                smoothness_weight=float(loss_config.get("smoothness_weight", 0.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        val_metrics = evaluate_planner(model, val_loader, config, device=args.device)
        summary = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "seconds": round(time.time() - start, 2),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            **cuda_memory_summary(args.device),
        }
        history.append(summary)
        print(json.dumps(summary, sort_keys=True))
        checkpoint = {
            "model": model.state_dict(),
            "planner_config": model_config.__dict__,
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            torch.save(checkpoint, run_dir / "best.pt")

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return 0


def resolve_model_config(config: dict[str, Any], sample_shapes: dict[str, tuple[int, ...]]) -> CoarsePlannerConfig:
    model_config = config["model"]
    target_config = config["target"]
    vlm_shape = sample_shapes["vlm_tokens"]
    state_shape = sample_shapes["state"]
    action_shape = sample_shapes["coarse_actions"]
    hidden_dim = _auto_int(model_config.get("hidden_dim"), vlm_shape[-1])
    state_dim = _auto_int(model_config.get("state_dim"), state_shape[-1])
    action_dim = _auto_int(model_config.get("action_dim"), action_shape[-1])
    num_plan_steps = int(model_config.get("num_plan_steps") or target_config["num_plan_steps"])
    planning_horizon = int(model_config.get("planning_horizon") or target_config["planning_horizon"])
    return CoarsePlannerConfig(
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        num_plan_steps=num_plan_steps,
        planning_horizon=planning_horizon,
        num_layers=int(model_config.get("num_layers", 3)),
        num_heads=int(model_config.get("num_heads", 8)),
        dropout=float(model_config.get("dropout", 0.0)),
        ffn_mult=int(model_config.get("ffn_mult", 4)),
    )


def _auto_int(value: Any, fallback: int) -> int:
    if value is None or str(value).lower() == "auto":
        return int(fallback)
    return int(value)


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
