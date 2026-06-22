from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coarse_planner.config import load_config, write_resolved_config
from coarse_planner.data import ShardBatchSampler, build_datasets
from coarse_planner.evaluate import evaluate_planner
from coarse_planner.latent_normalization import latent_normalizer_from_checkpoint, resolve_latent_normalizer
from himem_bridge_vla.model.planner import ActionSegmentAutoencoder, ActionSegmentAutoencoderConfig, CoarsePlanner, CoarsePlannerConfig
from himem_bridge_vla.training_loss import coarse_planner_intent_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone CoarsePlanner on precomputed VLM feature cache.")
    parser.add_argument("--config", default="coarse_planner/configs/default.yaml")
    parser.add_argument("--run-dir", default=None, help="Override outputs.run_dir.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override training.lr.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--resume-from", default=None, help="Resume model weights from a planner checkpoint.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Do not restore optimizer/scaler state when resuming.")
    parser.add_argument("--reset-best", action="store_true", help="Start best metric tracking from inf when resuming.")
    parser.add_argument("--reset-epoch", action="store_true", help="Start epoch numbering from 1 when loading weights.")
    parser.add_argument(
        "--convert-raw-latent-head",
        action="store_true",
        help="When latent normalization is enabled, convert a raw-z checkpoint head to normalized-z output.",
    )
    parser.add_argument("--segment-autoencoder-checkpoint", default=None, help="Override segment_autoencoder.checkpoint.")
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
    if args.learning_rate is not None:
        config["training"]["lr"] = float(args.learning_rate)
    if args.amp:
        config["training"]["amp"] = True
    if args.no_amp:
        config["training"]["amp"] = False
    if args.segment_autoencoder_checkpoint is not None:
        config.setdefault("segment_autoencoder", {})["checkpoint"] = args.segment_autoencoder_checkpoint
    init_config = config.get("initialization", {})
    resume_from = args.resume_from or init_config.get("resume_from")
    reset_optimizer = bool(args.reset_optimizer or init_config.get("reset_optimizer", False))
    reset_best = bool(args.reset_best or init_config.get("reset_best", False))
    reset_epoch = bool(args.reset_epoch or init_config.get("reset_epoch", False))
    convert_raw_latent_head = bool(args.convert_raw_latent_head or init_config.get("convert_raw_latent_head", False))
    run_dir = Path(config["outputs"]["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir / "resolved_config.yaml")
    set_seed(int(config.get("seed", 42)))

    train_dataset, val_dataset = build_datasets(config)
    train_shuffle = bool(config["training"].get("shuffle", True))
    batch_size = int(config["training"]["batch_size"])
    shuffle_mode = str(config["training"].get("shuffle_mode", "sample")).lower()
    if shuffle_mode == "shard":
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ShardBatchSampler(
                train_dataset,
                batch_size=batch_size,
                shuffle=train_shuffle,
                drop_last=bool(config["training"].get("drop_last", False)),
                seed=int(config.get("seed", 42)),
            ),
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=str(args.device).startswith("cuda"),
        )
    elif shuffle_mode == "sample":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=str(args.device).startswith("cuda"),
        )
    else:
        raise ValueError(f"training.shuffle_mode must be 'sample' or 'shard', got {shuffle_mode!r}")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    segment_autoencoder = load_segment_autoencoder(config, device=args.device)
    amp_enabled = _amp_enabled(config, args.device)
    latent_normalizer = resolve_latent_normalizer(
        config,
        run_dir=run_dir,
        segment_autoencoder=segment_autoencoder,
        train_loader=train_loader,
        device=args.device,
        amp_enabled=amp_enabled,
    )
    model_config = resolve_model_config(config, train_dataset.sample_shapes, latent_dim=segment_autoencoder.config.latent_dim)
    model = CoarsePlanner(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_config = config.get("loss", {})
    best_loss = _initial_best_value(config)
    history = []
    start_epoch = 1
    history_path = run_dir / "train_history.json"
    patience = int(config["training"].get("early_stopping_patience", 0) or 0)
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    if resume_from:
        checkpoint = torch.load(Path(str(resume_from)).expanduser(), map_location=args.device, weights_only=False)
        state_dict = checkpoint.get("model")
        if state_dict is None:
            raise KeyError(f"planner checkpoint lacks model weights: {resume_from}")
        model.load_state_dict(state_dict)
        checkpoint_normalizer = latent_normalizer_from_checkpoint(checkpoint, device=args.device)
        if latent_normalizer is None and checkpoint_normalizer is not None:
            latent_normalizer = checkpoint_normalizer
        if convert_raw_latent_head:
            if latent_normalizer is None:
                raise ValueError("--convert-raw-latent-head requires latent normalization stats")
            _convert_raw_latent_head_to_normalized_output(model, latent_normalizer)
        if not reset_optimizer:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(args.learning_rate)
        if history_path.exists() and not reset_epoch:
            history = json.loads(history_path.read_text())
            start_epoch = len(history) + 1
            best_loss = _best_history_value(history, config)
        else:
            start_epoch = 1 if reset_epoch else int(checkpoint.get("epoch", 0)) + 1
            best_loss = float(checkpoint.get("best_loss", checkpoint.get("val_metrics", {}).get("loss", float("inf"))))
        if reset_best:
            best_loss = _initial_best_value(config)

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        epoch_loss = 0.0
        steps = 0
        start = time.time()
        for batch in train_loader:
            with _autocast_context(amp_enabled):
                output = model(batch["vlm_tokens"].to(args.device), batch["state"].to(args.device))
                action_segments = batch["action_segments"].to(args.device)
                segment_mask = batch["action_segment_mask"].to(args.device)
                with torch.no_grad():
                    target_raw_latents = segment_autoencoder.encode(action_segments)
                if latent_normalizer is None:
                    target_loss_latents = target_raw_latents
                    predicted_loss_latents = output.predicted_latents
                    predicted_raw_latents = output.predicted_latents
                else:
                    target_loss_latents = latent_normalizer.normalize(target_raw_latents)
                    predicted_loss_latents = output.predicted_latents
                    predicted_raw_latents = latent_normalizer.unnormalize(output.predicted_latents)
                decoded_segments = segment_autoencoder.decode(predicted_raw_latents)
                loss = coarse_planner_intent_loss(
                    predicted_loss_latents,
                    target_loss_latents,
                    decoded_segments,
                    action_segments,
                    segment_mask,
                    latent_loss_weight=float(loss_config.get("latent_loss_weight", 1.0)),
                    chunk_loss_weight=float(loss_config.get("chunk_loss_weight", 1.0)),
                    gripper_indices=loss_config.get("gripper_indices", [-1]),
                    gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
                    token_loss_weights=loss_config.get("token_loss_weights"),
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu().item())
            steps += 1

        val_metrics = evaluate_planner(
            model,
            segment_autoencoder,
            val_loader,
            config,
            device=args.device,
            latent_normalizer=latent_normalizer,
        )
        summary = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "val_loss": val_metrics["loss"],
            "val_latent_mse": val_metrics["latent_mse"],
            "val_normalized_latent_mse": val_metrics["normalized_latent_mse"],
            "val_raw_latent_mse": val_metrics["raw_latent_mse"],
            "val_decoded_chunk_loss": val_metrics["decoded_chunk_loss"],
            "val_latent_cosine_similarity": val_metrics["latent_cosine_similarity"],
            "seconds": round(time.time() - start, 2),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            **cuda_memory_summary(args.device),
        }
        for key, value in sorted(val_metrics.items()):
            if key.startswith("latent_mse_u"):
                summary[f"val_{key}"] = value
        history.append(summary)
        print(json.dumps(summary, sort_keys=True))
        current_best_value = _checkpoint_selection_value(summary, val_metrics, config)
        improved = _is_better_checkpoint(current_best_value, best_loss, config, min_delta=min_delta)
        if improved:
            best_loss = float(current_best_value)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "best_metric": str(config.get("training", {}).get("save_best_metric", "val_loss")),
            "model": model.state_dict(),
            "planner_config": model_config.__dict__,
            "segment_autoencoder_config": segment_autoencoder.config.__dict__,
            "config": config,
            "val_metrics": val_metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "latent_normalizer": latent_normalizer.state_dict() if latent_normalizer is not None else None,
            "latent_normalization_enabled": latent_normalizer is not None,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run_dir / "best.pt")
        (run_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "best_value": best_loss,
                        "best_metric": str(config.get("training", {}).get("save_best_metric", "val_loss")),
                        "patience": patience,
                        "min_delta": min_delta,
                    },
                    sort_keys=True,
                )
            )
            break

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return 0


def resolve_model_config(config: dict[str, Any], sample_shapes: dict[str, tuple[int, ...]], *, latent_dim: int) -> CoarsePlannerConfig:
    model_config = config["model"]
    target_config = config["target"]
    vlm_shape = sample_shapes["vlm_tokens"]
    state_shape = sample_shapes["state"]
    hidden_dim = _auto_int(model_config.get("hidden_dim"), vlm_shape[-1])
    state_dim = _auto_int(model_config.get("state_dim"), state_shape[-1])
    num_plan_steps = int(model_config.get("num_plan_steps") or target_config["num_plan_steps"])
    planning_horizon = int(model_config.get("planning_horizon") or target_config["planning_horizon"])
    return CoarsePlannerConfig(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        latent_dim=int(model_config.get("latent_dim") or latent_dim),
        num_plan_steps=num_plan_steps,
        planning_horizon=planning_horizon,
        num_layers=int(model_config.get("num_layers", 3)),
        num_heads=int(model_config.get("num_heads", 8)),
        dropout=float(model_config.get("dropout", 0.0)),
        ffn_mult=int(model_config.get("ffn_mult", 4)),
        latent_head_hidden_dim=int(model_config.get("latent_head_hidden_dim", 512)),
    )


def load_segment_autoencoder(config: dict[str, Any], *, device: str | torch.device) -> ActionSegmentAutoencoder:
    ae_config = config.get("segment_autoencoder", {})
    checkpoint_path = ae_config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("segment_autoencoder.checkpoint is required for CoarsePlanner latent training")
    checkpoint = torch.load(Path(str(checkpoint_path)).expanduser(), map_location=device, weights_only=False)
    raw_config = checkpoint.get("segment_autoencoder_config") or checkpoint.get("autoencoder_config")
    if raw_config is None:
        raise KeyError(f"segment autoencoder checkpoint lacks config: {checkpoint_path}")
    state_dict = checkpoint.get("segment_autoencoder_state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError(f"segment autoencoder checkpoint lacks weights: {checkpoint_path}")
    model = ActionSegmentAutoencoder(ActionSegmentAutoencoderConfig(**raw_config)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


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


def _amp_enabled(config: dict[str, Any], device: str | torch.device) -> bool:
    return bool(config.get("training", {}).get("amp", str(device).startswith("cuda"))) and str(device).startswith("cuda")


def _autocast_context(enabled: bool) -> torch.amp.autocast_mode.autocast:
    return torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled)


def _checkpoint_selection_value(summary: dict[str, Any], val_metrics: dict[str, float], config: dict[str, Any]) -> float:
    metric = str(config.get("training", {}).get("save_best_metric", "val_loss"))
    if metric in summary:
        return float(summary[metric])
    if metric.startswith("val_"):
        raw_key = metric[len("val_") :]
        if raw_key in val_metrics:
            return float(val_metrics[raw_key])
    if metric in val_metrics:
        return float(val_metrics[metric])
    raise KeyError(f"save_best_metric={metric!r} was not found in validation metrics")


def _is_better_checkpoint(value: float, best_value: float, config: dict[str, Any], *, min_delta: float = 0.0) -> bool:
    mode = str(config.get("training", {}).get("save_best_mode", "min")).lower()
    if mode == "min":
        return value < best_value - float(min_delta)
    if mode == "max":
        return value > best_value + float(min_delta)
    raise ValueError(f"training.save_best_mode must be 'min' or 'max', got {mode!r}")


def _best_history_value(history: list[dict[str, Any]], config: dict[str, Any]) -> float:
    values = []
    for row in history:
        try:
            values.append(_checkpoint_selection_value(row, {}, config))
        except KeyError:
            continue
    if not values:
        return _initial_best_value(config)
    mode = str(config.get("training", {}).get("save_best_mode", "min")).lower()
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    raise ValueError(f"training.save_best_mode must be 'min' or 'max', got {mode!r}")


def _initial_best_value(config: dict[str, Any]) -> float:
    mode = str(config.get("training", {}).get("save_best_mode", "min")).lower()
    if mode == "min":
        return float("inf")
    if mode == "max":
        return float("-inf")
    raise ValueError(f"training.save_best_mode must be 'min' or 'max', got {mode!r}")


def _convert_raw_latent_head_to_normalized_output(model: CoarsePlanner, latent_normalizer: Any) -> None:
    last_linear = None
    for module in model.latent_head.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError("CoarsePlanner latent_head does not contain a Linear output layer")
    std = latent_normalizer.std.to(device=last_linear.weight.device, dtype=last_linear.weight.dtype)
    mean = latent_normalizer.mean.to(device=last_linear.weight.device, dtype=last_linear.weight.dtype)
    if last_linear.out_features != std.numel():
        raise ValueError(f"latent head out_features {last_linear.out_features} != normalizer dim {std.numel()}")
    with torch.no_grad():
        last_linear.weight.div_(std.unsqueeze(1))
        last_linear.bias.sub_(mean).div_(std)


if __name__ == "__main__":
    raise SystemExit(main())
