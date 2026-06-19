from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coarse_planner.config import load_config
from coarse_planner.data import PlannerFeatureDataset
from coarse_planner.train import load_segment_autoencoder
from himem_bridge_vla.model.planner import action_segment_autoencoder_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ActionSegmentAutoencoder on planner feature caches.")
    parser.add_argument("--config", default="coarse_planner/configs/libero_h64_segment_ae_v2.yaml")
    parser.add_argument("--data-root", default=None, help="Override data.root.")
    parser.add_argument("--checkpoint", default=None, help="Override segment_autoencoder.checkpoint.")
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-cache-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.checkpoint is not None:
        config.setdefault("segment_autoencoder", {})["checkpoint"] = args.checkpoint
    if args.shard_cache_size is not None:
        config["data"]["shard_cache_size"] = int(args.shard_cache_size)
    config.setdefault("training", {})["batch_size"] = int(args.batch_size)

    device = torch.device(args.device)
    model = load_segment_autoencoder(config, device=device)
    results: dict[str, Any] = {
        "config": str(Path(args.config)),
        "dataset_root": str(config["data"]["root"]),
        "checkpoint": str(config["segment_autoencoder"]["checkpoint"]),
        "batch_size": int(args.batch_size),
        "splits": {},
    }
    for split in args.splits:
        row = evaluate_split(config, model, split=split, device=device, batch_size=int(args.batch_size))
        results["splits"][split] = row
        print(json.dumps({split: row}, sort_keys=True), flush=True)

    if len(args.splits) > 1:
        results["splits"]["all"] = summarize_all([results["splits"][split] for split in args.splits])

    output_json = Path(
        args.output_json
        or Path(str(config.get("outputs", {}).get("segment_ae_run_dir", "coarse_planner/outputs/segment_ae")))
        / "eval_segment_autoencoder.json"
    ).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True))

    output_md = Path(args.output_md or output_json.with_suffix(".md")).expanduser()
    output_md.write_text(render_markdown(results))
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True))
    return 0


@torch.no_grad()
def evaluate_split(
    config: dict[str, Any],
    model: torch.nn.Module,
    *,
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    dataset = PlannerFeatureDataset(
        config["data"]["root"],
        split=split,
        manifest=config["data"].get("manifest", "manifest.json"),
        shard_cache_size=int(config["data"].get("shard_cache_size", 8)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    totals = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "dist_loss": 0.0,
        "motion_huber_sum": 0.0,
        "gripper_bce_sum": 0.0,
        "gripper_correct": 0.0,
        "gripper_total": 0.0,
        "active_segments": 0.0,
        "batches": 0,
        "samples": len(dataset),
    }
    amp_enabled = device.type == "cuda"
    loss_config = config.get("loss", {})
    ae_config = config.get("segment_autoencoder", {})
    for batch in loader:
        segments = batch["action_segments"].to(device)
        mask = batch["action_segment_mask"].to(device)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            loss, metrics = action_segment_autoencoder_loss(
                model,
                segments,
                mask,
                gripper_indices=loss_config.get("gripper_indices", [-1]),
                gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
                distance_loss_weight=float(ae_config.get("distance_loss_weight", 0.0)),
                dct_low_frequency=int(ae_config.get("dct_low_frequency", 4)),
                endpoint_distance_weight=float(ae_config.get("endpoint_distance_weight", 1.0)),
                gripper_distance_weight=float(ae_config.get("gripper_distance_weight", 1.0)),
            )
            output = model(segments)
            detailed = detailed_reconstruction_metrics(output.reconstruction, segments, mask)

        active = float(mask.sum().detach().cpu())
        totals["loss"] += float(loss.detach().cpu())
        totals["rec_loss"] += float(metrics["segment_ae_rec_loss"].detach().cpu())
        totals["dist_loss"] += float(metrics["segment_ae_dist_loss"].detach().cpu())
        totals["motion_huber_sum"] += float(detailed["motion_huber_sum"])
        totals["gripper_bce_sum"] += float(detailed["gripper_bce_sum"])
        totals["gripper_correct"] += float(detailed["gripper_correct"])
        totals["gripper_total"] += active
        totals["active_segments"] += active
        totals["batches"] += 1

    batches = max(int(totals["batches"]), 1)
    active_segments = max(float(totals["active_segments"]), 1.0)
    return {
        "samples": int(totals["samples"]),
        "active_segments": int(totals["active_segments"]),
        "batches": batches,
        "loss": float(totals["loss"]) / batches,
        "rec_loss": float(totals["rec_loss"]) / batches,
        "dist_loss": float(totals["dist_loss"]) / batches,
        "motion_huber": float(totals["motion_huber_sum"]) / active_segments,
        "gripper_bce": float(totals["gripper_bce_sum"]) / active_segments,
        "gripper_accuracy": float(totals["gripper_correct"]) / max(float(totals["gripper_total"]), 1.0),
    }


def detailed_reconstruction_metrics(
    reconstruction: torch.Tensor,
    target_segments: torch.Tensor,
    segment_mask: torch.Tensor,
) -> dict[str, float]:
    target = target_segments.to(device=reconstruction.device, dtype=reconstruction.dtype)
    mask = segment_mask.to(device=reconstruction.device, dtype=reconstruction.dtype)
    motion_loss = F.smooth_l1_loss(reconstruction[..., :6], target[..., :6], reduction="none").mean(dim=(-1, -2))
    grip_target = target[..., 6:7].clamp(0.0, 1.0)
    grip_bce = F.binary_cross_entropy_with_logits(reconstruction[..., 6:7], grip_target, reduction="none").mean(
        dim=(-1, -2)
    )
    grip_pred = (torch.sigmoid(reconstruction[..., 6:7]) >= 0.5).to(dtype=grip_target.dtype)
    grip_true = (grip_target >= 0.5).to(dtype=grip_target.dtype)
    grip_correct = (grip_pred == grip_true).to(dtype=mask.dtype).squeeze(-1).mean(dim=-1)
    return {
        "motion_huber_sum": float((motion_loss * mask).sum().detach().cpu()),
        "gripper_bce_sum": float((grip_bce * mask).sum().detach().cpu()),
        "gripper_correct": float((grip_correct * mask).sum().detach().cpu()),
    }


def summarize_all(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    samples = sum(int(row["samples"]) for row in rows)
    active = sum(int(row["active_segments"]) for row in rows)
    batches = sum(int(row["batches"]) for row in rows)
    return {
        "samples": samples,
        "active_segments": active,
        "batches": batches,
        "loss": weighted_average(rows, "loss", "batches"),
        "rec_loss": weighted_average(rows, "rec_loss", "batches"),
        "dist_loss": weighted_average(rows, "dist_loss", "batches"),
        "motion_huber": weighted_average(rows, "motion_huber", "active_segments"),
        "gripper_bce": weighted_average(rows, "gripper_bce", "active_segments"),
        "gripper_accuracy": weighted_average(rows, "gripper_accuracy", "active_segments"),
    }


def weighted_average(rows: list[dict[str, float | int]], value_key: str, weight_key: str) -> float:
    weight_sum = sum(float(row[weight_key]) for row in rows)
    if weight_sum <= 0:
        return 0.0
    return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / weight_sum


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Action Segment AE Evaluation",
        "",
        f"Dataset: `{results['dataset_root']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        "",
        "| split | samples | active segments | total loss | rec loss | dist loss | motion Huber | gripper BCE | gripper acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in results["splits"].items():
        lines.append(
            f"| {split} | {row['samples']} | {row['active_segments']} | {row['loss']:.6f} | "
            f"{row['rec_loss']:.6f} | {row['dist_loss']:.6f} | {row['motion_huber']:.6f} | "
            f"{row['gripper_bce']:.6f} | {row['gripper_accuracy']:.6f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
