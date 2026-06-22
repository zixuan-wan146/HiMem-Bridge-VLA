from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coarse_planner.config import load_config
from coarse_planner.data import PlannerFeatureDataset, build_datasets
from coarse_planner.latent_normalization import (
    latent_normalization_enabled,
    latent_normalizer_from_checkpoint,
    latent_normalizer_stats_path,
    load_latent_normalizer,
)
from coarse_planner.train import load_segment_autoencoder, resolve_model_config
from himem_bridge_vla.model.planner import CoarsePlanner, CoarsePlannerConfig, action_segment_reconstruction_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CoarsePlanner latent prediction errors.")
    parser.add_argument("--config", default="coarse_planner/configs/libero_h64_planner_znorm_v17.yaml")
    parser.add_argument("--checkpoint", default=None, help="Planner checkpoint. Defaults to outputs.run_dir/best.pt.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--nearest-chunk-size", type=int, default=256)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--no-holdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or Path(config["outputs"]["run_dir"]) / "best.pt").expanduser()
    device = torch.device(args.device)

    train_dataset, eval_dataset = build_datasets(config)
    segment_autoencoder = load_segment_autoencoder(config, device=device)
    model, checkpoint = load_planner(config, checkpoint_path, train_dataset.sample_shapes, segment_autoencoder, device=device)
    latent_normalizer = load_normalizer_for_diagnosis(config, checkpoint, checkpoint_path, device=device)

    amp_enabled = bool(config.get("training", {}).get("amp", str(device).startswith("cuda"))) and str(device).startswith("cuda")
    train_reference = collect_dataset(
        train_dataset,
        model=None,
        segment_autoencoder=segment_autoencoder,
        latent_normalizer=latent_normalizer,
        config=config,
        device=device,
        batch_size=args.batch_size,
        amp_enabled=amp_enabled,
    )

    query_collections: dict[str, dict[str, Any]] = {
        "original_eval": collect_dataset(
            eval_dataset,
            model=model,
            segment_autoencoder=segment_autoencoder,
            latent_normalizer=latent_normalizer,
            config=config,
            device=device,
            batch_size=args.batch_size,
            amp_enabled=amp_enabled,
        )
    }
    if not args.no_holdout:
        holdout_root = config.get("evaluation", {}).get("holdout_root")
        if holdout_root:
            query_collections.update(
                collect_holdout(
                    config,
                    holdout_root,
                    model=model,
                    segment_autoencoder=segment_autoencoder,
                    latent_normalizer=latent_normalizer,
                    device=device,
                    batch_size=args.batch_size,
                    amp_enabled=amp_enabled,
                )
            )

    results: dict[str, Any] = {
        "config": str(Path(args.config)),
        "checkpoint": str(checkpoint_path),
        "latent_normalization_enabled": latent_normalizer is not None,
        "target_raw_latent_mse": 0.08,
        "train_reference": reference_summary(train_reference),
        "datasets": {},
    }
    for name, collection in query_collections.items():
        dataset_result = summarize_collection(collection, latent_normalizer=latent_normalizer)
        dataset_result["baselines"] = {
            "train_token_mean": prediction_baseline_metrics(
                train_token_mean_prediction(train_reference, collection),
                collection,
                latent_normalizer=latent_normalizer,
            ),
            "input_context_nearest_neighbor": prediction_baseline_metrics(
                input_nearest_neighbor_prediction(
                    train_reference,
                    collection,
                    chunk_size=int(args.nearest_chunk_size),
                ),
                collection,
                latent_normalizer=latent_normalizer,
            ),
            "oracle_same_token_target_nearest_neighbor": prediction_baseline_metrics(
                oracle_target_nearest_neighbor_prediction(
                    train_reference,
                    collection,
                    chunk_size=int(args.nearest_chunk_size),
                ),
                collection,
                latent_normalizer=latent_normalizer,
            ),
        }
        model_raw = float(dataset_result["overall"]["raw_latent_mse"])
        dataset_result["gap_to_target"] = {
            "target_raw_latent_mse": 0.08,
            "absolute_gap": model_raw - 0.08,
            "ratio_to_target": model_raw / 0.08 if 0.08 > 0 else None,
        }
        results["datasets"][name] = dataset_result

    output_json = Path(args.output_json or checkpoint_path.parent / "latent_error_diagnostics.json").expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True))

    output_md = Path(args.output_md or output_json.with_suffix(".md")).expanduser()
    output_md.write_text(render_markdown(results))
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2, sort_keys=True))
    return 0


def load_planner(
    config: dict[str, Any],
    checkpoint_path: Path,
    sample_shapes: dict[str, tuple[int, ...]],
    segment_autoencoder: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[CoarsePlanner, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_planner_config = checkpoint.get("planner_config")
    if raw_planner_config is None:
        planner_config = resolve_model_config(
            config,
            sample_shapes,
            latent_dim=int(segment_autoencoder.config.latent_dim),
        )
    else:
        planner_config = CoarsePlannerConfig(**raw_planner_config)
    model = CoarsePlanner(planner_config).to(device)
    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise KeyError(f"planner checkpoint lacks model weights: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def load_normalizer_for_diagnosis(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: torch.device,
):
    normalizer = latent_normalizer_from_checkpoint(checkpoint, device=device)
    if normalizer is not None:
        return normalizer
    if not latent_normalization_enabled(config):
        return None
    stats_path = latent_normalizer_stats_path(config, checkpoint_path.parent)
    if not stats_path.exists():
        raise FileNotFoundError(f"latent normalization is enabled but stats were not found: {stats_path}")
    return load_latent_normalizer(stats_path, device=device)


def collect_holdout(
    config: dict[str, Any],
    holdout_root: str,
    *,
    model: CoarsePlanner,
    segment_autoencoder: torch.nn.Module,
    latent_normalizer: Any,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> dict[str, dict[str, Any]]:
    data_config = config["data"]
    splits = [str(data_config.get("train_split", "train")), str(data_config.get("eval_split", "eval"))]
    collections: dict[str, dict[str, Any]] = {}
    for split in splits:
        try:
            dataset = PlannerFeatureDataset(
                holdout_root,
                split=split,
                manifest=data_config.get("manifest", "manifest.json"),
                shard_cache_size=int(data_config.get("shard_cache_size", 8)),
            )
        except ValueError:
            continue
        collections[f"holdout_{split}"] = collect_dataset(
            dataset,
            model=model,
            segment_autoencoder=segment_autoencoder,
            latent_normalizer=latent_normalizer,
            config=config,
            device=device,
            batch_size=batch_size,
            amp_enabled=amp_enabled,
        )
    if len(collections) > 1:
        collections["holdout_all"] = concatenate_collections(list(collections.values()))
    return collections


@torch.no_grad()
def collect_dataset(
    dataset: torch.utils.data.Dataset,
    *,
    model: CoarsePlanner | None,
    segment_autoencoder: torch.nn.Module,
    latent_normalizer: Any,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    suites: list[str] = []
    tasks: list[str] = []
    episodes: list[str] = []
    frames: list[int] = []
    decoded_loss_total = 0.0
    decoded_loss_weight = 0
    loss_config = config.get("loss", {})
    for batch in loader:
        vlm_tokens = batch["vlm_tokens"].to(device)
        state = batch["state"].to(device)
        action_segments = batch["action_segments"].to(device)
        mask = batch["action_segment_mask"].to(device)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            target_raw = segment_autoencoder.encode(action_segments)
            if model is None:
                pred_raw = None
                pred_loss = None
            else:
                output = model(vlm_tokens, state)
                pred_loss = output.predicted_latents
                pred_raw = latent_normalizer.unnormalize(pred_loss) if latent_normalizer is not None else pred_loss
                decoded = segment_autoencoder.decode(pred_raw)
                decoded_loss = action_segment_reconstruction_loss(
                    decoded,
                    action_segments,
                    mask,
                    gripper_indices=loss_config.get("gripper_indices", [-1]),
                    gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
                )
                decoded_loss_total += float(decoded_loss.detach().cpu().item()) * int(vlm_tokens.shape[0])
                decoded_loss_weight += int(vlm_tokens.shape[0])
            target_loss = latent_normalizer.normalize(target_raw) if latent_normalizer is not None else target_raw

        context = torch.cat([vlm_tokens.float().mean(dim=1), state.float()], dim=-1)
        tensors["target_raw"].append(target_raw.detach().float().cpu())
        tensors["target_loss"].append(target_loss.detach().float().cpu())
        tensors["mask"].append(mask.detach().float().cpu())
        tensors["context"].append(context.detach().float().cpu())
        if pred_raw is not None and pred_loss is not None:
            tensors["pred_raw"].append(pred_raw.detach().float().cpu())
            tensors["pred_loss"].append(pred_loss.detach().float().cpu())

        suites.extend(_string_list(batch.get("task_suite"), len(vlm_tokens), "unknown_suite"))
        tasks.extend(_string_list(batch.get("task_description"), len(vlm_tokens), "unknown_task"))
        episodes.extend(_string_list(batch.get("episode_id"), len(vlm_tokens), "unknown_episode"))
        frame_value = batch.get("frame_index")
        if isinstance(frame_value, torch.Tensor):
            frames.extend(int(value) for value in frame_value.reshape(-1).tolist())
        else:
            frames.extend([0] * int(vlm_tokens.shape[0]))

    result = {key: torch.cat(value, dim=0) for key, value in tensors.items()}
    result.update(
        {
            "num_samples": int(len(dataset)),
            "task_suite": suites,
            "task_description": tasks,
            "episode_id": episodes,
            "frame_index": frames,
            "decoded_chunk_loss": decoded_loss_total / max(decoded_loss_weight, 1),
        }
    )
    return result


def concatenate_collections(collections: list[dict[str, Any]]) -> dict[str, Any]:
    if not collections:
        raise ValueError("cannot concatenate an empty collection list")
    result: dict[str, Any] = {}
    tensor_keys = [key for key, value in collections[0].items() if isinstance(value, torch.Tensor)]
    for key in tensor_keys:
        result[key] = torch.cat([collection[key] for collection in collections if key in collection], dim=0)
    for key in ("task_suite", "task_description", "episode_id", "frame_index"):
        result[key] = []
        for collection in collections:
            result[key].extend(collection[key])
    total_samples = sum(int(collection["num_samples"]) for collection in collections)
    result["num_samples"] = total_samples
    result["decoded_chunk_loss"] = sum(
        float(collection["decoded_chunk_loss"]) * int(collection["num_samples"]) for collection in collections
    ) / max(total_samples, 1)
    return result


def summarize_collection(collection: dict[str, Any], *, latent_normalizer: Any) -> dict[str, Any]:
    pred_raw = collection["pred_raw"]
    target_raw = collection["target_raw"]
    pred_loss = collection["pred_loss"]
    target_loss = collection["target_loss"]
    mask = collection["mask"]
    overall = prediction_metrics_from_tensors(pred_raw, target_raw, pred_loss, target_loss, mask)
    overall["decoded_chunk_loss"] = float(collection.get("decoded_chunk_loss", 0.0))
    return {
        "num_samples": int(collection["num_samples"]),
        "overall": overall,
        "per_token": per_token_metrics(pred_raw, target_raw, pred_loss, target_loss, mask),
        "per_suite": grouped_sample_metrics(collection, group_key="task_suite"),
        "worst_tasks": worst_grouped_sample_metrics(collection, group_key="task_description", top_k=12),
        "per_dimension": per_dimension_metrics(pred_raw, target_raw, mask),
    }


def reference_summary(reference: dict[str, Any]) -> dict[str, Any]:
    target = reference["target_raw"]
    mask = reference["mask"]
    active = mask.bool()
    active_target = target[active]
    return {
        "num_samples": int(reference["num_samples"]),
        "active_segments": int(active.sum().item()),
        "latent_dim": int(target.shape[-1]),
        "target_mean_abs": float(active_target.abs().mean().item()) if active_target.numel() else 0.0,
        "target_std_mean": float(active_target.std(dim=0, unbiased=False).mean().item()) if active_target.numel() else 0.0,
    }


def prediction_baseline_metrics(
    pred_raw: torch.Tensor,
    collection: dict[str, Any],
    *,
    latent_normalizer: Any,
) -> dict[str, float]:
    pred_loss = latent_normalizer.normalize(pred_raw) if latent_normalizer is not None else pred_raw
    return prediction_metrics_from_tensors(
        pred_raw,
        collection["target_raw"],
        pred_loss,
        collection["target_loss"],
        collection["mask"],
    )


def prediction_metrics_from_tensors(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    pred_loss: torch.Tensor,
    target_loss: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    mask = mask.float()
    mask3 = mask.unsqueeze(-1)
    active_tokens = float(mask.sum().item())
    denom = max(active_tokens * float(pred_raw.shape[-1]), 1.0)
    raw_mse = float(((pred_raw - target_raw).pow(2) * mask3).sum().item() / denom)
    normalized_mse = float(((pred_loss - target_loss).pow(2) * mask3).sum().item() / denom)
    cosine = F.cosine_similarity(pred_raw.float(), target_raw.float(), dim=-1)
    cosine_value = float((cosine * mask).sum().item() / max(active_tokens, 1.0))
    return {
        "raw_latent_mse": raw_mse,
        "normalized_latent_mse": normalized_mse,
        "latent_cosine_similarity": cosine_value,
    }


def per_token_metrics(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    pred_loss: torch.Tensor,
    target_loss: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    latent_dim = int(pred_raw.shape[-1])
    for token_index in range(int(pred_raw.shape[1])):
        token_mask = mask[:, token_index].float()
        active = float(token_mask.sum().item())
        if active <= 0:
            rows.append({"token": token_index, "active": 0, "raw_latent_mse": 0.0, "normalized_latent_mse": 0.0})
            continue
        denom = active * latent_dim
        raw_mse = ((pred_raw[:, token_index] - target_raw[:, token_index]).pow(2) * token_mask[:, None]).sum() / denom
        norm_mse = ((pred_loss[:, token_index] - target_loss[:, token_index]).pow(2) * token_mask[:, None]).sum() / denom
        cosine = F.cosine_similarity(pred_raw[:, token_index].float(), target_raw[:, token_index].float(), dim=-1)
        rows.append(
            {
                "token": token_index,
                "active": int(active),
                "raw_latent_mse": float(raw_mse.item()),
                "normalized_latent_mse": float(norm_mse.item()),
                "latent_cosine_similarity": float((cosine * token_mask).sum().item() / max(active, 1.0)),
            }
        )
    return rows


def per_dimension_metrics(pred_raw: torch.Tensor, target_raw: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    active = mask.bool()
    pred = pred_raw[active]
    target = target_raw[active]
    if pred.numel() == 0:
        return {"mean_mse": 0.0, "median_mse": 0.0, "mean_r2": 0.0, "worst_dims": []}
    mse = (pred - target).pow(2).mean(dim=0)
    var = target.var(dim=0, unbiased=False)
    r2 = 1.0 - mse / var.clamp_min(1.0e-8)
    worst = torch.argsort(mse, descending=True)[:12]
    return {
        "mean_mse": float(mse.mean().item()),
        "median_mse": float(mse.median().item()),
        "mean_target_variance": float(var.mean().item()),
        "mean_r2": float(r2.mean().item()),
        "median_r2": float(r2.median().item()),
        "min_r2": float(r2.min().item()),
        "worst_dims": [
            {
                "dim": int(index.item()),
                "mse": float(mse[index].item()),
                "target_variance": float(var[index].item()),
                "r2": float(r2[index].item()),
            }
            for index in worst
        ],
    }


def grouped_sample_metrics(collection: dict[str, Any], *, group_key: str) -> dict[str, dict[str, float | int]]:
    values = _sample_raw_mse(collection)
    groups: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(collection[group_key], values, strict=True):
        groups[str(name)].append(float(value))
    return {
        name: {"count": len(items), "raw_latent_mse": float(sum(items) / max(len(items), 1))}
        for name, items in sorted(groups.items())
    }


def worst_grouped_sample_metrics(collection: dict[str, Any], *, group_key: str, top_k: int) -> list[dict[str, float | int | str]]:
    grouped = grouped_sample_metrics(collection, group_key=group_key)
    rows = [{"name": name, **metrics} for name, metrics in grouped.items()]
    return sorted(rows, key=lambda row: float(row["raw_latent_mse"]), reverse=True)[:top_k]


def _sample_raw_mse(collection: dict[str, Any]) -> list[float]:
    pred = collection["pred_raw"]
    target = collection["target_raw"]
    mask = collection["mask"].float()
    per_token = (pred - target).pow(2).mean(dim=-1)
    per_sample = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    return [float(value) for value in per_sample.tolist()]


def train_token_mean_prediction(reference: dict[str, Any], query: dict[str, Any]) -> torch.Tensor:
    train_target = reference["target_raw"]
    train_mask = reference["mask"].float()
    mean = (train_target * train_mask.unsqueeze(-1)).sum(dim=0) / train_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    return mean.unsqueeze(0).expand(query["target_raw"].shape[0], -1, -1).clone()


def input_nearest_neighbor_prediction(reference: dict[str, Any], query: dict[str, Any], *, chunk_size: int) -> torch.Tensor:
    train_context = standardize_context(reference["context"])
    query_context = apply_context_standardization(query["context"], reference["context"])
    indices = nearest_indices(query_context, train_context, chunk_size=chunk_size)
    return reference["target_raw"][indices].clone()


def oracle_target_nearest_neighbor_prediction(reference: dict[str, Any], query: dict[str, Any], *, chunk_size: int) -> torch.Tensor:
    train_target = reference["target_raw"]
    train_mask = reference["mask"].bool()
    query_target = query["target_raw"]
    query_mask = query["mask"].bool()
    pred = torch.zeros_like(query_target)
    fallback = train_token_mean_prediction(reference, query)
    for token in range(query_target.shape[1]):
        active_train = train_mask[:, token]
        active_query = query_mask[:, token]
        if active_train.sum().item() == 0:
            pred[:, token] = fallback[:, token]
            continue
        train_token = train_target[active_train, token]
        query_token = query_target[:, token]
        indices = nearest_indices(query_token, train_token, chunk_size=chunk_size)
        pred[:, token] = train_token[indices]
        pred[~active_query, token] = 0.0
    return pred


def standardize_context(context: torch.Tensor) -> torch.Tensor:
    mean = context.mean(dim=0, keepdim=True)
    std = context.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    return (context - mean) / std


def apply_context_standardization(context: torch.Tensor, train_context: torch.Tensor) -> torch.Tensor:
    mean = train_context.mean(dim=0, keepdim=True)
    std = train_context.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    return (context - mean) / std


def nearest_indices(query: torch.Tensor, keys: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    indices = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size].float()
        distances = torch.cdist(chunk, keys.float(), p=2)
        indices.append(distances.argmin(dim=1).cpu())
    return torch.cat(indices, dim=0).long()


def _string_list(value: Any, length: int, fallback: str) -> list[str]:
    if value is None:
        return [fallback] * int(length)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, torch.Tensor):
        return [str(item) for item in value.reshape(-1).tolist()]
    return [str(value)] * int(length)


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Coarse Planner Latent Error Diagnostics",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Target raw latent MSE: `{results['target_raw_latent_mse']}`",
        "",
        "## Train Reference",
        "",
    ]
    for key, value in results["train_reference"].items():
        lines.append(f"- `{key}`: `{value}`")
    for name, dataset in results["datasets"].items():
        overall = dataset["overall"]
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- samples: `{dataset['num_samples']}`",
                f"- raw latent MSE: `{overall['raw_latent_mse']:.6f}`",
                f"- normalized latent MSE: `{overall['normalized_latent_mse']:.6f}`",
                f"- decoded chunk loss: `{overall['decoded_chunk_loss']:.6f}`",
                f"- latent cosine: `{overall['latent_cosine_similarity']:.6f}`",
                f"- gap to 0.08: `{dataset['gap_to_target']['absolute_gap']:.6f}`",
                "",
                "### Baselines",
                "",
                "| baseline | raw MSE | normalized MSE | cosine |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for baseline_name, metrics in dataset["baselines"].items():
            lines.append(
                f"| {baseline_name} | {metrics['raw_latent_mse']:.6f} | "
                f"{metrics['normalized_latent_mse']:.6f} | {metrics['latent_cosine_similarity']:.6f} |"
            )
        lines.extend(["", "### Per Token", "", "| token | active | raw MSE | norm MSE | cosine |", "| ---: | ---: | ---: | ---: | ---: |"])
        for row in dataset["per_token"]:
            lines.append(
                f"| {row['token']} | {row['active']} | {row['raw_latent_mse']:.6f} | "
                f"{row['normalized_latent_mse']:.6f} | {row.get('latent_cosine_similarity', 0.0):.6f} |"
            )
        lines.extend(["", "### Worst Tasks", "", "| task | count | raw MSE |", "| --- | ---: | ---: |"])
        for row in dataset["worst_tasks"][:8]:
            task = str(row["name"]).replace("|", "/")
            lines.append(f"| {task} | {row['count']} | {row['raw_latent_mse']:.6f} |")
        dim = dataset["per_dimension"]
        lines.extend(
            [
                "",
                "### Per Dimension Summary",
                "",
                f"- mean dim MSE: `{dim['mean_mse']:.6f}`",
                f"- mean target variance: `{dim['mean_target_variance']:.6f}`",
                f"- mean R2: `{dim['mean_r2']:.6f}`",
                f"- min R2: `{dim['min_r2']:.6f}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
