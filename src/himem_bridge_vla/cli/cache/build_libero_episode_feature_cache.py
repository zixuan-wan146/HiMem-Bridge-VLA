#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from himem_bridge_vla.dataset.libero import DEFAULT_LIBERO_VIEW_NAMES
from himem_bridge_vla.dataset.libero import LiberoEpisodeReader
from himem_bridge_vla.dataset.memory_token_cache import ImageStatsVLMHiddenStateEncoder
from himem_bridge_vla.dataset.memory_token_cache import ImageStatsVisualTokenEncoder
from himem_bridge_vla.dataset.memory_token_cache import InternVL3VLMHiddenStateEncoder
from himem_bridge_vla.dataset.memory_token_cache import InternVL3VisualTokenEncoder
from himem_bridge_vla.dataset.memory_token_cache import _build_minmax_normalization_manifest
from himem_bridge_vla.dataset.memory_token_cache import _update_running_minmax
from himem_bridge_vla.dataset.memory_token_cache import ensure_rank2_tokens
from himem_bridge_vla.dataset.memory_token_cache import resolve_torch_dtype
from himem_bridge_vla.path_utils import display_project_path
from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.utils.cuda_memory import cuda_memory_stats
from himem_bridge_vla.utils.cuda_memory import reserve_cuda_memory_floor


CACHE_FORMAT = "libero_episode_feature_cache"
CACHE_VERSION = 1
REPO_ROOT = find_repo_root(__file__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an episode-level processed LIBERO feature cache.")
    parser.add_argument("--episode-index", required=True, help="Episode replay JSON produced by build_libero_episode_replay_index.py.")
    parser.add_argument("--libero-root", default=None, help="Override LIBERO dataset root from the episode index.")
    parser.add_argument("--output-root", required=True, help="Directory for manifest.json and episode feature shards.")
    parser.add_argument("--encoder", choices=("internvl3", "image_stats"), default="internvl3")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--view-names", nargs="*", default=list(DEFAULT_LIBERO_VIEW_NAMES))
    parser.add_argument("--image-stats-hidden-dim", type=int, default=16)
    parser.add_argument("--image-stats-tokens-per-view", type=int, default=1)
    parser.add_argument("--include-vlm-hidden-states", action="store_true")
    parser.add_argument("--hidden-state-layers", nargs="*", default=("3", "6", "9", "12"))
    parser.add_argument(
        "--visual-batch-size",
        type=int,
        default=1,
        help="Number of frame-view images to encode in one visual-token forward pass.",
    )
    parser.add_argument(
        "--min-cuda-memory-gb",
        type=float,
        default=None,
        help="Reserve CUDA memory after encoder initialization so process usage stays above this floor.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-shard", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive when provided")
    if args.max_episodes_per_shard <= 0:
        raise ValueError("--max-episodes-per-shard must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.visual_batch_size <= 0:
        raise ValueError("--visual-batch-size must be positive")
    if args.min_cuda_memory_gb is not None and args.min_cuda_memory_gb <= 0:
        raise ValueError("--min-cuda-memory-gb must be positive when provided")
    view_names = tuple(str(name) for name in args.view_names)
    if not view_names:
        raise ValueError("--view-names must not be empty")

    index_path = Path(args.episode_index).expanduser()
    episode_index = json.loads(index_path.read_text(encoding="utf-8"))
    _validate_episode_index(episode_index, index_path)
    episodes = list(episode_index["episodes"])
    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    output_root = Path(args.output_root).expanduser()
    libero_root = Path(args.libero_root or episode_index["libero_root"]).expanduser()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "format": CACHE_FORMAT,
                    "episode_index": display_project_path(index_path, REPO_ROOT),
                    "libero_root": str(libero_root),
                    "output_root": display_project_path(output_root, REPO_ROOT),
                    "planned_episode_count": len(episodes),
                    "planned_node_count": sum(int(episode["node_count"]) for episode in episodes),
                    "planned_required_visual_frame_count": sum(
                        int(episode["required_visual_frame_count"]) for episode in episodes
                    ),
                    "encoder": args.encoder,
                    "include_vlm_hidden_states": bool(args.include_vlm_hidden_states),
                    "hidden_state_layers": [_parse_layer_selector(layer) for layer in args.hidden_state_layers],
                    "storage_dtype": args.storage_dtype,
                    "view_names": list(view_names),
                    "visual_batch_size": int(args.visual_batch_size),
                    "min_cuda_memory_gb": None
                    if args.min_cuda_memory_gb is None
                    else float(args.min_cuda_memory_gb),
                    "max_episodes_per_shard": int(args.max_episodes_per_shard),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    encoder = _build_encoder(args)
    hidden_state_encoder = _build_hidden_state_encoder(args, encoder=encoder)
    storage_dtype = resolve_torch_dtype(args.storage_dtype)
    memory_floor = _reserve_cuda_memory_floor(args)

    output_root.mkdir(parents=True, exist_ok=True)
    pending = []
    shards = []
    episode_count = 0
    node_count = 0
    required_visual_frame_count = 0
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None

    for episode in episodes:
        cached_episode = _encode_episode(
            episode,
            libero_root=libero_root,
            view_names=view_names,
            encoder=encoder,
            hidden_state_encoder=hidden_state_encoder,
            storage_dtype=storage_dtype,
            visual_batch_size=int(args.visual_batch_size),
        )
        action_min, action_max = _update_running_minmax(
            action_min,
            action_max,
            cached_episode["actions"],
            name="episode_actions",
        )
        state_min, state_max = _update_running_minmax(
            state_min,
            state_max,
            list(cached_episode["state_by_step"].values()),
            name="episode_state_by_step",
        )
        pending.append(cached_episode)
        episode_count += 1
        node_count += int(cached_episode["node_count"])
        required_visual_frame_count += int(cached_episode["required_visual_frame_count"])
        if len(pending) >= int(args.max_episodes_per_shard):
            shards.append(_write_shard(output_root, pending, start_index=episode_count - len(pending)))
            pending = []

    if pending:
        shards.append(_write_shard(output_root, pending, start_index=episode_count - len(pending)))

    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "benchmark": "LIBERO",
        "episode_index": str(index_path),
        "episode_index_format": episode_index["format"],
        "libero_root": str(libero_root),
        "output_root": str(output_root),
        "source_action_horizon": int(episode_index["action_horizon"]),
        "source_stride": int(episode_index["stride"]),
        "source_short_offsets": [int(offset) for offset in episode_index["short_offsets"]],
        "source_executed_action_stride": int(episode_index["executed_action_stride"]),
        "episode_count": episode_count,
        "node_count": node_count,
        "required_visual_frame_count": required_visual_frame_count,
        "encoder": encoder.name,
        "hidden_state_encoder": None if hidden_state_encoder is None else hidden_state_encoder.name,
        "hidden_state_layers": None
        if hidden_state_encoder is None
        else [_serialize_layer_selector(layer) for layer in hidden_state_encoder.selected_layers],
        "planner_vl_summary": None
        if hidden_state_encoder is None
        else {
            "enabled": bool(hasattr(hidden_state_encoder, "encode_current_features")),
            "source": "vlm_last_valid_token",
            "encoder": hidden_state_encoder.name,
        },
        "hidden_dim": int(encoder.hidden_dim),
        "tokens_per_view": None if encoder.tokens_per_view is None else int(encoder.tokens_per_view),
        "storage_dtype": str(args.storage_dtype),
        "view_names": list(view_names),
        "visual_batch_size": int(args.visual_batch_size),
        "min_cuda_memory_gb": None if args.min_cuda_memory_gb is None else float(args.min_cuda_memory_gb),
        "cuda_memory": _cuda_memory_stats(args.device),
        "cuda_memory_floor_reserved_gb": None if memory_floor is None else float(memory_floor.reserved_gb),
        "model_name": args.model_name if args.encoder == "internvl3" else None,
        "image_size": args.image_size if args.encoder == "internvl3" else None,
        "state_dim": None if state_min is None else int(state_min.shape[-1]),
        "action_dim": None if action_min is None else int(action_min.shape[-1]),
        "normalization": None
        if action_min is None or action_max is None or state_min is None or state_max is None
        else _build_minmax_normalization_manifest(
            benchmark="LIBERO",
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
        ),
        "action_normalization": None
        if action_min is None
        else {
            "enabled": True,
            "type": "train_split_minmax_to_minus_one_one",
            "clip_after_normalization": True,
            "clip_range": [-1.0, 1.0],
            "statistics_from": "episode_feature_cache",
        },
        "shards": shards,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "manifest": display_project_path(manifest_path, REPO_ROOT),
                "output_root": display_project_path(output_root, REPO_ROOT),
                "episode_count": episode_count,
                "node_count": node_count,
                "required_visual_frame_count": required_visual_frame_count,
                "shard_count": len(shards),
                "cuda_memory": manifest["cuda_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_episode_index(index: dict[str, Any], index_path: Path) -> None:
    if index.get("format") != "libero_episode_replay_index":
        raise ValueError(f"{index_path} is not a LIBERO episode replay index")
    if index.get("benchmark") != "LIBERO":
        raise ValueError(f"{index_path} benchmark must be LIBERO")
    if not isinstance(index.get("episodes"), list) or not index["episodes"]:
        raise ValueError(f"{index_path} contains no episodes")


def _encode_episode(
    episode: dict[str, Any],
    *,
    libero_root: Path,
    view_names: tuple[str, ...],
    encoder: Any,
    hidden_state_encoder: Any | None,
    storage_dtype: Any,
    visual_batch_size: int,
) -> dict[str, Any]:
    reader = LiberoEpisodeReader(
        libero_root / str(episode["source_path"]),
        demo_key=str(episode["episode_key"]),
        view_names=view_names,
    )
    prompt = str(episode["prompt"])
    state_by_step = {}
    current_features_by_step = {}
    node_current_steps = {int(node["current_step"]) for node in episode["nodes"]}
    frames_by_step = {}

    for step in [int(value) for value in episode["required_visual_steps"]]:
        frame = reader.read_frame(step)
        frames_by_step[int(step)] = frame
        state_by_step[int(step)] = np.asarray(frame.state_vector, dtype=np.float32)

    visual_tokens_by_step = _encode_visual_tokens_by_step(
        frames_by_step,
        encoder=encoder,
        storage_dtype=storage_dtype,
        batch_size=visual_batch_size,
    )

    if hidden_state_encoder is not None:
        for step in sorted(node_current_steps):
            frame = frames_by_step[int(step)]
            features = hidden_state_encoder.encode_current_features(frame.images_by_view, prompt)
            current_features_by_step[int(step)] = {
                "hidden_states": tuple(
                    np_or_tensor_to_storage(hidden_state, storage_dtype)
                    for hidden_state in features.hidden_states
                ),
                "planner_vl_summary": None
                if features.planner_vl_summary is None
                else np_or_tensor_to_storage(features.planner_vl_summary, storage_dtype).reshape(-1),
            }

    return {
        "episode_id": str(episode["episode_id"]),
        "suite": str(episode["suite"]),
        "task_name": str(episode["task_name"]),
        "prompt": prompt,
        "source_path": str(episode["source_path"]),
        "episode_key": str(episode["episode_key"]),
        "episode_length": int(episode["episode_length"]),
        "node_count": int(episode["node_count"]),
        "required_visual_steps": [int(step) for step in episode["required_visual_steps"]],
        "required_visual_frame_count": int(episode["required_visual_frame_count"]),
        "nodes": list(episode["nodes"]),
        "actions": reader.read_future_actions(0, int(episode["episode_length"])),
        "visual_tokens_by_step": visual_tokens_by_step,
        "state_by_step": state_by_step,
        "current_features_by_step": current_features_by_step,
    }


def _encode_visual_tokens_by_step(
    frames_by_step: dict[int, Any],
    *,
    encoder: Any,
    storage_dtype: Any,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    items = []
    for step in sorted(frames_by_step):
        for view_name, image in frames_by_step[step].images_by_view.items():
            items.append((int(step), str(view_name), image))
    if not items:
        return {}

    visual_tokens_by_step: dict[int, dict[str, Any]] = {int(step): {} for step in frames_by_step}
    for start in range(0, len(items), int(batch_size)):
        batch = items[start : start + int(batch_size)]
        if hasattr(encoder, "encode_images"):
            encoded_tokens = list(encoder.encode_images([image for _step, _view_name, image in batch]))
            if len(encoded_tokens) != len(batch):
                raise ValueError(
                    f"visual encoder returned {len(encoded_tokens)} images for {len(batch)} input images"
                )
        else:
            encoded_tokens = [
                encoder.encode_image(image)
                for _step, _view_name, image in batch
            ]
        for (step, view_name, _image), tokens in zip(batch, encoded_tokens, strict=True):
            visual_tokens_by_step[int(step)][str(view_name)] = ensure_rank2_tokens(
                tokens,
                storage_dtype=storage_dtype,
            )
    return visual_tokens_by_step


def np_or_tensor_to_storage(value: Any, dtype: Any):
    import torch

    return torch.as_tensor(value).detach().cpu().to(dtype=dtype)


def _write_shard(output_root: Path, episodes: list[dict[str, Any]], *, start_index: int) -> dict[str, Any]:
    import torch

    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    end_index = start_index + len(episodes)
    path = shard_dir / f"shard_{start_index:09d}_{end_index:09d}.pt"
    torch.save(
        {
            "format": CACHE_FORMAT,
            "version": CACHE_VERSION,
            "episodes": episodes,
        },
        path,
    )
    return {
        "path": str(path.relative_to(output_root)),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "episode_count": len(episodes),
        "node_count": sum(int(episode["node_count"]) for episode in episodes),
        "required_visual_frame_count": sum(int(episode["required_visual_frame_count"]) for episode in episodes),
    }


def _build_encoder(args: argparse.Namespace):
    if args.encoder == "image_stats":
        return ImageStatsVisualTokenEncoder(
            hidden_dim=args.image_stats_hidden_dim,
            tokens_per_view=args.image_stats_tokens_per_view,
        )
    return InternVL3VisualTokenEncoder(
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
        storage_dtype=args.storage_dtype,
    )


def _build_hidden_state_encoder(args: argparse.Namespace, *, encoder: Any):
    if not args.include_vlm_hidden_states:
        return None
    selected_layers = tuple(_parse_layer_selector(layer) for layer in args.hidden_state_layers)
    if not selected_layers:
        raise ValueError("--hidden-state-layers must not be empty when --include-vlm-hidden-states is set")
    if args.encoder == "image_stats":
        return ImageStatsVLMHiddenStateEncoder(
            hidden_dim=args.image_stats_hidden_dim,
            tokens_per_view=args.image_stats_tokens_per_view,
            selected_layers=selected_layers,
        )
    return InternVL3VLMHiddenStateEncoder(
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
        storage_dtype=args.storage_dtype,
        selected_layers=selected_layers,
        embedder=getattr(encoder, "embedder", None),
    )


def _reserve_cuda_memory_floor(args: argparse.Namespace):
    if args.min_cuda_memory_gb is None:
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--min-cuda-memory-gb requires CUDA")
    return reserve_cuda_memory_floor(
        torch,
        target_gb=float(args.min_cuda_memory_gb),
        device=args.device,
    )


def _cuda_memory_stats(device: str) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {}
    return dict(cuda_memory_stats(torch, device))


def _parse_layer_selector(raw: str) -> int | str:
    text = str(raw)
    try:
        return int(text)
    except ValueError:
        return text


def _serialize_layer_selector(layer: int | str) -> int | str:
    return int(layer) if isinstance(layer, int) else str(layer)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
