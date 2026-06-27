from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from himem_bridge_vla.dataset.memory_replay import read_memory_replay_jsonl
from himem_bridge_vla.dataset.memory_replay_frames import MemoryReplayFrameReader
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ACTION_KEY
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ROBOT_KEY
from himem_bridge_vla.dataset.rmbench import read_rmbench_instruction
from himem_bridge_vla.model.planner.action_segment_autoencoder import ActionSegmentAutoencoder
from himem_bridge_vla.model.planner.action_segment_autoencoder import ActionSegmentAutoencoderConfig
from himem_bridge_vla.utils.normalization import NormalizationStats


RMBENCH_PROGRESS_WARMUP_FORMAT = "libero_progress_vl_embedding_warmup_cache"
RMBENCH_PROGRESS_WARMUP_VERSION = 2


ActionNormalizer = Callable[[torch.Tensor], torch.Tensor]


class RMBenchVLSummaryEncoder(Protocol):
    name: str
    hidden_dim: int

    def encode_batch(self, batch: Sequence[tuple[Mapping[str, Image.Image], str]]) -> list[torch.Tensor]:
        ...


@dataclass(frozen=True)
class RMBenchProgressWarmupBuildResult:
    output_root: Path
    manifest_path: Path
    step_count: int
    window_count: int


def build_rmbench_progress_vl_embedding_cache(
    *,
    rmbench_root: str | Path,
    index_path: str | Path,
    output_root: str | Path,
    vl_encoder: RMBenchVLSummaryEncoder,
    action_horizon: int = 32,
    replan_stride: int = 16,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
    intent_encoder: ActionSegmentAutoencoder | None = None,
    intent_encoder_checkpoint: str | Path | None = None,
    action_normalizer: ActionNormalizer | None = None,
    norm_stats_path: str | Path | None = None,
    robot_key: str | None = DEFAULT_RMBENCH_ROBOT_KEY,
    storage_dtype: torch.dtype = torch.float32,
    view_names: Sequence[str] | None = None,
    max_steps: int | None = None,
    progress_interval: int | None = 100,
    vl_batch_size: int = 1,
) -> RMBenchProgressWarmupBuildResult:
    if action_horizon <= 0 or replan_stride <= 0 or burnin_replan_steps < 0 or loss_replan_steps <= 0:
        raise ValueError("invalid horizon/stride/window configuration")
    if int(vl_batch_size) <= 0:
        raise ValueError("vl_batch_size must be positive")

    rows = read_memory_replay_jsonl(index_path)
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        rows = rows[: int(max_steps)]
    if not rows:
        raise ValueError(f"RMBench replay index has no rows: {index_path}")

    data_path_root = Path(rmbench_root).expanduser()
    normalizer = action_normalizer or (lambda tensor: tensor)
    reader = MemoryReplayFrameReader(benchmark="RMBench", data_root=data_path_root, view_names=view_names)
    row_by_episode_step = {
        (str(row["episode_id"]), int(row["current_step"])): row
        for row in rows
        if str(row.get("benchmark", "")).upper() == "RMBENCH"
    }
    prompt_cache: dict[str, str] = {}

    steps: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        if str(row.get("benchmark", "")).upper() != "RMBENCH":
            continue
        current_step = int(row["current_step"])
        episode_id = str(row["episode_id"])
        if current_step % int(replan_stride) != 0:
            continue
        if int(row["action_valid_count"]) < int(action_horizon):
            continue

        sample = reader.read(row)
        future_actions = torch.as_tensor(sample.future_actions, dtype=torch.float32)
        if future_actions.shape != (int(action_horizon), 14):
            continue
        target_actions = normalizer(future_actions[:action_horizon]).float()

        if current_step == 0:
            executed_actions = torch.zeros(replan_stride, target_actions.shape[-1], dtype=torch.float32)
            executed_mask = torch.zeros(replan_stride, dtype=torch.bool)
        else:
            prev_row = row_by_episode_step.get((episode_id, current_step - int(replan_stride)))
            if prev_row is None:
                continue
            executed_raw = read_rmbench_action_slice(
                data_path_root,
                prev_row,
                start=current_step - int(replan_stride),
                end=current_step,
            )
            if executed_raw.shape != (int(replan_stride), 14):
                continue
            executed_actions = normalizer(torch.as_tensor(executed_raw, dtype=torch.float32)).float()
            executed_mask = torch.ones(replan_stride, dtype=torch.bool)

        prompt = rmbench_prompt_for_row(data_path_root, row, prompt_cache)
        target_intent = encode_target_intent(intent_encoder, target_actions)
        pending.append(
            {
                "images_by_view": sample.current.images_by_view,
                "step": {
                    "step_index": -1,
                    "sample_index": int(sample_index),
                    "episode_id": episode_id,
                    "suite": str(row.get("task_name") or episode_id.split(":")[0]),
                    "task_name": str(row.get("task_name", "")),
                    "prompt": prompt,
                    "current_step": current_step,
                    "replan_index": current_step // int(replan_stride),
                    "state": torch.as_tensor(sample.current.state_vector, dtype=torch.float32).cpu(),
                    "executed_actions": executed_actions.cpu(),
                    "executed_action_mask": executed_mask.cpu(),
                    "target_intent": target_intent.cpu(),
                },
            }
        )
        if len(pending) >= int(vl_batch_size):
            flush_rmbench_vl_summary_batch(
                pending,
                steps,
                vl_encoder=vl_encoder,
                storage_dtype=storage_dtype,
                progress_interval=progress_interval,
            )
            pending = []

    if pending:
        flush_rmbench_vl_summary_batch(
            pending,
            steps,
            vl_encoder=vl_encoder,
            storage_dtype=storage_dtype,
            progress_interval=progress_interval,
        )

    windows = build_rmbench_progress_windows(
        steps,
        burnin_replan_steps=burnin_replan_steps,
        loss_replan_steps=loss_replan_steps,
        allow_short_burnin=allow_short_burnin,
    )
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "data.pt"
    torch.save(
        {
            "format": RMBENCH_PROGRESS_WARMUP_FORMAT,
            "version": RMBENCH_PROGRESS_WARMUP_VERSION,
            "steps": steps,
            "windows": windows,
        },
        data_path,
    )
    manifest = {
        "format": RMBENCH_PROGRESS_WARMUP_FORMAT,
        "version": RMBENCH_PROGRESS_WARMUP_VERSION,
        "benchmark": "RMBench",
        "data_root": str(data_path_root),
        "index_path": str(Path(index_path).expanduser()),
        "data_path": data_path.name,
        "embedding": "vl_summary",
        "encoder": str(getattr(vl_encoder, "name", vl_encoder.__class__.__name__)),
        "hidden_dim": int(getattr(vl_encoder, "hidden_dim", int(steps[0]["vl_summary"].shape[-1]) if steps else 0)),
        "view_names": None if view_names is None else [str(name) for name in view_names],
        "action_horizon": int(action_horizon),
        "replan_stride": int(replan_stride),
        "burnin_replan_steps": int(burnin_replan_steps),
        "loss_replan_steps": int(loss_replan_steps),
        "allow_short_burnin": bool(allow_short_burnin),
        "vl_batch_size": int(vl_batch_size),
        "intent_encoder_checkpoint": None if intent_encoder_checkpoint is None else str(Path(intent_encoder_checkpoint).expanduser()),
        "norm_stats_path": None if norm_stats_path is None else str(Path(norm_stats_path).expanduser()),
        "robot_key": robot_key,
        "step_count": len(steps),
        "window_count": len(windows),
        "suite_window_counts": rmbench_window_suite_counts(windows),
        "sampler": {
            "default": "temperature_suite",
            "sampling_alpha": 0.5,
            "samples_per_epoch": 8192,
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RMBenchProgressWarmupBuildResult(
        output_root=output_path,
        manifest_path=manifest_path,
        step_count=len(steps),
        window_count=len(windows),
    )


def build_rmbench_progress_windows(
    steps: Sequence[Mapping[str, Any]],
    *,
    burnin_replan_steps: int = 8,
    loss_replan_steps: int = 8,
    allow_short_burnin: bool = True,
) -> list[dict[str, Any]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for step in steps:
        by_episode[str(step["episode_id"])].append(step)
    windows: list[dict[str, Any]] = []
    for episode_id, episode_steps in sorted(by_episode.items()):
        ordered = sorted(episode_steps, key=lambda step: int(step["replan_index"]))
        for start_pos in range(0, len(ordered) - int(loss_replan_steps) + 1):
            loss_steps = ordered[start_pos : start_pos + int(loss_replan_steps)]
            expected = list(range(int(loss_steps[0]["replan_index"]), int(loss_steps[0]["replan_index"]) + int(loss_replan_steps)))
            if [int(step["replan_index"]) for step in loss_steps] != expected:
                continue
            if not allow_short_burnin and start_pos < int(burnin_replan_steps):
                continue
            ctx_start_pos = max(0, start_pos - int(burnin_replan_steps))
            burnin_steps = ordered[ctx_start_pos:start_pos]
            start_k = int(loss_steps[0]["replan_index"])
            windows.append(
                {
                    "window_index": len(windows),
                    "episode_id": episode_id,
                    "suite": str(loss_steps[0].get("suite", "")),
                    "task_name": str(loss_steps[0].get("task_name", "")),
                    "ctx_start": int(ordered[ctx_start_pos]["replan_index"]) if burnin_steps else start_k,
                    "start_k": start_k,
                    "burnin_step_indices": [int(step["step_index"]) for step in burnin_steps],
                    "loss_step_indices": [int(step["step_index"]) for step in loss_steps],
                }
            )
    return windows


def flush_rmbench_vl_summary_batch(
    pending: Sequence[Mapping[str, Any]],
    steps: list[dict[str, Any]],
    *,
    vl_encoder: RMBenchVLSummaryEncoder,
    storage_dtype: torch.dtype,
    progress_interval: int | None,
) -> None:
    summaries = vl_encoder.encode_batch(
        [(item["images_by_view"], item["step"]["prompt"]) for item in pending]
    )
    if len(summaries) != len(pending):
        raise ValueError(f"VL encoder returned {len(summaries)} summaries for {len(pending)} pending samples")
    for item, vl_summary in zip(pending, summaries, strict=True):
        step = dict(item["step"])
        step["step_index"] = len(steps)
        step["vl_summary"] = torch.as_tensor(vl_summary).to(dtype=storage_dtype).cpu()
        steps.append(step)
        if progress_interval and len(steps) % int(progress_interval) == 0:
            print(
                json.dumps(
                    {
                        "event": "rmbench_progress_vl_embedding_build",
                        "steps": len(steps),
                        "sample_index": int(step["sample_index"]),
                        "episode_id": str(step["episode_id"]),
                        "current_step": int(step["current_step"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def rmbench_action_normalizer_from_stats(norm_stats_path: str | Path | None, *, robot_key: str | None = DEFAULT_RMBENCH_ROBOT_KEY) -> ActionNormalizer:
    if norm_stats_path is None:
        return lambda tensor: tensor
    stats = NormalizationStats(norm_stats_path, robot_key=robot_key)
    return lambda tensor: stats.normalize_action(tensor, robot_key=robot_key)


def load_rmbench_action_segment_autoencoder(checkpoint_path: str | Path, *, device: str | torch.device = "cpu") -> ActionSegmentAutoencoder:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device, weights_only=False)
    raw_config = checkpoint.get("segment_autoencoder_config")
    if raw_config is None:
        raise KeyError(f"checkpoint lacks segment_autoencoder_config: {checkpoint_path}")
    if int(raw_config.get("action_dim", -1)) != 14:
        raise ValueError(f"RMBench action AE must use action_dim=14, got {raw_config.get('action_dim')}")
    model = ActionSegmentAutoencoder(ActionSegmentAutoencoderConfig(**raw_config)).to(device)
    model.load_state_dict(checkpoint["segment_autoencoder_state_dict"])
    model.eval()
    return model


def encode_target_intent(intent_encoder: ActionSegmentAutoencoder | None, target_actions: torch.Tensor) -> torch.Tensor:
    if intent_encoder is None:
        flat = target_actions.reshape(-1)
        if flat.numel() < 128:
            flat = F.pad(flat, (0, 128 - flat.numel()))
        return F.normalize(flat[:128], dim=-1).float()
    device = next(intent_encoder.parameters()).device
    with torch.no_grad():
        latent = intent_encoder.encode(target_actions.to(device).unsqueeze(0)).squeeze(0)
    return F.normalize(latent.detach().cpu().float(), dim=-1)


def rmbench_prompt_for_row(data_root: Path, row: Mapping[str, Any], cache: dict[str, str]) -> str:
    instruction_path = str(row.get("instruction_path") or "")
    if instruction_path and instruction_path in cache:
        return cache[instruction_path]
    prompt = read_rmbench_instruction(data_root / instruction_path) if instruction_path else ""
    if not prompt:
        prompt = str(row.get("task_name") or "").replace("_", " ").strip()
    if instruction_path:
        cache[instruction_path] = prompt
    return prompt


def read_rmbench_action_slice(rmbench_root: Path, row: Mapping[str, Any], *, start: int, end: int) -> np.ndarray:
    import h5py

    with h5py.File(rmbench_root / str(row["source_path"]), "r") as handle:
        return np.asarray(handle[DEFAULT_RMBENCH_ACTION_KEY][int(start) : int(end)], dtype=np.float32)


def rmbench_window_suite_counts(windows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for window in windows:
        counts[str(window["suite"])] += 1
    return dict(sorted(counts.items()))


def rmbench_resolve_storage_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported storage dtype: {name!r}")
