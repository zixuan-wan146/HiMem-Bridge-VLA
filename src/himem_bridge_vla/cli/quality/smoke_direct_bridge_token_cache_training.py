#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_direct_bridge_token_cache_samples
from himem_bridge_vla.dataset.memory_token_cache import MEMORY_TOKEN_CACHE_FORMAT
from himem_bridge_vla.dataset.memory_token_cache import read_token_cache_manifest
from himem_bridge_vla.dataset.memory_token_cache import resolve_token_cache_manifest_path
from himem_bridge_vla.model.action_head.flow_matching import FlowmatchingActionHead
from himem_bridge_vla.model.planner import ProgressStateConfig
from himem_bridge_vla.model.planner import ProgressStatePlanner
from himem_bridge_vla.training_loss import masked_flow_matching_mse


@dataclass(frozen=True)
class TrainingSmokeShape:
    embed_dim: int
    hidden_dim: int
    horizon: int
    per_action_dim: int
    state_dim: int
    num_heads: int
    num_layers: int
    num_inference_timesteps: int
    current_tokens_per_view: int
    short_tokens_per_view: int
    memory_entry_tokens: int


PRESETS: dict[str, TrainingSmokeShape] = {
    "tiny": TrainingSmokeShape(
        embed_dim=32,
        hidden_dim=64,
        horizon=4,
        per_action_dim=3,
        state_dim=7,
        num_heads=4,
        num_layers=2,
        num_inference_timesteps=2,
        current_tokens_per_view=4,
        short_tokens_per_view=5,
        memory_entry_tokens=4,
    ),
    "final": TrainingSmokeShape(
        embed_dim=896,
        hidden_dim=1024,
        horizon=32,
        per_action_dim=7,
        state_dim=7,
        num_heads=8,
        num_layers=8,
        num_inference_timesteps=15,
        current_tokens_per_view=32,
        short_tokens_per_view=40,
        memory_entry_tokens=16,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test direct bridge-attn training from token-cache batches.")
    parser.add_argument("--preset", choices=sorted((*PRESETS, "auto")), default="tiny")
    parser.add_argument("--manifest", type=str, default=None, help="Optional replay-token cache manifest or cache root.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--action-horizon", type=int, default=None, help="Override action horizon for cache batches.")
    parser.add_argument("--memory-entry-tokens", type=int, default=None, help="Override packed tokens per short-memory entry.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override direct bridge block count.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override attention head count.")
    parser.add_argument(
        "--progress-planner-checkpoint",
        default=None,
        help="Optional progress-state planner warm-up checkpoint used to produce the plan token.",
    )
    return parser


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available")
    return device_arg


def build_synthetic_samples(shape: TrainingSmokeShape, *, batch_size: int) -> list[dict]:
    samples = []
    for index in range(batch_size):
        current_tokens_by_view = {
            "base": torch.randn(shape.current_tokens_per_view, shape.embed_dim),
            "wrist": torch.randn(shape.current_tokens_per_view, shape.embed_dim),
        }
        short_tokens_by_view = (
            {
                "base": torch.randn(shape.short_tokens_per_view, shape.embed_dim),
                "wrist": torch.randn(shape.short_tokens_per_view, shape.embed_dim),
            },
            {
                "base": torch.randn(shape.short_tokens_per_view, shape.embed_dim),
                "wrist": torch.randn(shape.short_tokens_per_view, shape.embed_dim),
            },
        )
        samples.append(
            {
                "sample_index": index,
                "benchmark": "SMOKE",
                "episode_id": f"smoke_{index}",
                "current_step": index,
                "current_tokens_by_view": current_tokens_by_view,
                "current_hidden_states": tuple(
                    torch.randn(shape.current_tokens_per_view * 2, shape.embed_dim)
                    for _ in range(4)
                ),
                "current_state": torch.randn(shape.state_dim),
                "short_tokens_by_view": short_tokens_by_view,
                "short_steps": torch.tensor([max(0, index - 16), max(0, index - 8)]),
                "short_mask": torch.tensor([True, True]),
                "future_actions": torch.randn(shape.horizon, shape.per_action_dim),
                "action_valid_count": shape.horizon,
                "executed_actions": torch.randn(16, shape.per_action_dim),
                "executed_action_mask": torch.ones(16, dtype=torch.bool),
            }
        )
    return samples


def load_cache_samples(manifest: str | Path, *, batch_size: int) -> tuple[list[dict], dict[str, object]]:
    manifest_path = resolve_token_cache_manifest_path(manifest)
    manifest_payload = read_token_cache_manifest(manifest_path)
    manifest_format = manifest_payload.get("format")
    if manifest_format != MEMORY_TOKEN_CACHE_FORMAT:
        raise ValueError(
            f"{manifest_path} has format {manifest_format!r}; expected {MEMORY_TOKEN_CACHE_FORMAT!r}. "
            "Progress warm-up embedding caches are not visual-token replay caches."
        )

    dataset = MemoryTokenCacheDataset(manifest_path, max_samples=batch_size)
    if len(dataset) < batch_size:
        raise ValueError(f"token cache has {len(dataset)} samples, but batch_size={batch_size}")
    return [dataset[index] for index in range(batch_size)], manifest_payload


def select_num_heads(embed_dim: int, preferred: int) -> int:
    preferred = int(preferred)
    if preferred > 0 and embed_dim % preferred == 0:
        return preferred
    for candidate in (16, 12, 8, 6, 4, 3, 2, 1):
        if candidate <= embed_dim and embed_dim % candidate == 0:
            return candidate
    return 1


def infer_shape_from_batch(
    base: TrainingSmokeShape,
    batch: dict[str, torch.Tensor],
    *,
    preset_name: str,
    memory_entry_tokens: int,
    num_layers: int | None,
    num_heads: int | None,
) -> TrainingSmokeShape:
    embed_dim = int(batch["fused_tokens"].shape[-1])
    horizon = int(batch["actions"].shape[1])
    per_action_dim = int(batch["actions"].shape[-1])
    state_dim = int(batch["states"].shape[-1])

    if preset_name == "final" and embed_dim == base.embed_dim:
        hidden_dim = base.hidden_dim
        inferred_layers = base.num_layers
        inference_steps = base.num_inference_timesteps
    else:
        hidden_dim = max(32, embed_dim * 2)
        inferred_layers = 2
        inference_steps = 2

    resolved_heads = select_num_heads(embed_dim, base.num_heads if num_heads is None else num_heads)
    return replace(
        base,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        per_action_dim=per_action_dim,
        state_dim=state_dim,
        num_heads=resolved_heads,
        num_layers=inferred_layers if num_layers is None else int(num_layers),
        num_inference_timesteps=inference_steps,
        current_tokens_per_view=int(batch["fused_tokens"].shape[1]),
        short_tokens_per_view=int(batch["memory_context"].shape[1]),
        memory_entry_tokens=int(memory_entry_tokens),
    )


def build_head(shape: TrainingSmokeShape, *, state_dim: int, per_action_dim: int, device: str) -> FlowmatchingActionHead:
    config = SimpleNamespace(
        embed_dim=shape.embed_dim,
        hidden_dim=shape.hidden_dim,
        ffn_dim=shape.embed_dim * 4,
        action_dim=shape.horizon * per_action_dim,
        horizon=shape.horizon,
        per_action_dim=per_action_dim,
        state_dim=state_dim,
        state_hidden_dim=max(shape.hidden_dim, shape.embed_dim),
        num_heads=shape.num_heads,
        num_layers=shape.num_layers,
        dropout=0.0,
        num_inference_timesteps=shape.num_inference_timesteps,
        num_categories=1,
        num_plan_slots=8,
        visual_gate_lambda=0.5,
        plan_gate_lambda=0.25,
        short_memory_time_bins=2,
        max_vlm_tokens=None,
    )
    return FlowmatchingActionHead(config=config).to(device)


def load_progress_planner(checkpoint_path: str, *, device: str) -> tuple[ProgressStatePlanner, dict[str, object]]:
    checkpoint_file = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "progress_state_planner_warmup":
        raise ValueError(f"invalid progress planner checkpoint format: {checkpoint.get('format')!r}")
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise KeyError(f"progress planner checkpoint lacks model_config: {checkpoint_file}")
    planner = ProgressStatePlanner(ProgressStateConfig(**model_config))
    planner.load_state_dict(checkpoint["model_state_dict"])
    planner.to(device)
    planner.eval()
    info = {
        "checkpoint": str(checkpoint_file),
        "hidden_dim": int(planner.config.hidden_dim),
        "state_dim": int(planner.config.state_dim),
        "action_dim": int(planner.config.action_dim),
        "replan_stride": int(planner.config.replan_stride),
    }
    return planner, info


def build_progress_plan_tokens(
    *,
    checkpoint_path: str,
    fused_tokens: torch.Tensor,
    states: torch.Tensor,
    executed_actions: torch.Tensor | None,
    executed_action_mask: torch.Tensor | None,
    per_action_dim: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    if executed_actions is None or executed_action_mask is None:
        raise ValueError("progress planner smoke requires executed_actions and executed_action_mask in the batch")
    planner, info = load_progress_planner(checkpoint_path, device=device)
    if int(planner.config.hidden_dim) != int(fused_tokens.shape[-1]):
        raise ValueError(f"progress planner hidden_dim {planner.config.hidden_dim} != token dim {fused_tokens.shape[-1]}")
    if int(planner.config.state_dim) != int(states.shape[-1]):
        raise ValueError(f"progress planner state_dim {planner.config.state_dim} != batch state dim {states.shape[-1]}")
    if int(planner.config.action_dim) != int(per_action_dim):
        raise ValueError(f"progress planner action_dim {planner.config.action_dim} != per-action dim {per_action_dim}")
    expected_action_shape = (int(fused_tokens.shape[0]), int(planner.config.replan_stride), int(planner.config.action_dim))
    if tuple(executed_actions.shape) != expected_action_shape:
        raise ValueError(f"executed_actions shape {tuple(executed_actions.shape)} != {expected_action_shape}")
    expected_mask_shape = expected_action_shape[:2]
    if tuple(executed_action_mask.shape) != expected_mask_shape:
        raise ValueError(f"executed_action_mask shape {tuple(executed_action_mask.shape)} != {expected_mask_shape}")

    with torch.no_grad():
        output = planner.forward_step(
            planner.initial_state(fused_tokens.shape[0], device=fused_tokens.device, dtype=fused_tokens.dtype),
            fused_tokens.mean(dim=1),
            states,
            executed_actions,
            executed_action_mask,
        )
    return output.planner_token.detach(), info


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    if args.preset == "auto" and not args.manifest:
        raise ValueError("--preset auto requires --manifest so dimensions can be inferred")
    base_shape = PRESETS["tiny"] if args.preset == "auto" else PRESETS[args.preset]
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if args.memory_entry_tokens is not None and int(args.memory_entry_tokens) <= 0:
        raise ValueError("--memory-entry-tokens must be positive")
    if args.action_horizon is not None and int(args.action_horizon) <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.num_layers is not None and int(args.num_layers) <= 0:
        raise ValueError("--num-layers must be positive")
    if args.num_heads is not None and int(args.num_heads) <= 0:
        raise ValueError("--num-heads must be positive")

    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    manifest_payload = None
    if args.manifest:
        samples, manifest_payload = load_cache_samples(args.manifest, batch_size=int(args.batch_size))
    else:
        samples = build_synthetic_samples(base_shape, batch_size=int(args.batch_size))
    memory_entry_tokens = (
        base_shape.memory_entry_tokens if args.memory_entry_tokens is None else int(args.memory_entry_tokens)
    )
    action_horizon = base_shape.horizon if args.action_horizon is None else int(args.action_horizon)
    batch = collate_direct_bridge_token_cache_samples(
        samples,
        memory_entry_tokens=memory_entry_tokens,
        action_horizon=action_horizon,
    )
    per_action_dim = int(batch["actions"].shape[-1])
    state_dim = int(batch["states"].shape[-1])
    shape = base_shape
    if args.manifest:
        shape = infer_shape_from_batch(
            base_shape,
            batch,
            preset_name=str(args.preset),
            memory_entry_tokens=memory_entry_tokens,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    else:
        shape = replace(
            shape,
            horizon=int(batch["actions"].shape[1]),
            per_action_dim=per_action_dim,
            state_dim=state_dim,
            num_layers=shape.num_layers if args.num_layers is None else int(args.num_layers),
            num_heads=select_num_heads(shape.embed_dim, shape.num_heads if args.num_heads is None else args.num_heads),
            memory_entry_tokens=memory_entry_tokens,
        )
    if int(batch["fused_tokens"].shape[-1]) != shape.embed_dim:
        raise ValueError(
            f"batch hidden dim {batch['fused_tokens'].shape[-1]} does not match preset {shape.embed_dim}"
        )
    if int(batch["actions"].shape[1]) != shape.horizon:
        raise ValueError(f"batch horizon {batch['actions'].shape[1]} does not match shape horizon {shape.horizon}")

    head = build_head(shape, state_dim=state_dim, per_action_dim=per_action_dim, device=device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr))

    fused_tokens = batch["fused_tokens"].to(device=device)
    states = batch["states"].to(device=device)
    actions = batch["actions"].to(device=device)
    action_mask = batch["action_mask"].to(device=device)
    memory_context = batch["memory_context"].to(device=device)
    memory_context_mask = batch["memory_context_mask"].to(device=device)
    short_memory_time_ids = batch["short_memory_time_ids"].to(device=device)
    vlm_hidden_states = batch.get("vlm_hidden_states")
    if vlm_hidden_states is not None:
        vlm_hidden_states = [hidden_state.to(device=device) for hidden_state in vlm_hidden_states]
    executed_actions = batch.get("executed_actions")
    executed_action_mask = batch.get("executed_action_mask")
    if executed_actions is not None:
        executed_actions = executed_actions.to(device=device, dtype=fused_tokens.dtype)
    if executed_action_mask is not None:
        executed_action_mask = executed_action_mask.to(device=device)

    progress_planner_info = None
    if args.progress_planner_checkpoint:
        plan_tokens, progress_planner_info = build_progress_plan_tokens(
            checkpoint_path=str(args.progress_planner_checkpoint),
            fused_tokens=fused_tokens,
            states=states,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            per_action_dim=per_action_dim,
            device=device,
        )
    else:
        plan_tokens = torch.randn(actions.shape[0], 1, shape.embed_dim, device=device)

    losses = []
    grad_norms = []
    for _ in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        pred_velocity, noise = head(
            fused_tokens,
            state=states,
            actions_gt=actions,
            action_mask=action_mask,
            vlm_hidden_states=vlm_hidden_states,
            short_memory_tokens=memory_context,
            short_memory_time_ids=short_memory_time_ids,
            short_memory_mask=memory_context_mask,
            plan_tokens=plan_tokens,
        )
        target_velocity = (actions - noise).reshape(actions.shape[0], -1)
        loss = masked_flow_matching_mse(pred_velocity, target_velocity, action_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training smoke loss: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(grad_norm.detach().cpu().item()))

    if not all(value > 0.0 for value in grad_norms):
        raise RuntimeError(f"expected positive gradient norms, got {grad_norms}")

    return {
        "device": device,
        "resolved_shape": asdict(shape),
        "preset": asdict(shape),
        "manifest": None if args.manifest is None else str(args.manifest),
        "manifest_format": None if manifest_payload is None else manifest_payload.get("format"),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "fused_tokens_shape": tuple(batch["fused_tokens"].shape),
        "memory_context_shape": tuple(batch["memory_context"].shape),
        "vlm_hidden_state_shapes": None
        if vlm_hidden_states is None
        else [tuple(hidden_state.shape) for hidden_state in vlm_hidden_states],
        "actions_shape": tuple(batch["actions"].shape),
        "losses": losses,
        "grad_norms": grad_norms,
        "finite": True,
        "progress_planner": progress_planner_info,
        "plan_token_source": "progress_planner" if progress_planner_info is not None else "random",
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_smoke(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
