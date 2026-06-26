#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.model.action_head.flow_matching import FlowmatchingActionHead
from himem_bridge_vla.model.planner import ProgressStateConfig
from himem_bridge_vla.model.planner import ProgressStatePlanner


@dataclass(frozen=True)
class SmokeShape:
    embed_dim: int
    hidden_dim: int
    horizon: int
    per_action_dim: int
    num_heads: int
    num_layers: int
    num_inference_timesteps: int
    vlm_tokens: int
    short_memory_tokens: int


PRESETS: dict[str, SmokeShape] = {
    "tiny": SmokeShape(
        embed_dim=32,
        hidden_dim=64,
        horizon=4,
        per_action_dim=3,
        num_heads=4,
        num_layers=2,
        num_inference_timesteps=2,
        vlm_tokens=8,
        short_memory_tokens=8,
    ),
    "final": SmokeShape(
        embed_dim=896,
        hidden_dim=1024,
        horizon=32,
        per_action_dim=7,
        num_heads=8,
        num_layers=8,
        num_inference_timesteps=15,
        vlm_tokens=64,
        short_memory_tokens=32,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test direct bridge-attention action inference.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="final")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
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


def run_smoke(
    shape: SmokeShape,
    *,
    device: str,
    seed: int,
    progress_planner_checkpoint: str | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    action_dim = shape.horizon * shape.per_action_dim
    head = FlowmatchingActionHead(
        embed_dim=shape.embed_dim,
        hidden_dim=shape.hidden_dim,
        action_dim=action_dim,
        horizon=shape.horizon,
        per_action_dim=shape.per_action_dim,
        num_heads=shape.num_heads,
        num_layers=shape.num_layers,
        num_inference_timesteps=shape.num_inference_timesteps,
    ).to(device)
    head.eval()

    with torch.no_grad():
        fused_tokens = torch.randn(1, shape.vlm_tokens, shape.embed_dim, device=device)
        hidden_states = [torch.randn(1, shape.vlm_tokens, shape.embed_dim, device=device) for _ in range(4)]
        short_memory = torch.randn(1, shape.short_memory_tokens, shape.embed_dim, device=device)
        split = shape.short_memory_tokens // 2
        short_time_ids = torch.tensor([[0] * split + [1] * (shape.short_memory_tokens - split)], device=device)
        short_mask = torch.ones(1, shape.short_memory_tokens, dtype=torch.bool, device=device)
        progress_planner_info = None
        if progress_planner_checkpoint:
            planner, progress_planner_info = load_progress_planner(progress_planner_checkpoint, device=device)
            if planner.config.hidden_dim != shape.embed_dim:
                raise ValueError(
                    f"progress planner hidden_dim {planner.config.hidden_dim} does not match preset embed_dim {shape.embed_dim}"
                )
            progress_output = planner.forward_step(
                planner.initial_state(1, device=torch.device(device), dtype=fused_tokens.dtype),
                fused_tokens.mean(dim=1),
                torch.randn(1, planner.config.state_dim, device=device, dtype=fused_tokens.dtype),
                torch.randn(
                    1,
                    planner.config.replan_stride,
                    planner.config.action_dim,
                    device=device,
                    dtype=fused_tokens.dtype,
                ),
                torch.ones(1, planner.config.replan_stride, dtype=torch.bool, device=device),
            )
            plan_tokens = progress_output.planner_token
        else:
            plan_tokens = torch.randn(1, 1, shape.embed_dim, device=device)
        state = torch.randn(1, 7, device=device)
        action_mask = torch.ones(1, shape.per_action_dim, device=device)
        action_mask[:, -1] = 0

        action = head.get_action(
            fused_tokens,
            state=state,
            action_mask=action_mask,
            vlm_hidden_states=hidden_states,
            short_memory_tokens=short_memory,
            short_memory_time_ids=short_time_ids,
            short_memory_mask=short_mask,
            plan_tokens=plan_tokens,
        ).view(1, shape.horizon, shape.per_action_dim)

    finite = bool(torch.isfinite(action).all().item())
    masked_abs_max = float(action[:, :, -1].abs().max().item())
    if not finite:
        raise RuntimeError("direct bridge inference produced non-finite actions")
    if masked_abs_max != 0.0:
        raise RuntimeError(f"masked action dimension changed during inference: abs max={masked_abs_max}")

    return {
        "device": device,
        "preset": shape.__dict__,
        "action_shape": tuple(action.shape),
        "finite": finite,
        "masked_last_dim_abs_max": masked_abs_max,
        "progress_planner": progress_planner_info,
    }


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


def main() -> int:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    result = run_smoke(
        PRESETS[args.preset],
        device=device,
        seed=args.seed,
        progress_planner_checkpoint=args.progress_planner_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
