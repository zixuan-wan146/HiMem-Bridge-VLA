from __future__ import annotations

import argparse
import logging
import os
import sys

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.training.stage2.libero.config import build_stage2_config


REPO_ROOT = find_repo_root(__file__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage2 full E2E LIBERO policy from raw episodes")
    parser.add_argument("--config", type=str, default=None, help="Project-relative Stage2 YAML config.")

    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_wandb", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_swanlab", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--bridge_himem_config", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_base_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--normalization_source_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--cache_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--save_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_checkpoint", type=str, default=argparse.SUPPRESS)

    parser.add_argument("--dataset_type", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--sequence_len", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--stage2_sampling_mode", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--sample_valid_future_horizon_only",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--shuffle_episodes", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--load_vlm", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_vlm", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_action_head", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_enabled", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_progress_planner", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--enable_bridge_aux_loss", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--horizon", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--per_action_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--state_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--memory_entry_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--short_memory_time_bins", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--action_head_ffn_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_plan_slots", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_vlm_tokens", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min_lr_ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad_clip_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--log_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_min_step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--resume_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--resume_pretrain", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--reset_best_loss_on_resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--num_inference_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inference_tau_schedule", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--avoid_endpoint_tau", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = build_arg_parser().parse_args(argv)
    config = build_stage2_config(args, repo_root=REPO_ROOT, validate_external_artifacts=True)
    from himem_bridge_vla.training.stage2.common.loop import train_stage2

    try:
        train_stage2(config, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Cleaning up Stage2 training...")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
