from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

from himem_bridge_vla.training.common.distributed import unwrap_training_model


def save_training_checkpoint(
    torch: Any,
    save_dir: str,
    *,
    step: int,
    model_engine: Any,
    optimizer: Any,
    scheduler: Any,
    loss: Any,
    accelerator: Any,
    config: dict[str, Any],
    norm_stats: dict[str, Any] | None,
    tag: str | None = None,
    best_loss: float | None = None,
) -> None:
    checkpoint_tag = tag or f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, checkpoint_tag)
    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning("Checkpoint directory %s exists. Removing before overwrite.", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()
    loss_value = float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss)
    client_state = {
        "step": step,
        "next_step": step + 1,
        "checkpoint_tag": checkpoint_tag,
        "loss": loss_value,
        "best_loss": float(best_loss) if best_loss is not None else loss_value,
        "config": config,
    } if accelerator.is_main_process else {}
    if hasattr(model_engine, "save_checkpoint"):
        model_engine.save_checkpoint(save_dir, tag=checkpoint_tag, client_state=client_state)
        checkpoint_type = "ds_model"
        checkpoint_file = "mp_rank_00_model_states.pt"
    else:
        checkpoint_type = "torch_model"
        checkpoint_file = "model.pt"

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if checkpoint_type == "torch_model":
            unwrapped = unwrap_training_model(accelerator, model_engine)
            torch.save(
                {
                    "format": "stage1_torch_checkpoint",
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "client_state": client_state,
                    "config": config,
                },
                os.path.join(checkpoint_dir, checkpoint_file),
            )
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return
    with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    if norm_stats is not None:
        with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w", encoding="utf-8") as handle:
            json.dump(norm_stats, handle, indent=2)
    with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w", encoding="utf-8") as handle:
        json.dump({"type": checkpoint_type, "version": 0.0, "checkpoints": checkpoint_file}, handle, indent=2)
    logging.info("Saved checkpoint to %s", checkpoint_dir)


def load_training_checkpoint(
    torch: Any,
    model_engine: Any,
    *,
    load_dir: str,
    accelerator: Any,
    tag: str,
    optimizer: Any,
    load_optimizer_states: bool,
    resume_pretrain: bool,
) -> tuple[int, dict[str, Any]]:
    if hasattr(model_engine, "load_checkpoint"):
        try:
            _load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=load_optimizer_states and not resume_pretrain,
                load_lr_scheduler_states=load_optimizer_states and not resume_pretrain,
            )
            if accelerator.is_main_process:
                logging.info("Loaded DeepSpeed checkpoint from %s/%s", load_dir, tag)
            return _client_state_step(client_state), client_state
        except Exception:
            _load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
            if accelerator.is_main_process:
                logging.info("Loaded checkpoint weights only from %s/%s", load_dir, tag)
            return _client_state_step(client_state), client_state

    checkpoint_path = os.path.join(load_dir, tag, "model.pt")
    payload = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    if payload.get("format") != "stage1_torch_checkpoint":
        raise ValueError(f"unsupported torch checkpoint format in {checkpoint_path}")
    unwrapped = unwrap_training_model(accelerator, model_engine)
    unwrapped.load_state_dict(payload["model_state_dict"], strict=True)
    if load_optimizer_states and not resume_pretrain and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    client_state = dict(payload.get("client_state") or {})
    if accelerator.is_main_process:
        logging.info("Loaded torch checkpoint from %s", checkpoint_path)
    return _client_state_step(client_state), client_state


def _client_state_step(client_state: dict[str, Any]) -> int:
    raw_step = client_state.get("next_step", client_state.get("step", 0))
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        logging.warning("Checkpoint client_state step %r is not numeric; resuming from 0.", raw_step)
        return 0
