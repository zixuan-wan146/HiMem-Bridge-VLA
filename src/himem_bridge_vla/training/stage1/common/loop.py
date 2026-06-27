from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from himem_bridge_vla.experiment_config import resolve_experiment_config
from himem_bridge_vla.reproducibility import set_global_seed, write_experiment_snapshot
from himem_bridge_vla.training.common.checkpoint import load_training_checkpoint, save_training_checkpoint
from himem_bridge_vla.training.common.distributed import get_and_clip_grad_norm, unwrap_training_model
from himem_bridge_vla.training.common.logging import setup_file_logging
from himem_bridge_vla.training.common.optimizer import build_param_groups
from himem_bridge_vla.training.common.scheduler import get_lr_lambda
from himem_bridge_vla.training.stage1.common.batch_contract import validate_stage1_window_batch
from himem_bridge_vla.training.stage1.common.dataset import prepare_stage1_dataloader, prepare_stage1_dataset
from himem_bridge_vla.training.stage1.common.loss import stage1_flow_matching_loss
from himem_bridge_vla.training.stage1.libero.validators import enforce_stage1_contract
from himem_bridge_vla.training_config import resolve_training_config_paths, validate_training_config


def train_stage1(config: dict[str, Any], *, repo_root: str | Path) -> None:
    runtime = _load_runtime()
    torch = runtime["torch"]
    Accelerator = runtime["Accelerator"]
    DistributedType = runtime["DistributedType"]
    AdamW = runtime["AdamW"]
    LambdaLR = runtime["LambdaLR"]
    tqdm = runtime["tqdm"]
    HiMemBridgeVLA = runtime["HiMemBridgeVLA"]

    repo_root = Path(repo_root)
    accelerator = Accelerator()
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    enforce_stage1_contract(config)
    validate_training_config(config, repo_root=repo_root)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)

    save_dir = str(config.get("save_dir", "local_data/runs/stage1/default"))
    setup_file_logging(save_dir, is_main_process=accelerator.is_main_process, filename_prefix="stage1_train_log")
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
        logging.info("Resolved Stage1 config written to %s", save_dir)
        logging.info("Seed=%s deterministic=%s", seed, deterministic)

    dataset = prepare_stage1_dataset(config, repo_root=repo_root)
    dataloader = prepare_stage1_dataloader(dataset, config)

    model = HiMemBridgeVLA(config)
    config = model.config
    enforce_stage1_contract(config)
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
    model.train()
    model.set_finetune_flags()

    lr = float(config.get("lr", 5e-5))
    weight_decay = float(config.get("weight_decay", 1e-3))
    param_groups = build_param_groups(
        model,
        weight_decay,
        base_lr=lr,
        lr_groups=config.get("lr_groups") or {},
    )
    optimizer = AdamW(param_groups, lr=lr)
    if accelerator.is_main_process:
        logging.info("Optimizer=AdamW base_lr=%s weight_decay=%s", lr, weight_decay)
        for index, group in enumerate(param_groups):
            params = sum(parameter.numel() for parameter in group["params"])
            logging.info(
                "Optimizer group %s | lr=%s | weight_decay=%s | params=%.3fM",
                group.get("name", f"group_{index}"),
                group.get("lr", lr),
                group.get("weight_decay", weight_decay),
                params / 1e6,
            )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model
    unwrapped_model = unwrap_training_model(accelerator, model)

    max_steps = int(config.get("max_steps", 10000))
    warmup_steps = int(config.get("warmup_steps", 1000))
    scheduler = LambdaLR(
        optimizer,
        get_lr_lambda(
            warmup_steps,
            max_steps,
            resume_step=0,
            min_lr_ratio=float(config.get("min_lr_ratio", 0.1)),
        ),
    )

    os.makedirs(save_dir, exist_ok=True)
    norm_stats = getattr(dataset, "arm2stats_dict", None)
    step = 0
    best_loss = float("inf")

    if bool(config.get("resume", False)):
        resume_path = str(config.get("resume_path", "")).rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)
        step, client_state = load_training_checkpoint(
            torch,
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            optimizer=optimizer,
            load_optimizer_states=True,
            resume_pretrain=bool(config.get("resume_pretrain", False)),
        )
        best_loss = float(client_state.get("best_loss", float("inf")))
        scheduler = LambdaLR(
            optimizer,
            get_lr_lambda(
                warmup_steps,
                max_steps,
                resume_step=step,
                min_lr_ratio=float(config.get("min_lr_ratio", 0.1)),
            ),
        )
        if accelerator.is_main_process:
            logging.info("Resuming Stage1 from %s/%s at step %s", resume_dir, resume_tag, step)
    elif accelerator.is_main_process:
        logging.info("Starting fresh Stage1 training")

    log_interval = int(config.get("log_interval", 10))
    ckpt_interval = int(config.get("ckpt_interval", 5000))
    best_ckpt_enabled = int(config.get("best_ckpt_interval", 1000)) != 0
    best_ckpt_min_step = int(config.get("best_ckpt_min_step", config.get("warmup_steps", 0)))
    max_norm = float(config.get("grad_clip_norm", 1.0))
    bridge_enabled = bool(unwrapped_model.use_bridge)
    last_loss = None

    while step < max_steps:
        for batch in tqdm(dataloader, desc="Stage1", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            validate_stage1_window_batch(batch)

            loss, extra_metrics, last_tensors = _run_trajectory_window_batch(
                torch=torch,
                model=model,
                unwrapped_model=unwrapped_model,
                batch=batch,
                config=config,
                accelerator=accelerator,
                bridge_enabled=bridge_enabled,
                step=step,
            )
            if not _check_numerical_stability(torch, step, loss=loss, **last_tensors):
                raise FloatingPointError(f"Non-finite tensor detected at Stage1 step {step}")
            last_tensors.clear()

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            total_norm, clipped_norm = get_and_clip_grad_norm(torch, accelerator, model, loss, max_norm)
            optimizer.step()
            scheduler.step()

            if step % log_interval == 0:
                _log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, extra_metrics)

            loss_value = float(loss.detach().cpu().item())
            checkpoint_loss = loss.detach()
            last_loss = checkpoint_loss
            is_best_tensor = torch.tensor(
                int(accelerator.is_main_process and step >= best_ckpt_min_step and loss_value < best_loss),
                device=accelerator.device,
            )
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)

            is_best = is_best_tensor.item() == 1
            if is_best:
                best_loss = loss_value

            should_save_best = is_best and best_ckpt_enabled
            if should_save_best:
                save_training_checkpoint(
                    torch,
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=checkpoint_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=norm_stats,
                    tag="step_best",
                    best_loss=best_loss,
                )
                if accelerator.is_main_process:
                    logging.info("Saved best Stage1 checkpoint at step %s loss %.6f", step, loss_value)

            step += 1
            if step % ckpt_interval == 0 and step > 0:
                save_training_checkpoint(
                    torch,
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=checkpoint_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=norm_stats,
                    best_loss=best_loss if best_loss < float("inf") else None,
                )
            del loss, checkpoint_loss

    if last_loss is None:
        raise RuntimeError("Stage1 training loop did not run any steps")
    save_training_checkpoint(
        torch,
        save_dir,
        step=step,
        model_engine=model_engine,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=last_loss,
        accelerator=accelerator,
        config=config,
        norm_stats=norm_stats,
        tag="step_final",
        best_loss=best_loss if best_loss < float("inf") else None,
    )
    if accelerator.is_main_process:
        logging.info("Final Stage1 checkpoint saved to step_final/")


def _load_runtime() -> dict[str, Any]:
    try:
        import torch
        from accelerate import Accelerator
        from accelerate import DistributedType
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        from tqdm import tqdm

        from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
    except ModuleNotFoundError as exc:
        missing = exc.name or "a Stage1 training dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for Stage1 training. Run inside the prepared training environment."
        ) from exc

    return {
        "torch": torch,
        "Accelerator": Accelerator,
        "DistributedType": DistributedType,
        "AdamW": AdamW,
        "LambdaLR": LambdaLR,
        "tqdm": tqdm,
        "HiMemBridgeVLA": HiMemBridgeVLA,
    }


def _get_autocast_context(torch: Any, device: Any):
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optional_batch_tensor(torch: Any, batch: dict[str, Any], key: str, device: Any, dtype: Any | None):
    value = batch.get(key)
    if value is None:
        return None
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _slice_progress_state(progress_state: Any, batch_indices: Any) -> Any:
    from himem_bridge_vla.model.planner import ProgressState

    if isinstance(progress_state, ProgressState):
        return ProgressState(
            completed_events=progress_state.completed_events.index_select(0, batch_indices),
            current_stage=progress_state.current_stage.index_select(0, batch_indices),
        )
    return progress_state.index_select(0, batch_indices)


def _scatter_progress_state(progress_state: Any, batch_indices: Any, updated_state: Any) -> Any:
    from himem_bridge_vla.model.planner import ProgressState

    if isinstance(progress_state, ProgressState):
        completed = progress_state.completed_events.clone()
        current = progress_state.current_stage.clone()
        completed.index_copy_(
            0,
            batch_indices,
            updated_state.completed_events.to(device=completed.device, dtype=completed.dtype),
        )
        current.index_copy_(
            0,
            batch_indices,
            updated_state.current_stage.to(device=current.device, dtype=current.dtype),
        )
        return ProgressState(completed_events=completed, current_stage=current)
    output = progress_state.clone()
    output.index_copy_(0, batch_indices, updated_state.to(device=output.device, dtype=output.dtype))
    return output


def _run_trajectory_window_batch(
    *,
    torch: Any,
    model: Any,
    unwrapped_model: Any,
    batch: dict[str, Any],
    config: dict[str, Any],
    accelerator: Any,
    bridge_enabled: bool,
    step: int,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    if unwrapped_model.progress_state_planner is None:
        raise ValueError("Stage1 trajectory training requires progress_state_planner")

    device = accelerator.device
    dtype = torch.bfloat16
    batch_size = int(batch["batch_size"])
    progress_state = unwrapped_model.progress_state_planner.initial_state(batch_size, device=device, dtype=dtype)
    loss_terms = []
    action_loss_values = []
    loss_rows = 0
    last_tensors: dict[str, Any] = {}

    for step_batch in batch["trajectory_steps"]:
        batch_indices = step_batch["batch_indices"].to(device=device)
        loss_mask = step_batch["loss_mask"].to(device=device).bool()
        active_progress_state = _slice_progress_state(progress_state, batch_indices)

        states = step_batch["states"].to(device=device, dtype=dtype)
        actions_gt = step_batch["actions"].to(device=device, dtype=dtype)
        action_mask = step_batch["action_mask"].to(device=device)
        fused_tokens = step_batch["fused_tokens"].to(device=device, dtype=dtype)
        raw_hidden_states = step_batch.get("vlm_hidden_states")
        hidden_states = None
        if raw_hidden_states is not None:
            hidden_states = [hidden_state.to(device=device, dtype=dtype) for hidden_state in raw_hidden_states]
        memory_context = _optional_batch_tensor(torch, step_batch, "memory_context", device, dtype)
        memory_context_mask = _optional_batch_tensor(torch, step_batch, "memory_context_mask", device, None)
        short_memory_time_ids = _optional_batch_tensor(torch, step_batch, "short_memory_time_ids", device, None)
        executed_actions = _optional_batch_tensor(torch, step_batch, "executed_actions", device, dtype)
        executed_action_mask = _optional_batch_tensor(torch, step_batch, "executed_action_mask", device, None)
        planner_vl_summary = _optional_batch_tensor(torch, step_batch, "planner_vl_summary", device, dtype)
        plan_token_mask = _optional_batch_tensor(torch, step_batch, "plan_token_mask", device, None)

        context = nullcontext() if bool(loss_mask.any().item()) else torch.no_grad()
        with context, _get_autocast_context(torch, device):
            pred_velocity, noise = model(
                fused_tokens,
                state=states,
                actions_gt=actions_gt,
                action_mask=action_mask,
                hidden_states=hidden_states if bridge_enabled else None,
                memory_context=memory_context,
                memory_context_mask=memory_context_mask,
                short_memory_time_ids=short_memory_time_ids,
                executed_actions=executed_actions,
                executed_action_mask=executed_action_mask,
                progress_state=active_progress_state,
                planner_vl_summary=planner_vl_summary,
                plan_token_mask=plan_token_mask,
            )

        planner_output = unwrapped_model.last_progress_planner_output
        if planner_output is None:
            raise RuntimeError("progress_state_planner did not produce an output during Stage1 trajectory training")
        progress_state = _scatter_progress_state(progress_state, batch_indices, planner_output.progress_state)

        if bool(loss_mask.any().item()):
            if action_mask[loss_mask].sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0 for a trajectory loss slice")
            action_loss = stage1_flow_matching_loss(
                pred_velocity=pred_velocity[loss_mask],
                noise=noise[loss_mask],
                actions_gt=actions_gt[loss_mask],
                action_mask=action_mask[loss_mask],
            )
            loss_terms.append(action_loss)
            action_loss_values.append(float(action_loss.detach().cpu().item()))
            loss_rows += int(loss_mask.sum().item())

        last_tensors = {
            "states": states.detach(),
            "actions_gt": actions_gt.detach(),
            "fused_tokens": fused_tokens.detach(),
            "pred_velocity": pred_velocity.detach(),
        }

    if not loss_terms:
        raise ValueError("Stage1 trajectory window batch produced no loss terms")
    loss = torch.stack(loss_terms).mean()
    extra_metrics = {
        "action_loss": float(loss.detach().cpu().item()),
        "trajectory_loss_steps": float(len(loss_terms)),
        "trajectory_loss_rows": float(loss_rows),
    }
    if action_loss_values:
        extra_metrics["action_loss_step_mean"] = float(sum(action_loss_values) / len(action_loss_values))
    return loss, extra_metrics, last_tensors


def _check_numerical_stability(torch: Any, step: int, **named_tensors: Any) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info("[Stage1 step %s] Non-finite detected in %s", step, name)
            return False
    return True


def _log_training_step(
    step: int,
    loss: Any,
    total_norm: Any,
    clipped_norm: Any,
    scheduler: Any,
    dataloader: Any,
    accelerator: Any,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    if not accelerator.is_main_process:
        return
    current_epoch = step / len(dataloader)
    logging.info("Estimated Stage1 epoch: %.2f", current_epoch)
    logging.info("[Stage1 step %s] loss=%.4f lr=%s", step, float(loss.item()), scheduler.get_last_lr()[0])
    logging.info("[Stage1 step %s] grad_norm=%s clipped_norm=%s", step, total_norm, clipped_norm)
    for name, value in sorted((extra_metrics or {}).items()):
        logging.info("[Stage1 step %s] %s=%.4f", step, name, value)
