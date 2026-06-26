from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.training_config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)
from himem_bridge_vla.dataset.config_utils import resolve_dataset_config_paths
from himem_bridge_vla.experiment_config import resolve_experiment_config
from himem_bridge_vla.path_utils import display_project_path, normalize_project_relative_path, project_path
from himem_bridge_vla.reproducibility import (
    build_torch_generator,
    seed_data_worker,
    set_global_seed,
    write_experiment_snapshot,
)
from himem_bridge_vla.training_loss import (
    boundary_bce_loss,
    coarse_planner_intent_loss,
    masked_flow_matching_mse,
    progress_smooth_l1_loss,
)

torch: Any = None
DataLoader: Any = None
tqdm: Any = None
LambdaLR: Any = None
HiMemBridgeVLA: Any = None
Accelerator: Any = None
DistributedType: Any = None
AdamW: Any = None
accelerator: Any = None
wandb: Any = None
swanlab: Any = None
WANDB_ACTIVE = False
SWANLAB_ACTIVE = False
_RUNTIME_LOADED = False


def _ensure_training_runtime() -> None:
    global torch, DataLoader, tqdm, LambdaLR, HiMemBridgeVLA, Accelerator, DistributedType, AdamW
    global accelerator, wandb, swanlab, _RUNTIME_LOADED

    if _RUNTIME_LOADED:
        return

    try:
        import torch as torch_module
        from accelerate import Accelerator as AcceleratorClass
        from accelerate import DistributedType as DistributedTypeClass
        from torch.optim import AdamW as AdamWClass
        from torch.optim.lr_scheduler import LambdaLR as LambdaLRClass
        from torch.utils.data import DataLoader as DataLoaderClass
        from tqdm import tqdm as tqdm_function

        from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA as HiMemBridgeVLAClass
    except ModuleNotFoundError as exc:
        missing = exc.name or "a training runtime dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for training. Install the runtime dependencies from requirements.txt "
            "or run this script in the prepared training environment."
        ) from exc

    torch = torch_module
    DataLoader = DataLoaderClass
    tqdm = tqdm_function
    LambdaLR = LambdaLRClass
    HiMemBridgeVLA = HiMemBridgeVLAClass
    Accelerator = AcceleratorClass
    DistributedType = DistributedTypeClass
    AdamW = AdamWClass
    accelerator = AcceleratorClass()

    try:
        import wandb as wandb_module
    except ModuleNotFoundError:
        wandb_module = None
    try:
        import swanlab as swanlab_module
    except ModuleNotFoundError:
        swanlab_module = None

    wandb = wandb_module
    swanlab = swanlab_module
    _RUNTIME_LOADED = True

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


def get_autocast_context(device):
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


def unwrap_training_model(model):
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(model)
    return getattr(model, "module", model)


def validate_batch_image_masks(image_masks, step: int) -> None:
    flat_masks = image_masks.reshape(image_masks.shape[0], -1)
    empty_indices = torch.where(flat_masks.sum(dim=1) == 0)[0]
    if empty_indices.numel() > 0:
        raise ValueError(
            f"[Step {step}] image_mask has no active image for batch indices "
            f"{empty_indices.detach().cpu().tolist()}"
        )


def _optional_batch_tensor(batch: dict, key: str, device: torch.device, dtype: torch.dtype | None):
    value = batch.get(key)
    if value is None:
        return None
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def encode_batch_embeddings(model, prompts, images_batch, image_masks, *, return_hidden_states: bool):
    fused_tokens_list = []
    hidden_states_per_sample = []

    for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
        embedding = model.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=False,
            return_hidden_states=return_hidden_states,
        )
        if hasattr(embedding, "fused_tokens"):
            fused_tokens_list.append(embedding.fused_tokens.to(dtype=torch.bfloat16))
            hidden_states_per_sample.append(
                [hidden_state.to(dtype=torch.bfloat16) for hidden_state in embedding.hidden_states]
            )
        else:
            fused_tokens_list.append(embedding.to(dtype=torch.bfloat16))

    fused_tokens = torch.cat(fused_tokens_list, dim=0)
    hidden_states = None
    if hidden_states_per_sample:
        layer_count = len(hidden_states_per_sample[0])
        if any(len(sample) != layer_count for sample in hidden_states_per_sample):
            raise ValueError("All samples must return the same number of selected hidden-state layers")
        hidden_states = [
            torch.cat([sample[layer_index] for sample in hidden_states_per_sample], dim=0)
            for layer_index in range(layer_count)
        ]
    return fused_tokens, hidden_states


def compute_bridge_auxiliary_loss(model, batch: dict, config: dict) -> tuple[Any, dict[str, float]]:
    bridge_output = getattr(unwrap_training_model(model), "last_bridge_output", None)
    if bridge_output is None:
        return None, {}

    aux_loss = None
    metrics: dict[str, float] = {}
    boundary_weight = float(config.get("boundary_loss_weight", 1.0))
    progress_weight = float(config.get("progress_loss_weight", 0.2))

    if boundary_weight > 0.0 and "boundary" in batch:
        boundary_loss = boundary_bce_loss(bridge_output.boundary_logits, batch["boundary"])
        aux_loss = boundary_weight * boundary_loss if aux_loss is None else aux_loss + boundary_weight * boundary_loss
        metrics["boundary_loss"] = float(boundary_loss.detach().cpu().item())

    if progress_weight > 0.0 and "progress" in batch and hasattr(bridge_output, "progress_logits"):
        progress_loss = progress_smooth_l1_loss(bridge_output.progress_logits, batch["progress"])
        aux_loss = progress_weight * progress_loss if aux_loss is None else aux_loss + progress_weight * progress_loss
        metrics["progress_loss"] = float(progress_loss.detach().cpu().item())

    return aux_loss, metrics


def load_action_segment_autoencoder(checkpoint_path: str | None, *, device: Any):
    if not checkpoint_path:
        return None
    from himem_bridge_vla.model.planner import ActionSegmentAutoencoder, ActionSegmentAutoencoderConfig

    checkpoint_file = Path(str(checkpoint_path)).expanduser()
    if not checkpoint_file.is_absolute():
        checkpoint_file = project_path(checkpoint_file, REPO_ROOT)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    raw_config = checkpoint.get("segment_autoencoder_config") or checkpoint.get("autoencoder_config")
    if raw_config is None:
        raise KeyError(f"segment autoencoder checkpoint lacks config: {checkpoint_path}")
    state_dict = checkpoint.get("segment_autoencoder_state_dict") or checkpoint.get("model")
    if state_dict is None:
        raise KeyError(f"segment autoencoder checkpoint lacks weights: {checkpoint_path}")
    autoencoder = ActionSegmentAutoencoder(ActionSegmentAutoencoderConfig(**raw_config)).to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    return autoencoder


def compute_coarse_planner_loss(model, batch: dict, config: dict, segment_autoencoder=None) -> tuple[Any, dict[str, float]]:
    planner_output = getattr(unwrap_training_model(model), "last_coarse_planner_output", None)
    if planner_output is None or segment_autoencoder is None:
        return None, {}
    if "action_segments" not in batch or "action_segment_mask" not in batch:
        return None, {}

    action_segments = batch["action_segments"].to(
        device=planner_output.predicted_latents.device,
        dtype=planner_output.predicted_latents.dtype,
    )
    segment_mask = batch["action_segment_mask"].to(device=planner_output.predicted_latents.device)
    torch_module = torch
    if torch_module is None:
        import torch as torch_module
    with torch_module.no_grad():
        target_latents = segment_autoencoder.encode(action_segments)
    decoded_segments = segment_autoencoder.decode(planner_output.predicted_latents)
    planner_loss = coarse_planner_intent_loss(
        planner_output.predicted_latents,
        target_latents,
        decoded_segments,
        action_segments,
        segment_mask,
        latent_loss_weight=float(config.get("coarse_planner_latent_loss_weight", 1.0)),
        chunk_loss_weight=float(config.get("coarse_planner_chunk_loss_weight", 1.0)),
        gripper_indices=config.get("coarse_planner_gripper_indices", [-1]),
        gripper_loss_weight=float(config.get("coarse_planner_gripper_loss_weight", 2.0)),
    )
    loss_weight = float(config.get("coarse_planner_loss_weight", 0.2))
    weighted_loss = loss_weight * planner_loss
    return weighted_loss, {
        "coarse_planner_loss": float(planner_loss.detach().cpu().item()),
        "coarse_planner_loss_weighted": float(weighted_loss.detach().cpu().item()),
    }


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    batch_dict = {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }
    if all("planner_prompt" in item for item in batch):
        batch_dict["planner_prompts"] = [item["planner_prompt"] for item in batch]
    if all("planner_images" in item for item in batch):
        batch_dict["planner_images"] = [item["planner_images"] for item in batch]
    if all("planner_image_mask" in item for item in batch):
        batch_dict["planner_image_masks"] = torch.stack([item["planner_image_mask"] for item in batch], dim=0)
    if all("planner_state" in item for item in batch):
        batch_dict["planner_states"] = torch.stack([item["planner_state"] for item in batch], dim=0)
    for optional_key in ("boundary", "progress", "skill_id"):
        if all(optional_key in item for item in batch):
            batch_dict[optional_key] = torch.stack([item[optional_key] for item in batch], dim=0)
    for optional_key in ("action_segments", "action_segment_mask"):
        if all(optional_key in item for item in batch):
            batch_dict[optional_key] = torch.stack([item[optional_key] for item in batch], dim=0)
    for optional_key in (
        "memory_context",
        "memory_context_mask",
        "short_memory_time_ids",
        "executed_actions",
        "executed_action_mask",
        "planner_vl_summary",
        "plan_token_mask",
    ):
        if all(optional_key in item for item in batch):
            batch_dict[optional_key] = torch.stack([item[optional_key] for item in batch], dim=0)
    for optional_key in ("episode_id", "frame_index", "global_frame_index", "segment_id", "segment_start", "segment_end"):
        if all(optional_key in item for item in batch):
            batch_dict[optional_key] = [item[optional_key] for item in batch]
    return batch_dict

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: dict, accelerator: Accelerator):
    global WANDB_ACTIVE

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            logging.info("WandB logging disabled by config.")
            WANDB_ACTIVE = False
            return

        if wandb is None:
            logging.warning("wandb is not installed; skipping WandB logging.")
            WANDB_ACTIVE = False
            return

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
        WANDB_ACTIVE = True

def init_swanlab(config: dict, accelerator: Accelerator):
    global SWANLAB_ACTIVE

    if accelerator is None or accelerator.is_main_process:
        if get_with_warning(config, "disable_swanlab", False):
            logging.info("SwanLab logging disabled by config.")
            SWANLAB_ACTIVE = False
            return
        if swanlab is None:
            logging.warning("swanlab is not installed; skipping SwanLab logging.")
            SWANLAB_ACTIVE = False
            return
        swanlab.init(
            project=config.get("wandb_project", "default_run"),
            name=config.get("run_name", "default_run"),
            config=config
        )
        SWANLAB_ACTIVE = True

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_type = get_with_warning(config, "dataset_type", "simulation")
    image_size = get_with_warning(config, "image_size", 448)
    max_samples = get_with_warning(config, "max_samples_per_file", None)
    horizon = get_with_warning(config, "horizon", 50)
    binarize_gripper = get_with_warning(config, "binarize_gripper", False)
    use_augmentation = get_with_warning(config, "use_augmentation", False)
    if dataset_type == "simulation":
        import yaml

        from himem_bridge_vla.dataset.simulation_dataset import SimulationDataset

        dataset_config_path = project_path(
            config.get("dataset_config_path"),
            REPO_ROOT,
            label="--dataset_config_path",
        )
        with dataset_config_path.open("r", encoding="utf-8") as f:
            dataset_config = yaml.safe_load(f)
        dataset_config_base_dir = project_path(
            get_with_warning(config, "dataset_config_base_dir", "."),
            REPO_ROOT,
            label="--dataset_config_base_dir",
        )
        dataset_config = resolve_dataset_config_paths(dataset_config, dataset_config_base_dir)

        dataset = SimulationDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            binarize_gripper=binarize_gripper,
            cache_dir=config.get("cache_dir"),
            use_augmentation=use_augmentation,
            action_segment_config=build_action_segment_dataset_config(config),
        )
    elif dataset_type == "memory_token_cache":
        from himem_bridge_vla.dataset import MemoryTokenCacheDataset

        dataset_config_path = project_path(
            config.get("dataset_config_path"),
            REPO_ROOT,
            label="--dataset_config_path",
        )
        dataset_config_base_dir = dataset_config_path.parent
        dataset = MemoryTokenCacheDataset(dataset_config_path, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(
            "Loaded %s samples using %s (%s), dataset paths resolved from %s",
            len(dataset),
            config.get("dataset_config_path"),
            dataset_type,
            display_project_path(dataset_config_base_dir, REPO_ROOT),
        )
    return dataset


def build_action_segment_dataset_config(config: dict) -> dict | None:
    if not bool(config.get("coarse_planner_enabled", False)):
        return None
    return {
        "enabled": True,
        "num_plan_steps": int(config.get("coarse_planner_num_plan_steps", 1)),
        "planning_horizon": int(config.get("coarse_planner_planning_horizon", 32)),
        "action_dim": int(config.get("coarse_planner_action_dim", config.get("per_action_dim", 7))),
    }


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)
    seed = int(get_with_warning(config, "seed", 42))
    dataset_type = get_with_warning(config, "dataset_type", "simulation")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check dataset_config_path and source data paths.")

    collate_fn = custom_collate_fn
    if dataset_type == "memory_token_cache":
        from functools import partial

        from himem_bridge_vla.dataset import collate_direct_bridge_token_cache_samples

        collate_fn = partial(
            collate_direct_bridge_token_cache_samples,
            memory_entry_tokens=int(config.get("memory_entry_tokens", 16)),
            action_horizon=int(config.get("horizon", 32)),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(seed),
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Dataloader has no batches. Dataset size={len(dataset)}, batch_size={batch_size}, drop_last=True."
        )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, extra_metrics=None):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        metrics = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if extra_metrics:
            metrics.update(extra_metrics)
            for metric_name, metric_value in sorted(extra_metrics.items()):
                logging.info(f"[Step {step}] {metric_name}: {metric_value:.4f}")
        if WANDB_ACTIVE:
            wandb.log(metrics)
        if SWANLAB_ACTIVE:
            swanlab.log(metrics)

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None, tag=None):
    if not hasattr(model_engine, "save_checkpoint"):
        raise RuntimeError(
            "Checkpoint saving requires a DeepSpeed-prepared model. "
            "Launch with accelerate and a DeepSpeed config, as shown in README.md."
        )
    checkpoint_tag = tag or f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, checkpoint_tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        "step": step,
        "checkpoint_tag": checkpoint_tag,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
    } if accelerator.is_main_process else {} 

    model_engine.save_checkpoint(save_dir, tag=checkpoint_tag, client_state=client_state)
    
    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2)
                
        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):
    if not hasattr(model_engine, "load_checkpoint"):
        raise RuntimeError(
            "Checkpoint resume requires a DeepSpeed-prepared model. "
            "Launch with accelerate and a DeepSpeed config, as shown in README.md."
        )

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return _client_state_step(client_state, accelerator), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return _client_state_step(client_state, accelerator), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")


def _client_state_step(client_state, accelerator) -> int:
    raw_step = client_state.get("step", 0)
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        if accelerator is None or accelerator.is_main_process:
            logging.warning("Checkpoint client_state step %r is not numeric; resuming scheduler from step 0.", raw_step)
        return 0


def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    if not decay and not no_decay:
        raise ValueError("No trainable parameters found. Check finetune flags and Bridge-HiMem config.")
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config):
    _ensure_training_runtime()

    config = resolve_experiment_config(config)
    validate_training_config(config, repo_root=REPO_ROOT)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)

    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    setup_logging(save_dir)
    if accelerator is None or accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
        logging.info("Resolved experiment config written to %s", save_dir)
        logging.info("Seed=%s deterministic=%s", seed, deterministic)
    
    # === WandB and Swanlab ===
    init_wandb(config, accelerator)
    init_swanlab(config, accelerator)

    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # === Model ===
    model = HiMemBridgeVLA(config)
    config = model.config
    bridge_enabled = bool(model.use_bridge)
    if accelerator is None or accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
    model.train()
    model.set_finetune_flags()

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}")


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
    unwrapped_model = unwrap_training_model(model)
    if not hasattr(model_engine, "save_checkpoint"):
        raise RuntimeError(
            "This training script currently expects DeepSpeed checkpoint support. "
            "Use `accelerate launch --deepspeed_config_file configs/deepspeed/ds_config.json ...`."
        )
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    
    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)

    # === Resume training from checkpoint ===
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)

    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)

        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        
        modules_to_inspect = {
            "vision_model": unwrapped_model.embedder.model.vision_model,
            "language_model": unwrapped_model.embedder.model.language_model,
            "action_head": unwrapped_model.action_head,
        }
        if unwrapped_model.bridge_adapter is not None:
            modules_to_inspect["bridge_adapter"] = unwrapped_model.bridge_adapter
        if unwrapped_model.coarse_planner is not None:
            modules_to_inspect["coarse_planner"] = unwrapped_model.coarse_planner
        inspect_named_submodules(modules_to_inspect)

    segment_autoencoder = load_action_segment_autoencoder(
        config.get("coarse_planner_segment_autoencoder_checkpoint"),
        device=accelerator.device,
    )

    # === Training Loop ===
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            states = batch["states"].to(device=accelerator.device, dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(device=accelerator.device, dtype=torch.bfloat16)
            action_mask = batch["action_mask"].to(device=accelerator.device)
            embodiment_ids = batch.get("embodiment_ids")
            if embodiment_ids is not None:
                embodiment_ids = embodiment_ids.to(device=accelerator.device)

            if "fused_tokens" in batch:
                fused_tokens = batch["fused_tokens"].to(device=accelerator.device, dtype=torch.bfloat16)
                raw_hidden_states = batch.get("vlm_hidden_states")
                hidden_states = None
                if raw_hidden_states is not None:
                    hidden_states = [
                        hidden_state.to(device=accelerator.device, dtype=torch.bfloat16)
                        for hidden_state in raw_hidden_states
                    ]
            else:
                prompts = batch["prompts"]
                images_batch = batch["images"]
                image_masks = batch["image_masks"]
                if embodiment_ids is None:
                    raise ValueError("image training batches must include embodiment_ids")
                validate_batch_image_masks(image_masks, step)
                fused_tokens, hidden_states = encode_batch_embeddings(
                    model,
                    prompts,
                    images_batch,
                    image_masks,
                    return_hidden_states=bridge_enabled,
                )
            planner_fused_tokens = None
            planner_states = None
            memory_context = _optional_batch_tensor(batch, "memory_context", accelerator.device, torch.bfloat16)
            memory_context_mask = _optional_batch_tensor(batch, "memory_context_mask", accelerator.device, None)
            short_memory_time_ids = _optional_batch_tensor(batch, "short_memory_time_ids", accelerator.device, None)
            executed_actions = _optional_batch_tensor(batch, "executed_actions", accelerator.device, torch.bfloat16)
            executed_action_mask = _optional_batch_tensor(batch, "executed_action_mask", accelerator.device, None)
            planner_vl_summary = _optional_batch_tensor(batch, "planner_vl_summary", accelerator.device, torch.bfloat16)
            plan_token_mask = _optional_batch_tensor(batch, "plan_token_mask", accelerator.device, None)
            if "planner_images" in batch:
                planner_image_masks = batch.get("planner_image_masks")
                if planner_image_masks is None:
                    raise ValueError("planner_images requires planner_image_masks")
                validate_batch_image_masks(planner_image_masks, step)
                planner_fused_tokens, _ = encode_batch_embeddings(
                    model,
                    batch.get("planner_prompts", prompts),
                    batch["planner_images"],
                    planner_image_masks,
                    return_hidden_states=False,
                )
            if "planner_states" in batch:
                planner_states = batch["planner_states"].to(device=accelerator.device, dtype=torch.bfloat16)

            with get_autocast_context(accelerator.device):

                pred_velocity, noise = model(
                    fused_tokens,
                    state=states,
                    actions_gt=actions_gt,
                    action_mask=action_mask,
                    embodiment_ids=embodiment_ids,
                    hidden_states=hidden_states,
                    memory_context=memory_context,
                    memory_context_mask=memory_context_mask,
                    short_memory_time_ids=short_memory_time_ids,
                    executed_actions=executed_actions,
                    executed_action_mask=executed_action_mask,
                    planner_vl_summary=planner_vl_summary,
                    planner_fused_tokens=planner_fused_tokens,
                    planner_state=planner_states,
                    plan_token_mask=plan_token_mask,
                )
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            if pred_velocity.shape != target_velocity.shape:
                raise ValueError(f"pred_velocity shape {pred_velocity.shape} != target_velocity shape {target_velocity.shape}")

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")
            

            action_loss = masked_flow_matching_mse(pred_velocity, target_velocity, action_mask)
            loss = action_loss
            extra_metrics = {"action_loss": float(action_loss.detach().cpu().item())}
            bridge_aux_loss, bridge_metrics = compute_bridge_auxiliary_loss(model, batch, config)
            if bridge_aux_loss is not None:
                loss = loss + bridge_aux_loss
                extra_metrics.update(bridge_metrics)
                extra_metrics["bridge_aux_loss"] = float(bridge_aux_loss.detach().cpu().item())
            coarse_planner_loss, coarse_planner_metrics = compute_coarse_planner_loss(
                model,
                batch,
                config,
                segment_autoencoder=segment_autoencoder,
            )
            if coarse_planner_loss is not None:
                loss = loss + coarse_planner_loss
                extra_metrics.update(coarse_planner_metrics)
            
            # === NaN/Inf check ===
            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss
            ):
                raise FloatingPointError(f"Non-finite tensor detected at step {step}")

            # === Backward and optimizer step ===
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            # === Clip grad norm ===
            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)

            optimizer.step()
            scheduler.step()
            
            # === Logging ===
            if step % log_interval == 0:
                log_training_step(
                    step,
                    loss,
                    total_norm,
                    clipped_norm,
                    scheduler,
                    dataloader,
                    accelerator,
                    extra_metrics=extra_metrics,
                )
   
            # === Save best checkpoint ===
            loss_value = loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)
            
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)
            
            if is_best_tensor.item() == 1 and step > 1000:
                if accelerator.is_main_process:
                    logging.info("Saving best checkpoint")
                save_checkpoint(
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                    tag="step_best",
                )
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
         
    # === Save final model ===
    save_checkpoint(
        save_dir,
        step=step,
        model_engine=model_engine,
        loss=loss,
        accelerator=accelerator,
        config=config,
        norm_stats=dataset.arm2stats_dict,
        tag="step_final",
    )
    logging.info("Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HiMem-Bridge-VLA")
    parser.add_argument("--config", type=str, default=None, help="Optional checked-in YAML training profile.")

    # Basic config
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--vlm_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--action_head", type=str, choices=["flowmatching"], default=argparse.SUPPRESS)
    parser.add_argument("--bridge_himem_config", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--return_cls_only", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument(
        "--disable_wandb",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--disable_swanlab",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Disable SwanLab logging.",
    )

    # Dataset
    parser.add_argument("--dataset_type", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset_config_base_dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Base directory for relative paths inside dataset_config_path. Defaults to the repository root.",
    )
    parser.add_argument("--cache_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--image_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--binarize_gripper",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Whether to binarize gripper state/action.",
    )
    parser.add_argument(
        "--use_augmentation",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable data augmentation on images",
    )

    # Training
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--grad_clip_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--boundary_loss_weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--progress_loss_weight", type=float, default=argparse.SUPPRESS)

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save_dir", type=str, default=argparse.SUPPRESS)

    # Resume
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--resume_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--resume_pretrain", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    # Finetuning
    parser.add_argument("--finetune_vlm", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_action_head", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_coarse_planner", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_progress_planner", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    # Misc
    parser.add_argument("--progress_planner_enabled", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_checkpoint", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_replan_stride", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--per_action_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--state_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--horizon", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--action_head_ffn_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_plan_slots", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--visual_gate_lambda", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--plan_gate_lambda", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--short_memory_time_bins", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_vlm_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_inference_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    return parser


def build_training_config(args: argparse.Namespace) -> dict:
    cli_overrides = vars(args).copy()
    config_path = cli_overrides.pop("config", None)
    if config_path:
        config_candidate = project_path(config_path, REPO_ROOT, label="--config")
        file_config = load_training_config(config_candidate)
        file_config["training_config_path"] = normalize_project_relative_path(
            config_candidate,
            REPO_ROOT,
            label="--config",
        )
    else:
        file_config = {}

    config = merge_training_config(
        default_training_config(REPO_ROOT),
        file_config=file_config,
        cli_overrides=cli_overrides,
    )
    config["repo_root"] = "."
    return resolve_training_config_paths(config, REPO_ROOT)


def main() -> int:
    os.chdir(REPO_ROOT)
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_training_config(args)
    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator is None or accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
