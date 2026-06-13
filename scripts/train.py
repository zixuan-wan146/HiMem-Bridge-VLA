import sys
import os
import math
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wandb
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
import logging
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW
from himem_bridge_vla.training_config import validate_training_config
from himem_bridge_vla.dataset.config_utils import resolve_dataset_config_paths
from himem_bridge_vla.experiment_config import resolve_experiment_config
from himem_bridge_vla.reproducibility import set_global_seed, write_experiment_snapshot

import warnings

accelerator = Accelerator()

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
    for optional_key in ("boundary", "progress", "skill_id"):
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

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def init_swanlab(config: dict, accelerator: Accelerator):

    if accelerator is None or accelerator.is_main_process:
        swanlab.init(
            project=config.get("wandb_project", "default_run"),
            name=config.get("run_name", "default_run"),
            config=config
        )

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

        with open(config.get("dataset_config_path"), 'r') as f:
            dataset_config = yaml.safe_load(f)
        dataset_config_base_dir = Path(
            get_with_warning(config, "dataset_config_base_dir", str(PROJECT_ROOT))
        ).expanduser().resolve()
        dataset_config = resolve_dataset_config_paths(dataset_config, dataset_config_base_dir)

        dataset = SimulationDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            binarize_gripper=binarize_gripper,
            cache_dir=config.get("cache_dir"),
            use_augmentation=use_augmentation
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(
            "Loaded %s samples using %s (%s), dataset paths resolved from %s",
            len(dataset),
            config.get("dataset_config_path"),
            dataset_type,
            dataset_config_base_dir,
        )
    return dataset


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check dataset_config_path and source data paths.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
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

def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        wandb.log({
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            
        })
        swanlab.log({
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
    
        })

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, config=None, norm_stats=None):
    if not hasattr(model_engine, "save_checkpoint"):
        raise RuntimeError(
            "Checkpoint saving requires a DeepSpeed-prepared model. "
            "Launch with accelerate and a DeepSpeed config, as shown in README.md."
        )
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        "step": step,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
    } if accelerator.is_main_process else {} 

    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    
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
        return client_state.get("step", 0), client_state
        
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
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    

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
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config):
    config = resolve_experiment_config(config)
    validate_training_config(config)

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
    
    # === loss function ===
    loss_fn = nn.MSELoss() 

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
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head,
        }
        if model.bridge_adapter is not None:
            modules_to_inspect["bridge_adapter"] = model.bridge_adapter
        if model.memory_writer is not None:
            modules_to_inspect["memory_writer"] = model.memory_writer
        inspect_named_submodules(modules_to_inspect)

    # === Training Loop ===
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            embodiment_ids = batch["embodiment_ids"]
            fused_tokens_list = []
            hidden_states_per_sample = []
            
            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                embedding = model.get_vl_embeddings(
                    images=images,
                    image_mask=image_mask,
                    prompt=prompt,
                    return_cls_only=False,
                    return_hidden_states=bridge_enabled,
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

            with get_autocast_context(accelerator.device):

                pred_velocity, noise = model(
                    fused_tokens,
                    state=states,
                    actions_gt=actions_gt,
                    action_mask=action_mask,
                    embodiment_ids=embodiment_ids,
                    hidden_states=hidden_states,
                )
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            if pred_velocity.shape != target_velocity.shape:
                raise ValueError(f"pred_velocity shape {pred_velocity.shape} != target_velocity shape {target_velocity.shape}")

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")
            

            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask
            loss = loss_fn(pred_velocity_mask, target_velocity)
            scale_factor = action_mask.numel() / (action_mask.sum() + 1e-8)
            loss = loss * scale_factor
            
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
                log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator)
   
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
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict 
                )
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
         
    # === Save final model ===
    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=loss, accelerator=accelerator, config=config, norm_stats=dataset.arm2stats_dict)
    logging.info("Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train HiMem-Bridge-VLA")

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--bridge_himem_config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="simulation")
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument(
        "--dataset_config_base_dir",
        type=str,
        default=str(REPO_ROOT),
        help="Base directory for relative paths inside dataset_config_path. Defaults to the repository root.",
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)
