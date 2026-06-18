import argparse
import asyncio
from contextlib import nullcontext
import json
import logging
import sys
from pathlib import Path

import torch
import websockets
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from himem_bridge_vla.runtime_config import (
    DEFAULT_MAX_MESSAGE_SIZE,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    IMAGE_SIZE,
    TARGET_STATE_DIM,
)
from himem_bridge_vla.image_preprocessing import rgb_array_to_pil
from himem_bridge_vla.server_protocol import checkpoint_normalizer_dim, validate_inference_request
from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.transition_trigger_manager import ServerTransitionTriggerManager
from himem_bridge_vla.utils.normalization import NormalizationStats


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
    return resolved


def load_checkpoint_payload(ckpt_path: Path, *, allow_unsafe_checkpoint_load: bool):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe_checkpoint_load:
            raise RuntimeError(
                "Checkpoint could not be loaded with torch.load(weights_only=True). "
                "Only enable unsafe pickle loading for trusted local DeepSpeed checkpoints."
            ) from exc
        logging.warning(
            "Falling back to torch.load(weights_only=False). Only use this with trusted local checkpoints. "
            "Original safe-load error: %s",
            exc,
        )
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def load_model_and_normalizer(
    ckpt_dir,
    device: str = "cuda",
    inference_steps: int = 32,
    allow_unsafe_checkpoint_load: bool = False,
):
    device = resolve_device(device)
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    stats_path = ckpt_dir / "norm_stats.json"
    ckpt_path = ckpt_dir / "mp_rank_00_model_states.pt"

    for path in (config_path, stats_path, ckpt_path):
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    config["device"] = str(device)
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = inference_steps

    model = HiMemBridgeVLA(config).eval()
    checkpoint = load_checkpoint_payload(
        ckpt_path,
        allow_unsafe_checkpoint_load=allow_unsafe_checkpoint_load,
    )
    state_dict = checkpoint["module"] if "module" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    normalizer_dim = checkpoint_normalizer_dim(config)
    normalizer = NormalizationStats(stats, target_dim=normalizer_dim)
    logging.info("Loaded normalization stats robot_keys=%s default_robot_key=%s", normalizer.robot_keys, normalizer.robot_key)
    return model, normalizer


def decode_image_from_list(img_list, device) -> torch.Tensor:
    pil = rgb_array_to_pil(img_list, IMAGE_SIZE)
    return transforms.ToTensor()(pil).to(device)


def pad_state_tensor(state: torch.Tensor, target_dim: int = TARGET_STATE_DIM) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] > target_dim:
        raise ValueError(f"State dimension {state.shape[1]} exceeds expected {target_dim}")
    if state.shape[1] < target_dim:
        padding = torch.zeros((state.shape[0], target_dim - state.shape[1]), device=state.device)
        state = torch.cat([state, padding], dim=1)
    return state


def infer_from_json_dict(data: dict, model, normalizer, transition_manager: ServerTransitionTriggerManager | None = None):
    device = next(model.parameters()).device
    model_state_dim = int(model.config.get("state_dim", TARGET_STATE_DIM))
    model_action_dim = int(getattr(model, "per_action_dim", model.config.get("per_action_dim", TARGET_STATE_DIM)))
    request = validate_inference_request(
        data,
        target_state_dim=model_state_dim,
        target_action_dim=model_action_dim,
        max_action_mask_dim=TARGET_STATE_DIM,
    )
    transition_result, memory_write_gate = update_transition_trigger(request, transition_manager)
    coarse_plan_refresh = bool(request["reset_transition_trigger"])
    if transition_result is not None:
        coarse_plan_refresh = coarse_plan_refresh or bool(transition_result.should_plan)

    images = [decode_image_from_list(img, device) for img in request["image"]]
    for img in images:
        expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(img.shape) != expected_shape:
            raise ValueError(f"image_size must be {expected_shape}, got {tuple(img.shape)}")

    state = torch.tensor(request["state"], dtype=torch.float32, device=device)
    robot_key = request["robot_key"]
    norm_state = normalizer.normalize_state(
        pad_state_tensor(state, target_dim=model_state_dim),
        robot_key=robot_key,
    ).to(dtype=torch.float32)

    prompt = request["prompt"]
    image_mask = torch.tensor(request["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([request["action_mask"]], dtype=torch.int32, device=device)

    autocast_context = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
            episode_id=request["episode_id"],
            session_id=request["session_id"],
            reset_memory=request["reset_memory"],
            memory_write_gate=memory_write_gate,
            coarse_plan_refresh=coarse_plan_refresh,
        )
        if action.numel() % model_action_dim != 0:
            raise ValueError(f"Model returned {action.numel()} action values, not divisible by per_action_dim={model_action_dim}")
        action = action.reshape(1, -1, model_action_dim)
        action = normalizer.denormalize_action(action[0], robot_key=robot_key)
        actions = action.cpu().numpy().tolist()
        if not request["return_debug"]:
            return actions
        response = {"actions": actions}
        if transition_result is not None:
            response["transition_trigger"] = transition_result.to_dict()
        return response


def update_transition_trigger(
    request: dict,
    transition_manager: ServerTransitionTriggerManager | None,
):
    if transition_manager is None:
        return None, None

    episode_key = transition_episode_key(request["episode_id"], request["session_id"])
    if request["transition_frame"] is None:
        if request["reset_transition_trigger"] and episode_key:
            transition_manager.reset(episode_key)
        return None, None

    result = transition_manager.update(
        episode_key=episode_key,
        dataset_name=request["transition_dataset_name"],
        frame=request["transition_frame"],
        frame_index=request["transition_frame_index"],
        reset=request["reset_transition_trigger"],
    )
    memory_write_gate = 1.0 if result.memory_write else 0.0
    return result, memory_write_gate


def transition_episode_key(episode_id: str | None, session_id: str | None) -> str | None:
    if episode_id and session_id:
        return f"{session_id}:{episode_id}"
    return episode_id or session_id


async def handle_request(websocket, model, normalizer, transition_manager: ServerTransitionTriggerManager | None = None):
    logging.info("Client connected")
    try:
        async for message in websocket:
            try:
                json_data = json.loads(message)
                logging.info("Received JSON observation")
                actions = infer_from_json_dict(json_data, model, normalizer, transition_manager=transition_manager)
                await websocket.send(json.dumps(actions))
                logging.info("Sent action chunk")
            except Exception as exc:
                logging.exception("Failed to handle request")
                await websocket.send(json.dumps({"error": str(exc)}))
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the HiMem-Bridge-VLA websocket inference server.")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory containing config.json and weights.")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference_steps", type=int, default=32)
    parser.add_argument(
        "--allow_unsafe_checkpoint_load",
        action="store_true",
        help="Allow torch.load(weights_only=False) fallback for trusted local DeepSpeed checkpoints.",
    )
    parser.add_argument(
        "--transition_trigger_package",
        default=None,
        help="Optional selected transition_trigger package directory for memory-write/replan decisions.",
    )
    parser.add_argument(
        "--transition_dataset_name",
        default=None,
        help="Default transition_trigger dataset schema, e.g. robomme_four_tasks or rmbench_9tasks.",
    )
    parser.add_argument(
        "--transition_device",
        default=None,
        help="Device for the transition trigger; defaults to --device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading HiMem-Bridge-VLA model...")
    model, normalizer = load_model_and_normalizer(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        inference_steps=args.inference_steps,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )
    transition_manager = None
    if args.transition_trigger_package:
        transition_device = args.transition_device or args.device
        transition_manager = ServerTransitionTriggerManager.from_package(
            args.transition_trigger_package,
            device=transition_device,
            default_dataset_name=args.transition_dataset_name,
        )
        logging.info(
            "Loaded transition trigger package=%s dataset=%s device=%s",
            args.transition_trigger_package,
            args.transition_dataset_name,
            transition_device,
        )

    async def main():
        logging.info(f"HiMem-Bridge-VLA server running at ws://{args.host}:{args.port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, transition_manager=transition_manager),
            args.host,
            args.port,
            max_size=DEFAULT_MAX_MESSAGE_SIZE,
        ):
            await asyncio.Future()

    asyncio.run(main())
