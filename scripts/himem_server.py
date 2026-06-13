import argparse
import asyncio
from contextlib import nullcontext
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import websockets
from PIL import Image
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
from himem_bridge_vla.server_protocol import validate_inference_request
from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.utils.normalization import NormalizationStats


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
    return resolved


def load_model_and_normalizer(ckpt_dir, device: str = "cuda", inference_steps: int = 32):
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
    # DeepSpeed checkpoints include non-tensor metadata; load only trusted checkpoints.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["module"] if "module" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    return model, NormalizationStats(stats)


def decode_image_from_list(img_list, device) -> torch.Tensor:
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
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


def infer_from_json_dict(data: dict, model, normalizer):
    device = next(model.parameters()).device
    request = validate_inference_request(data)

    images = [decode_image_from_list(img, device) for img in request["image"]]
    for img in images:
        expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(img.shape) != expected_shape:
            raise ValueError(f"image_size must be {expected_shape}, got {tuple(img.shape)}")

    state = torch.tensor(request["state"], dtype=torch.float32, device=device)
    norm_state = normalizer.normalize_state(pad_state_tensor(state)).to(dtype=torch.float32)

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
            reset_memory=request["reset_memory"],
        )
        action = action.reshape(1, -1, TARGET_STATE_DIM)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()


async def handle_request(websocket, model, normalizer):
    logging.info("Client connected")
    try:
        async for message in websocket:
            try:
                json_data = json.loads(message)
                logging.info("Received JSON observation")
                actions = infer_from_json_dict(json_data, model, normalizer)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading HiMem-Bridge-VLA model...")
    model, normalizer = load_model_and_normalizer(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        inference_steps=args.inference_steps,
    )

    async def main():
        logging.info(f"HiMem-Bridge-VLA server running at ws://{args.host}:{args.port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            args.host,
            args.port,
            max_size=DEFAULT_MAX_MESSAGE_SIZE,
        ):
            await asyncio.Future()

    asyncio.run(main())
