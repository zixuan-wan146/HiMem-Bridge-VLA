from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import torch
import websockets

from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.runtime.contract import checkpoint_normalizer_dim
from himem_bridge_vla.runtime.contract import policy_request_from_json
from himem_bridge_vla.runtime.inference_engine import PolicyInferenceEngine
from himem_bridge_vla.runtime.memory_builder import RuntimePolicyState
from himem_bridge_vla.runtime_config import DEFAULT_MAX_MESSAGE_SIZE
from himem_bridge_vla.runtime_config import DEFAULT_SERVER_HOST
from himem_bridge_vla.runtime_config import DEFAULT_SERVER_PORT
from himem_bridge_vla.runtime_config import TARGET_STATE_DIM
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
                "Only enable unsafe pickle loading for trusted local checkpoints."
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
    inference_steps: int = 15,
    allow_unsafe_checkpoint_load: bool = False,
):
    device = resolve_device(device)
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    stats_path = ckpt_dir / "norm_stats.json"
    ckpt_path = ckpt_dir / "model.pt"

    for path in (config_path, stats_path, ckpt_path):
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    checkpoint_load_vlm = bool(config.get("load_vlm", True))
    config["device"] = str(device)
    config["load_vlm"] = True
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = inference_steps

    model = HiMemBridgeVLA(config).eval()
    checkpoint = load_checkpoint_payload(
        ckpt_path,
        allow_unsafe_checkpoint_load=allow_unsafe_checkpoint_load,
    )
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "stage1_torch_checkpoint":
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    load_result = model.load_state_dict(state_dict, strict=checkpoint_load_vlm)
    if not checkpoint_load_vlm:
        bad_missing = [key for key in load_result.missing_keys if not key.startswith("embedder.")]
        if bad_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected non-VLM checkpoint mismatch while loading token-cache checkpoint: "
                f"missing={bad_missing[:20]}, unexpected={load_result.unexpected_keys[:20]}"
            )
        logging.info(
            "Loaded token-cache checkpoint action-side weights with VLM initialized from base model "
            "(ignored %d missing embedder keys).",
            len(load_result.missing_keys),
        )
    model = model.to(device)

    normalizer_dim = checkpoint_normalizer_dim(config)
    normalizer = NormalizationStats(stats, target_dim=normalizer_dim)
    logging.info("Loaded normalization stats robot_keys=%s default_robot_key=%s", normalizer.robot_keys, normalizer.robot_key)
    return model, normalizer


async def handle_request(websocket, engine: PolicyInferenceEngine):
    logging.info("Client connected")
    runtime_state = RuntimePolicyState()
    try:
        async for message in websocket:
            try:
                request = policy_request_from_json(json.loads(message))
                logging.info("Received policy request benchmark=%s", request.benchmark)
                actions = engine.infer(request, runtime_state)
                await websocket.send(json.dumps(actions))
                logging.info("Sent action chunk")
            except Exception as exc:
                logging.exception("Failed to handle request")
                await websocket.send(json.dumps({"error": str(exc)}))
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected.")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run the HiMem-Bridge-VLA websocket inference server.")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint directory containing config.json and weights.")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference_steps", type=int, default=15)
    parser.add_argument(
        "--allow_unsafe_checkpoint_load",
        action="store_true",
        help="Allow torch.load(weights_only=False) fallback for trusted local checkpoints.",
    )
    return parser.parse_args(argv)


async def serve(engine: PolicyInferenceEngine, *, host: str, port: int) -> None:
    logging.info("HiMem-Bridge-VLA server running at ws://%s:%s", host, port)
    async with websockets.serve(
        lambda ws: handle_request(ws, engine),
        host,
        port,
        max_size=DEFAULT_MAX_MESSAGE_SIZE,
        ping_interval=None,
        ping_timeout=None,
    ):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading HiMem-Bridge-VLA model...")
    model, normalizer = load_model_and_normalizer(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        inference_steps=args.inference_steps,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )
    engine = PolicyInferenceEngine(
        model,
        normalizer,
        state_dim=int(model.config.get("state_dim", TARGET_STATE_DIM)),
    )
    asyncio.run(serve(engine, host=args.host, port=args.port))
    return 0
