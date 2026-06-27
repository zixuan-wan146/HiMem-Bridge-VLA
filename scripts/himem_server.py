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
    inference_steps: int = 15,
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
    state_dict = checkpoint["module"] if "module" in checkpoint else checkpoint
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


class RuntimePolicyState:
    def __init__(self) -> None:
        self.executed_actions: torch.Tensor | None = None
        self.executed_action_mask: torch.Tensor | None = None
        self.previous_visual_tokens: torch.Tensor | None = None

    def reset(self, model: HiMemBridgeVLA) -> None:
        self.executed_actions = None
        self.executed_action_mask = None
        self.previous_visual_tokens = None
        model.reset_progress_state()

    def progress_inputs(
        self,
        model: HiMemBridgeVLA,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        planner = model.progress_state_planner
        if planner is None:
            return None, None

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        if self.executed_actions is None:
            actions = torch.zeros(1, stride, action_dim, device=device, dtype=dtype)
            mask = torch.zeros(1, stride, device=device, dtype=torch.bool)
            return actions, mask

        return (
            self.executed_actions.to(device=device, dtype=dtype),
            self.executed_action_mask.to(device=device) if self.executed_action_mask is not None else None,
        )

    def store_executed_actions(self, model: HiMemBridgeVLA, normalized_action: torch.Tensor) -> None:
        planner = model.progress_state_planner
        if planner is None:
            return

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        action = normalized_action[:, :stride, :action_dim].detach()
        if action.shape[1] != stride:
            pad = torch.zeros(
                action.shape[0],
                stride - action.shape[1],
                action_dim,
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad], dim=1)
        self.executed_actions = action.cpu()
        self.executed_action_mask = torch.ones(action.shape[:2], dtype=torch.bool)

    def short_memory_inputs(
        self,
        model: HiMemBridgeVLA,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not bool(getattr(model, "use_direct_bridge", False)):
            return None, None, None
        capacity = int(model.config.get("memory_short_capacity", 2))
        entry_tokens = int(model.config.get("memory_entry_tokens", 16))
        hidden_dim = int(model.config.get("embed_dim", 896))
        if capacity <= 0 or entry_tokens <= 0:
            return None, None, None

        memory = torch.zeros(1, capacity * entry_tokens, hidden_dim, device=device, dtype=dtype)
        mask = torch.zeros(1, capacity * entry_tokens, device=device, dtype=torch.bool)
        time_ids = torch.arange(capacity, device=device, dtype=torch.long).repeat_interleave(entry_tokens)
        time_ids = time_ids.unsqueeze(0)

        if self.previous_visual_tokens is not None:
            packed = _pack_runtime_visual_tokens(
                self.previous_visual_tokens.to(device=device, dtype=dtype),
                target_tokens=entry_tokens,
            )
            memory[:, :entry_tokens, :] = packed
            mask[:, :entry_tokens] = True
        return memory, mask, time_ids

    def store_visual_tokens(self, visual_tokens: torch.Tensor | None) -> None:
        if visual_tokens is None:
            return
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != 1:
            raise ValueError(f"visual_tokens must have shape [1, N, D], got {tuple(visual_tokens.shape)}")
        self.previous_visual_tokens = visual_tokens.detach().cpu()


def _pack_runtime_visual_tokens(tokens: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"tokens must have shape [1, N, D] or [N, D], got {tuple(tokens.shape)}")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    token_count = int(tokens.shape[1])
    if token_count <= 0:
        raise ValueError("visual token sequence must be non-empty")
    if token_count == target_tokens:
        return tokens.contiguous()
    if token_count < target_tokens:
        output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
        output[:, :token_count, :] = tokens
        return output

    boundaries = torch.linspace(0, token_count, steps=target_tokens + 1, device=tokens.device).round().long()
    output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(token_count, start + 1)
        output[:, index, :] = tokens[:, start:end, :].mean(dim=1)
    return output


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


def infer_from_json_dict(data: dict, model, normalizer, runtime_state: RuntimePolicyState | None = None):
    device = next(model.parameters()).device
    model_state_dim = int(model.config.get("state_dim", TARGET_STATE_DIM))
    model_action_dim = int(getattr(model, "per_action_dim", model.config.get("per_action_dim", TARGET_STATE_DIM)))
    request = validate_inference_request(
        data,
        target_state_dim=model_state_dim,
        target_action_dim=model_action_dim,
        max_action_mask_dim=TARGET_STATE_DIM,
    )
    if runtime_state is not None and bool(data.get("reset_memory", False)):
        runtime_state.reset(model)

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
        embedding_output = model.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_hidden_states=model.use_bridge,
        )
        if hasattr(embedding_output, "fused_tokens"):
            hidden_states = embedding_output.hidden_states
            visual_tokens = getattr(embedding_output, "visual_tokens", None)
            fused_tokens = visual_tokens if visual_tokens is not None else embedding_output.fused_tokens
        else:
            fused_tokens = embedding_output
            hidden_states = None
            visual_tokens = None

        executed_actions, executed_action_mask = (None, None)
        memory_context, memory_context_mask, short_memory_time_ids = (None, None, None)
        if runtime_state is not None:
            executed_actions, executed_action_mask = runtime_state.progress_inputs(
                model,
                device=device,
                dtype=fused_tokens.dtype,
            )
            memory_context, memory_context_mask, short_memory_time_ids = runtime_state.short_memory_inputs(
                model,
                device=device,
                dtype=fused_tokens.dtype,
            )

        action = model.predict_action(
            fused_tokens,
            norm_state,
            action_mask=action_mask,
            hidden_states=hidden_states,
            memory_context=memory_context,
            memory_context_mask=memory_context_mask,
            short_memory_time_ids=short_memory_time_ids,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
        )
        if action.numel() % model_action_dim != 0:
            raise ValueError(f"Model returned {action.numel()} action values, not divisible by per_action_dim={model_action_dim}")
        normalized_action = action.reshape(1, -1, model_action_dim)
        if runtime_state is not None:
            runtime_state.store_executed_actions(model, normalized_action)
            runtime_state.store_visual_tokens(visual_tokens)
        denormalized_action = normalizer.denormalize_action(normalized_action[0], robot_key=robot_key)
        actions = denormalized_action.cpu().numpy().tolist()
        if not request["return_debug"]:
            return actions
        return {"actions": actions}


async def handle_request(websocket, model, normalizer):
    logging.info("Client connected")
    runtime_state = RuntimePolicyState()
    try:
        async for message in websocket:
            try:
                json_data = json.loads(message)
                logging.info("Received JSON observation")
                actions = infer_from_json_dict(json_data, model, normalizer, runtime_state)
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
    parser.add_argument("--inference_steps", type=int, default=15)
    parser.add_argument(
        "--allow_unsafe_checkpoint_load",
        action="store_true",
        help="Allow torch.load(weights_only=False) fallback for trusted local DeepSpeed checkpoints.",
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
    async def main():
        logging.info(f"HiMem-Bridge-VLA server running at ws://{args.host}:{args.port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            args.host,
            args.port,
            max_size=DEFAULT_MAX_MESSAGE_SIZE,
            ping_interval=None,
            ping_timeout=None,
        ):
            await asyncio.Future()

    asyncio.run(main())
