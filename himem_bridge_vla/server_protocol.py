from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .runtime_config import MAX_VIEWS, TARGET_STATE_DIM
except ImportError:
    from himem_bridge_vla.runtime_config import MAX_VIEWS, TARGET_STATE_DIM


REQUIRED_REQUEST_FIELDS = ("image", "state", "image_mask", "action_mask")


def validate_inference_request(
    data: Mapping[str, Any],
    *,
    max_views: int = MAX_VIEWS,
    target_state_dim: int = TARGET_STATE_DIM,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Inference request must be a JSON object, got {type(data).__name__}")

    missing_fields = [field for field in REQUIRED_REQUEST_FIELDS if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required request fields: {missing_fields}")

    images = _validate_images(data["image"], max_views=max_views)
    state = _validate_state(data["state"], target_state_dim=target_state_dim)
    image_mask = normalize_binary_mask(data["image_mask"], max_views, "image_mask")
    action_mask = normalize_binary_mask(data["action_mask"], target_state_dim, "action_mask")

    if sum(image_mask) == 0:
        raise ValueError("image_mask must activate at least one image")
    if sum(action_mask) == 0:
        raise ValueError("action_mask must activate at least one action dimension")

    prompt = data.get("prompt", "")
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        prompt = str(prompt)

    episode_id = data.get("episode_id")
    if episode_id is not None and not isinstance(episode_id, str):
        episode_id = str(episode_id)

    return {
        "image": images,
        "state": state,
        "prompt": prompt,
        "image_mask": image_mask,
        "action_mask": action_mask,
        "episode_id": episode_id,
        "reset_memory": bool(data.get("reset_memory", False)),
        "return_debug": bool(data.get("return_debug", False)),
    }


def normalize_binary_mask(mask: Any, target_dim: int, field_name: str = "mask") -> list[int]:
    try:
        flat_mask = np.asarray(mask, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a sequence of 0/1 values") from exc

    if flat_mask.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if flat_mask.size > target_dim:
        raise ValueError(f"{field_name} length {flat_mask.size} exceeds target dimension {target_dim}")
    invalid_values = sorted({int(value) for value in flat_mask.tolist()} - {0, 1})
    if invalid_values:
        raise ValueError(f"{field_name} must contain only 0/1 values, got {invalid_values}")
    if flat_mask.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.int32)
        padded[: flat_mask.size] = flat_mask
        flat_mask = padded
    return flat_mask.astype(int).tolist()


def _validate_images(images: Any, *, max_views: int) -> list[Any]:
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        raise ValueError("image must be a sequence of image arrays")
    if len(images) != max_views:
        raise ValueError(f"Must provide exactly {max_views} images, got {len(images)}")
    for index, image in enumerate(images):
        _validate_image_array(image, index)
    return list(images)


def _validate_image_array(image: Any, index: int) -> None:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"image[{index}] must have shape HxWx3, got ndim={array.ndim}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"image[{index}] must have non-empty height and width, got shape={array.shape}")
    if array.shape[2] != 3:
        raise ValueError(f"image[{index}] must have 3 channels, got shape={array.shape}")
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"image[{index}] must contain numeric pixel values, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"image[{index}] must contain only finite pixel values")
    if array.min() < 0 or array.max() > 255:
        raise ValueError(f"image[{index}] pixel values must be in the 0..255 range")


def _validate_state(state: Any, *, target_state_dim: int) -> list[float]:
    try:
        flat_state = np.asarray(state, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("state must be a sequence of numeric values") from exc

    if flat_state.size == 0:
        raise ValueError("state must not be empty")
    if flat_state.size > target_state_dim:
        raise ValueError(f"state length {flat_state.size} exceeds target dimension {target_state_dim}")
    if not np.isfinite(flat_state).all():
        raise ValueError("state must contain only finite values")
    return [float(value) for value in flat_state.tolist()]
