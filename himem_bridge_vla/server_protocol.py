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
    target_action_dim: int | None = None,
    max_action_mask_dim: int = TARGET_STATE_DIM,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"Inference request must be a JSON object, got {type(data).__name__}")

    missing_fields = [field for field in REQUIRED_REQUEST_FIELDS if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required request fields: {missing_fields}")

    target_action_dim = target_state_dim if target_action_dim is None else target_action_dim
    images = _validate_images(data["image"], max_views=max_views)
    state = _validate_state(data["state"], target_state_dim=target_state_dim)
    image_mask = normalize_binary_mask(data["image_mask"], max_views, "image_mask")
    action_mask = normalize_action_mask(
        data["action_mask"],
        target_action_dim=target_action_dim,
        max_action_mask_dim=max_action_mask_dim,
    )

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
    session_id = data.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        session_id = str(session_id)
    robot_key = data.get("robot_key")
    if robot_key is not None and not isinstance(robot_key, str):
        robot_key = str(robot_key)

    return {
        "image": images,
        "state": state,
        "prompt": prompt,
        "image_mask": image_mask,
        "action_mask": action_mask,
        "episode_id": episode_id,
        "session_id": session_id,
        "robot_key": robot_key,
        "reset_memory": bool(data.get("reset_memory", False)),
        "return_debug": bool(data.get("return_debug", False)),
    }


def normalize_binary_mask(mask: Any, target_dim: int, field_name: str = "mask") -> list[int]:
    flat_mask = _coerce_binary_mask(mask, field_name)

    if flat_mask.size > target_dim:
        raise ValueError(f"{field_name} length {flat_mask.size} exceeds target dimension {target_dim}")
    if flat_mask.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.int32)
        padded[: flat_mask.size] = flat_mask
        flat_mask = padded
    return flat_mask.astype(int).tolist()


def normalize_action_mask(
    mask: Any,
    *,
    target_action_dim: int,
    max_action_mask_dim: int = TARGET_STATE_DIM,
) -> list[int]:
    if target_action_dim <= 0:
        raise ValueError(f"target_action_dim must be positive, got {target_action_dim}")
    if max_action_mask_dim < target_action_dim:
        raise ValueError(
            f"max_action_mask_dim {max_action_mask_dim} must be >= target_action_dim {target_action_dim}"
        )

    flat_mask = _coerce_binary_mask(mask, "action_mask")
    if flat_mask.size > max_action_mask_dim:
        raise ValueError(f"action_mask length {flat_mask.size} exceeds maximum dimension {max_action_mask_dim}")
    if flat_mask.size > target_action_dim:
        trailing = flat_mask[target_action_dim:]
        if np.any(trailing != 0):
            raise ValueError(
                f"action_mask has active dimensions beyond model action dimension {target_action_dim}"
            )
        flat_mask = flat_mask[:target_action_dim]
    if flat_mask.size < target_action_dim:
        padded = np.zeros(target_action_dim, dtype=np.int32)
        padded[: flat_mask.size] = flat_mask
        flat_mask = padded
    return flat_mask.astype(int).tolist()


def checkpoint_normalizer_dim(config: Mapping[str, Any], default_dim: int = TARGET_STATE_DIM) -> int:
    return max(
        _positive_int_or_default(config.get("state_dim"), default_dim),
        _positive_int_or_default(config.get("per_action_dim"), default_dim),
    )


def _positive_int_or_default(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_binary_mask(mask: Any, field_name: str) -> np.ndarray:
    try:
        flat_mask = np.asarray(mask, dtype=np.int32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a sequence of 0/1 values") from exc

    if flat_mask.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    invalid_values = sorted({int(value) for value in flat_mask.tolist()} - {0, 1})
    if invalid_values:
        raise ValueError(f"{field_name} must contain only 0/1 values, got {invalid_values}")
    return flat_mask


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
