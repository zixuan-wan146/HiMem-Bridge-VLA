from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from himem_bridge_vla.runtime.contract import PolicyRequest

from .action import DEFAULT_ACTION_DIM
from .observation import DEFAULT_CAMERA_NAMES
from .observation import build_rmbench_images_by_view
from .observation import build_rmbench_state
from .spec import RMBENCH_SPEC


DEFAULT_ROBOT_KEY = "rmbench"


def build_request_from_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    state_source: str = "endpose",
    action_dim: int = DEFAULT_ACTION_DIM,
    robot_key: str = DEFAULT_ROBOT_KEY,
) -> dict[str, Any]:
    request = PolicyRequest(
        benchmark=RMBENCH_SPEC.name,
        prompt=str(prompt or ""),
        images_by_view=build_rmbench_images_by_view(observation, camera_names=camera_names),
        state=build_rmbench_state(observation, state_source=state_source),
        action_dim=int(action_dim),
        robot_key=str(robot_key),
    )
    return policy_request_to_json(request)


def encode_obs(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation


def policy_request_to_json(request: PolicyRequest) -> dict[str, Any]:
    return {
        "benchmark": request.benchmark,
        "prompt": request.prompt,
        "images_by_view": {
            view_name: image.astype("uint8").tolist()
            for view_name, image in request.images_by_view.items()
        },
        "state": request.state.astype("float32").tolist(),
        "action_dim": int(request.action_dim),
        "robot_key": request.robot_key,
        "reset_memory": bool(request.reset_memory),
    }
