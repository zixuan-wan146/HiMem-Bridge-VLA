from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

import numpy as np

try:
    from websockets.sync.client import connect
except ImportError as exc:  # pragma: no cover - exercised only in missing dependency envs
    connect = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None


DEFAULT_SERVER_URI = os.getenv("HIMEM_SERVER_URI", "ws://127.0.0.1:9000")
DEFAULT_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
DEFAULT_ACTION_HORIZON = 32
DEFAULT_ACTION_DIM = 14
DEFAULT_ROBOT_KEY = "rmbench"


class RMBenchHiMemPolicy:
    def __init__(
        self,
        *,
        server_uri: str = DEFAULT_SERVER_URI,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        state_source: str = "endpose",
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        action_dim: int = DEFAULT_ACTION_DIM,
        action_type: str = "qpos",
        robot_key: str = DEFAULT_ROBOT_KEY,
        request_timeout: float = 120.0,
        stop_on_success: bool = True,
    ) -> None:
        if int(action_horizon) <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if int(action_dim) <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if state_source not in {"endpose", "qpos"}:
            raise ValueError(f"state_source must be 'endpose' or 'qpos', got {state_source!r}")
        if not camera_names:
            raise ValueError("camera_names must contain at least one camera")

        self.server_uri = str(server_uri)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.state_source = str(state_source)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.action_type = str(action_type)
        self.robot_key = str(robot_key)
        self.request_timeout = float(request_timeout)
        self.stop_on_success = bool(stop_on_success)
        self.obs_cache: list[dict[str, Any]] = []
        self._websocket = None

    def reset(self) -> None:
        self.obs_cache.clear()

    def close(self) -> None:
        if self._websocket is not None:
            self._websocket.close()
            self._websocket = None

    def encode_observation(self, observation: Mapping[str, Any], prompt: str = "") -> dict[str, Any]:
        return build_request_from_observation(
            observation,
            prompt=prompt,
            camera_names=self.camera_names,
            state_source=self.state_source,
            action_dim=self.action_dim,
            robot_key=self.robot_key,
        )

    def update_obs(self, obs: Mapping[str, Any]) -> None:
        self.obs_cache[:] = [dict(obs)]

    def get_action(self) -> list[list[float]]:
        if not self.obs_cache:
            raise RuntimeError("obs_cache is empty; call update_obs before get_action")
        return self.request_actions(self.obs_cache[-1])

    def request_actions(self, request: Mapping[str, Any]) -> list[list[float]]:
        websocket = self._connect()
        websocket.send(json.dumps(request))
        response = websocket.recv(timeout=self.request_timeout)
        return parse_action_response(
            response,
            horizon=self.action_horizon,
            action_dim=self.action_dim,
        )

    def _connect(self):
        if connect is None:
            raise RuntimeError("websockets.sync.client is required for RMBench HiMem policy") from _WEBSOCKETS_IMPORT_ERROR
        if self._websocket is None:
            self._websocket = connect(
                self.server_uri,
                open_timeout=self.request_timeout,
                close_timeout=self.request_timeout,
            )
        return self._websocket


def build_request_from_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    state_source: str = "endpose",
    action_dim: int = DEFAULT_ACTION_DIM,
    robot_key: str = DEFAULT_ROBOT_KEY,
) -> dict[str, Any]:
    images: list[list[list[list[int]]]] = []
    image_mask: list[int] = []
    for camera_name in camera_names:
        image = _extract_rgb(observation, camera_name)
        images.append(image.astype(np.uint8).tolist())
        image_mask.append(1)

    return {
        "image": images,
        "state": _build_state(observation, state_source=state_source),
        "prompt": str(prompt or ""),
        "image_mask": image_mask,
        "action_mask": [1] * int(action_dim),
        "robot_key": str(robot_key),
    }


def encode_obs(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return observation


def get_model(usr_args: Mapping[str, Any]) -> RMBenchHiMemPolicy:
    return RMBenchHiMemPolicy(
        server_uri=str(usr_args.get("server_uri") or DEFAULT_SERVER_URI),
        camera_names=tuple(usr_args.get("camera_names") or DEFAULT_CAMERA_NAMES),
        state_source=str(usr_args.get("state_source", "endpose")),
        action_horizon=int(usr_args.get("action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(usr_args.get("action_dim", DEFAULT_ACTION_DIM)),
        action_type=str(usr_args.get("action_type", "qpos")),
        robot_key=str(usr_args.get("robot_key", DEFAULT_ROBOT_KEY)),
        request_timeout=float(usr_args.get("request_timeout", 120.0)),
        stop_on_success=bool(usr_args.get("stop_on_success", True)),
    )


def eval(TASK_ENV, model: RMBenchHiMemPolicy, observation):
    prompt = TASK_ENV.get_instruction()
    request = model.encode_observation(observation, prompt=prompt)
    model.update_obs(request)
    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action, action_type=model.action_type)
        if model.stop_on_success and getattr(TASK_ENV, "eval_success", False):
            break
        observation = TASK_ENV.get_obs()
        request = model.encode_observation(observation, prompt=TASK_ENV.get_instruction())
        model.update_obs(request)
    return observation


def reset_model(model: RMBenchHiMemPolicy) -> None:
    model.reset()


def parse_action_response(message: str, *, horizon: int, action_dim: int) -> list[list[float]]:
    if int(horizon) <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if int(action_dim) <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        if "error" in payload:
            raise RuntimeError(f"HiMem server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list, got {type(payload).__name__}")
    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    parsed: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < action_dim:
            raise ValueError(f"Action at step {step} has dimension {len(row)}, expected at least {action_dim}")
        parsed.append([_to_float(value, step, dim) for dim, value in enumerate(row[:action_dim])])
    return parsed


def _extract_rgb(observation: Mapping[str, Any], camera_name: str) -> np.ndarray:
    try:
        image = observation["observation"][camera_name]["rgb"]
    except KeyError as exc:
        raise KeyError(f"RMBench observation is missing camera {camera_name!r} rgb") from exc
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"camera {camera_name!r} rgb must have shape HxWx3, got {array.shape}")
    if array.size == 0:
        raise ValueError(f"camera {camera_name!r} rgb is empty")
    if array.min() < 0 or array.max() > 255:
        raise ValueError(f"camera {camera_name!r} rgb values must be in 0..255")
    return array


def _build_state(observation: Mapping[str, Any], *, state_source: str) -> list[float]:
    if state_source == "qpos":
        return _build_qpos_state(observation)
    return _build_endpose_state(observation)


def _build_qpos_state(observation: Mapping[str, Any]) -> list[float]:
    joint_action = observation["joint_action"]
    if "vector" in joint_action:
        return _flat_float_list(joint_action["vector"], "joint_action.vector")
    return _flat_float_list(
        list(joint_action["left_arm"])
        + [joint_action["left_gripper"]]
        + list(joint_action["right_arm"])
        + [joint_action["right_gripper"]],
        "joint_action split fields",
    )


def _build_endpose_state(observation: Mapping[str, Any]) -> list[float]:
    endpose = observation["endpose"]
    return _flat_float_list(
        list(np.asarray(endpose["left_endpose"]).reshape(-1))
        + list(np.asarray(endpose["left_gripper"]).reshape(-1))
        + list(np.asarray(endpose["right_endpose"]).reshape(-1))
        + list(np.asarray(endpose["right_gripper"]).reshape(-1)),
        "endpose state",
    )


def _flat_float_list(values: Any, field_name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return [float(value) for value in array.tolist()]


def _to_float(value: Any, step: int, dim: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc

