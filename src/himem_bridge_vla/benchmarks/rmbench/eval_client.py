from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from .action import DEFAULT_ACTION_DIM
from .action import DEFAULT_ACTION_HORIZON
from .action import parse_action_response
from .request_builder import DEFAULT_CAMERA_NAMES
from .request_builder import DEFAULT_ROBOT_KEY
from .request_builder import build_request_from_observation

try:
    from websockets.sync.client import connect
except ImportError as exc:  # pragma: no cover - exercised only in missing dependency envs
    connect = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None


DEFAULT_SERVER_URI = os.getenv("HIMEM_SERVER_URI", "ws://127.0.0.1:9000")


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
