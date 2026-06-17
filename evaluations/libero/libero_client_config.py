from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, MutableMapping


DEFAULT_TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_MAX_STEPS = [25, 25, 25, 95]


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = _env_value(environ, name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_list(environ: Mapping[str, str], name: str, default: list[str]) -> list[str]:
    value = _env_value(environ, name)
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one non-empty item")
    return items


def env_int_list(environ: Mapping[str, str], name: str, default: list[int]) -> list[int]:
    value = _env_value(environ, name)
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers, got {value!r}") from exc


def align_max_steps(max_steps: list[int], task_suites: list[str]) -> list[int]:
    if len(max_steps) == 1 and len(task_suites) > 1:
        return max_steps * len(task_suites)
    if len(max_steps) != len(task_suites):
        raise ValueError(
            "HIMEM_LIBERO_MAX_STEPS must provide one integer per task suite: "
            f"got {len(max_steps)} values for {len(task_suites)} suites"
        )
    return max_steps


@dataclass(frozen=True)
class LiberoClientConfig:
    horizon: int
    max_steps: list[int]
    server_url: str
    ckpt_name: str
    task_suites: list[str]
    log_dir: str
    video_dir: str
    log_file: str
    result_file: str
    num_episodes: int
    task_limit: int
    seed: int
    mujoco_gl: str
    transition_replan_action_limit: int
    transition_dataset_name: str | None
    transition_trace_file: str | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LiberoClientConfig":
        environ = os.environ if environ is None else environ
        ckpt_name = environ.get("HIMEM_LIBERO_CKPT_NAME", "HiMem_libero_all")
        log_dir = environ.get("HIMEM_LIBERO_LOG_DIR", "./log_file")
        video_dir = environ.get("HIMEM_LIBERO_VIDEO_DIR", f"./video_log_file/{ckpt_name}")
        log_file = environ.get("HIMEM_LIBERO_LOG_FILE", os.path.join(log_dir, f"{ckpt_name}.txt"))
        result_file = environ.get("HIMEM_LIBERO_RESULT_FILE", os.path.join(log_dir, f"{ckpt_name}_results.json"))
        task_suites = env_list(environ, "HIMEM_LIBERO_TASK_SUITES", DEFAULT_TASK_SUITES)
        max_steps = align_max_steps(env_int_list(environ, "HIMEM_LIBERO_MAX_STEPS", DEFAULT_MAX_STEPS), task_suites)
        transition_dataset_name = _env_value(environ, "HIMEM_LIBERO_TRANSITION_DATASET_NAME")
        transition_trace_file = _resolve_transition_trace_file(
            environ,
            transition_dataset_name=transition_dataset_name,
            result_file=result_file,
            log_dir=log_dir,
            ckpt_name=ckpt_name,
        )

        config = cls(
            horizon=env_int(environ, "HIMEM_LIBERO_HORIZON", 14),
            max_steps=max_steps,
            server_url=environ.get("HIMEM_SERVER_URI", environ.get("HIMEM_LIBERO_SERVER_URL", "ws://127.0.0.1:9000")),
            ckpt_name=ckpt_name,
            task_suites=task_suites,
            log_dir=log_dir,
            video_dir=video_dir,
            log_file=log_file,
            result_file=result_file,
            num_episodes=env_int(environ, "HIMEM_LIBERO_EPISODES", 10),
            task_limit=env_int(environ, "HIMEM_LIBERO_TASK_LIMIT", 0),
            seed=env_int(environ, "HIMEM_LIBERO_SEED", 42),
            mujoco_gl=environ.get("HIMEM_MUJOCO_GL", "osmesa"),
            transition_replan_action_limit=env_int(environ, "HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT", 0),
            transition_dataset_name=transition_dataset_name,
            transition_trace_file=transition_trace_file,
        )
        config.validate()
        return config

    @property
    def SERVER_URL(self) -> str:
        return self.server_url

    @property
    def SEED(self) -> int:
        return self.seed

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"HIMEM_LIBERO_HORIZON must be positive, got {self.horizon}")
        if self.num_episodes <= 0:
            raise ValueError(f"HIMEM_LIBERO_EPISODES must be positive, got {self.num_episodes}")
        if self.task_limit < 0:
            raise ValueError(f"HIMEM_LIBERO_TASK_LIMIT must be non-negative, got {self.task_limit}")
        invalid_max_steps = [value for value in self.max_steps if value <= 0]
        if invalid_max_steps:
            raise ValueError(f"HIMEM_LIBERO_MAX_STEPS values must be positive, got {invalid_max_steps}")
        if self.transition_replan_action_limit < 0:
            raise ValueError(
                "HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT must be non-negative, "
                f"got {self.transition_replan_action_limit}"
            )
        if self.mujoco_gl not in {"osmesa", "egl", "glfw"}:
            raise ValueError(f"HIMEM_MUJOCO_GL must be one of osmesa, egl, glfw; got {self.mujoco_gl!r}")


def _resolve_transition_trace_file(
    environ: Mapping[str, str],
    *,
    transition_dataset_name: str | None,
    result_file: str,
    log_dir: str,
    ckpt_name: str,
) -> str | None:
    explicit = _env_value(environ, "HIMEM_LIBERO_TRANSITION_TRACE_FILE")
    if explicit is not None:
        return explicit
    if transition_dataset_name is None:
        return None
    result_dir = os.path.dirname(result_file) or log_dir or "."
    return os.path.join(result_dir, f"{ckpt_name}_transition_trace.jsonl")


def configure_mujoco_environment(
    config: LiberoClientConfig,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    environ.setdefault("MUJOCO_GL", config.mujoco_gl)
    if config.mujoco_gl == "egl":
        environ.setdefault("PYOPENGL_PLATFORM", "egl")
