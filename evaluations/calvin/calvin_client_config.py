from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, MutableMapping


DEFAULT_CALVIN_ROOT = "/root/autodl-tmp/calvin"
DEFAULT_CALVIN_DATASET_RELATIVE_PATH = "dataset/task_ABC_D"
VALID_GRIPPER_MODES = {"openvla", "passthrough", "sign"}
VALID_RESET_MEMORY_SCOPES = {"sequence", "subtask", "never"}


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


def env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = _env_value(environ, name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _default_dataset_path(calvin_root: str) -> str:
    return str(Path(calvin_root).expanduser() / DEFAULT_CALVIN_DATASET_RELATIVE_PATH)


@dataclass(frozen=True)
class CalvinClientConfig:
    horizon: int
    max_steps_per_subtask: int
    server_url: str
    ckpt_name: str
    calvin_root: str
    dataset_path: str
    annotations_path: str
    log_dir: str
    video_dir: str
    log_file: str
    result_file: str
    manifest_file: str
    num_sequences: int
    sequence_offset: int
    seed: int
    mujoco_gl: str
    gripper_mode: str
    reset_memory_scope: str
    save_video: bool
    video_fps: int
    show_gui: bool

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CalvinClientConfig":
        environ = os.environ if environ is None else environ
        ckpt_name = environ.get("HIMEM_CALVIN_CKPT_NAME", "HiMem_calvin_eval")
        calvin_root = environ.get("HIMEM_CALVIN_ROOT", DEFAULT_CALVIN_ROOT)
        dataset_path = environ.get("HIMEM_CALVIN_DATASET_PATH", _default_dataset_path(calvin_root))
        log_dir = environ.get("HIMEM_CALVIN_LOG_DIR", "./log_file")
        video_dir = environ.get("HIMEM_CALVIN_VIDEO_DIR", f"./video_log_file/{ckpt_name}")
        log_file = environ.get("HIMEM_CALVIN_LOG_FILE", os.path.join(log_dir, f"{ckpt_name}.txt"))
        result_file = environ.get("HIMEM_CALVIN_RESULT_FILE", os.path.join(log_dir, f"{ckpt_name}_results.json"))
        manifest_file = environ.get(
            "HIMEM_CALVIN_MANIFEST_FILE",
            os.path.join(log_dir, f"{ckpt_name}_run_manifest.json"),
        )

        config = cls(
            horizon=env_int(environ, "HIMEM_CALVIN_HORIZON", 14),
            max_steps_per_subtask=env_int(environ, "HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK", 360),
            server_url=environ.get("HIMEM_SERVER_URI", environ.get("HIMEM_CALVIN_SERVER_URL", "ws://127.0.0.1:9000")),
            ckpt_name=ckpt_name,
            calvin_root=calvin_root,
            dataset_path=dataset_path,
            annotations_path=environ.get("HIMEM_CALVIN_ANNOTATIONS_PATH", ""),
            log_dir=log_dir,
            video_dir=video_dir,
            log_file=log_file,
            result_file=result_file,
            manifest_file=manifest_file,
            num_sequences=env_int(environ, "HIMEM_CALVIN_NUM_SEQUENCES", 1000),
            sequence_offset=env_int(environ, "HIMEM_CALVIN_SEQUENCE_OFFSET", 0),
            seed=env_int(environ, "HIMEM_CALVIN_SEED", 42),
            mujoco_gl=environ.get("HIMEM_MUJOCO_GL", "osmesa"),
            gripper_mode=environ.get("HIMEM_CALVIN_GRIPPER_MODE", "openvla"),
            reset_memory_scope=environ.get("HIMEM_CALVIN_RESET_MEMORY_SCOPE", "sequence"),
            save_video=env_bool(environ, "HIMEM_CALVIN_SAVE_VIDEO", False),
            video_fps=env_int(environ, "HIMEM_CALVIN_VIDEO_FPS", 30),
            show_gui=env_bool(environ, "HIMEM_CALVIN_SHOW_GUI", False),
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
            raise ValueError(f"HIMEM_CALVIN_HORIZON must be positive, got {self.horizon}")
        if self.max_steps_per_subtask <= 0:
            raise ValueError(
                f"HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK must be positive, got {self.max_steps_per_subtask}"
            )
        if self.num_sequences <= 0:
            raise ValueError(f"HIMEM_CALVIN_NUM_SEQUENCES must be positive, got {self.num_sequences}")
        if self.sequence_offset < 0:
            raise ValueError(f"HIMEM_CALVIN_SEQUENCE_OFFSET must be non-negative, got {self.sequence_offset}")
        if self.video_fps <= 0:
            raise ValueError(f"HIMEM_CALVIN_VIDEO_FPS must be positive, got {self.video_fps}")
        if self.mujoco_gl not in {"osmesa", "egl", "glfw"}:
            raise ValueError(f"HIMEM_MUJOCO_GL must be one of osmesa, egl, glfw; got {self.mujoco_gl!r}")
        if self.gripper_mode not in VALID_GRIPPER_MODES:
            raise ValueError(
                "HIMEM_CALVIN_GRIPPER_MODE must be one of "
                f"{sorted(VALID_GRIPPER_MODES)}, got {self.gripper_mode!r}"
            )
        if self.reset_memory_scope not in VALID_RESET_MEMORY_SCOPES:
            raise ValueError(
                "HIMEM_CALVIN_RESET_MEMORY_SCOPE must be one of "
                f"{sorted(VALID_RESET_MEMORY_SCOPES)}, got {self.reset_memory_scope!r}"
            )


def configure_calvin_environment(
    config: CalvinClientConfig,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    environ.setdefault("CALVIN_ROOT", config.calvin_root)
    environ.setdefault("MUJOCO_GL", config.mujoco_gl)
    if config.mujoco_gl == "egl":
        environ.setdefault("PYOPENGL_PLATFORM", "egl")
