from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_RMBENCH_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
DEFAULT_RMBENCH_ACTION_KEY = "joint_action/vector"
DEFAULT_RMBENCH_SETTING = "demo_clean"
DEFAULT_RMBENCH_ROBOT_KEY = "rmbench"


@dataclass(frozen=True)
class RMBenchEpisodeFile:
    task_name: str
    hdf5_path: Path
    instruction_path: Path | None


@dataclass(frozen=True)
class RMBenchFrame:
    tau: int
    instruction: str
    images_by_view: Mapping[str, Image.Image]
    joint_action: np.ndarray
    endpose_by_arm: Mapping[str, np.ndarray]
    gripper_by_arm: Mapping[str, np.ndarray]
    state_vector: np.ndarray


@dataclass(frozen=True)
class RMBenchStateActionArrays:
    states: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class RMBenchNormalizationResult:
    stats: Mapping[str, Any]
    metadata: Mapping[str, Any]


class RMBenchEpisodeReader:
    """Read one local RMBench HDF5 episode without importing the simulator."""

    def __init__(
        self,
        hdf5_path: str | Path,
        *,
        instruction_path: str | Path | None = None,
        camera_names: Sequence[str] = DEFAULT_RMBENCH_CAMERA_NAMES,
        action_key: str = DEFAULT_RMBENCH_ACTION_KEY,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser()
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)
        self.instruction_path = _resolve_instruction_path(self.hdf5_path, instruction_path)
        self.instruction = read_rmbench_instruction(self.instruction_path)
        self.camera_names = tuple(str(name) for name in camera_names)
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        self.action_key = str(action_key)

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            self.length = _dataset_length(handle, self.action_key)
            self.action_dim = int(np.asarray(handle[self.action_key].shape[-1]).item())
            self._validate_camera_lengths(handle)

    def __len__(self) -> int:
        return self.length

    def read_frame(self, index: int) -> RMBenchFrame:
        index = int(index)
        if index < 0 or index >= self.length:
            raise IndexError(f"frame index {index} out of range for episode length {self.length}")

        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            images_by_view = {
                camera_name: decode_rmbench_rgb(handle[_camera_rgb_key(camera_name)][index])
                for camera_name in self.camera_names
            }
            joint_action = np.asarray(handle[self.action_key][index], dtype=np.float32).reshape(-1)
            endpose_by_arm = _read_arm_arrays(
                handle,
                index,
                {
                    "left": ("endpose/left_endpose",),
                    "right": ("endpose/right_endpose",),
                },
            )
            gripper_by_arm = _read_arm_arrays(
                handle,
                index,
                {
                    "left": ("endpose/left_gripper", "joint_action/left_gripper"),
                    "right": ("endpose/right_gripper", "joint_action/right_gripper"),
                },
            )
            state_vector = build_rmbench_state_vector(endpose_by_arm, gripper_by_arm)

        return RMBenchFrame(
            tau=index,
            instruction=self.instruction,
            images_by_view=images_by_view,
            joint_action=joint_action,
            endpose_by_arm=endpose_by_arm,
            gripper_by_arm=gripper_by_arm,
            state_vector=state_vector,
        )

    def read_future_actions(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise IndexError(f"invalid action slice [{start}, {end}) for episode length {self.length}")
        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as handle:
            return np.asarray(handle[self.action_key][start:end], dtype=np.float32)

    def _validate_camera_lengths(self, handle: Any) -> None:
        for camera_name in self.camera_names:
            key = _camera_rgb_key(camera_name)
            if key not in handle:
                raise KeyError(f"camera {camera_name!r} is missing from {self.hdf5_path}: expected {key!r}")
            camera_length = int(handle[key].shape[0])
            if camera_length != self.length:
                raise ValueError(
                    f"camera {camera_name!r} length {camera_length} does not match action length {self.length}"
                )


def iter_rmbench_episode_files(
    rmbench_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    setting: str = DEFAULT_RMBENCH_SETTING,
) -> Iterator[RMBenchEpisodeFile]:
    root = Path(rmbench_root).expanduser()
    data_root = root / "data"
    task_names = tuple(tasks) if tasks is not None else _discover_tasks(data_root)
    for task_name in task_names:
        task_root = data_root / task_name / setting
        hdf5_dir = task_root / "data"
        instruction_dir = task_root / "instructions"
        for hdf5_path in sorted(hdf5_dir.glob("*.hdf5")):
            instruction_path = instruction_dir / f"{hdf5_path.stem}.json"
            yield RMBenchEpisodeFile(
                task_name=task_name,
                hdf5_path=hdf5_path,
                instruction_path=instruction_path if instruction_path.exists() else None,
            )


def read_rmbench_state_action_arrays(
    hdf5_path: str | Path,
    *,
    action_key: str = DEFAULT_RMBENCH_ACTION_KEY,
) -> RMBenchStateActionArrays:
    h5py = _require_h5py()
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        actions = np.asarray(handle[action_key], dtype=np.float32)
        endpose_by_arm = {
            "left": np.asarray(handle["endpose/left_endpose"], dtype=np.float32),
            "right": np.asarray(handle["endpose/right_endpose"], dtype=np.float32),
        }
        gripper_by_arm = {
            arm_name: np.asarray(handle[key], dtype=np.float32)
            for arm_name, key in {
                "left": "endpose/left_gripper",
                "right": "endpose/right_gripper",
            }.items()
            if key in handle
        }

    states = build_rmbench_state_matrix(endpose_by_arm, gripper_by_arm)
    if states.shape[0] != actions.shape[0]:
        raise ValueError(f"state length {states.shape[0]} does not match action length {actions.shape[0]}")
    return RMBenchStateActionArrays(states=states, actions=actions)


def compute_rmbench_normalization_stats(
    rmbench_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    setting: str = DEFAULT_RMBENCH_SETTING,
    max_episodes_per_task: int | None = None,
    robot_key: str = DEFAULT_RMBENCH_ROBOT_KEY,
    action_key: str = DEFAULT_RMBENCH_ACTION_KEY,
) -> dict[str, Any]:
    return dict(
        compute_rmbench_normalization_result(
            rmbench_root,
            tasks=tasks,
            setting=setting,
            max_episodes_per_task=max_episodes_per_task,
            robot_key=robot_key,
            action_key=action_key,
        ).stats
    )


def compute_rmbench_normalization_result(
    rmbench_root: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    setting: str = DEFAULT_RMBENCH_SETTING,
    max_episodes_per_task: int | None = None,
    robot_key: str = DEFAULT_RMBENCH_ROBOT_KEY,
    action_key: str = DEFAULT_RMBENCH_ACTION_KEY,
) -> RMBenchNormalizationResult:
    if max_episodes_per_task is not None and int(max_episodes_per_task) <= 0:
        raise ValueError("max_episodes_per_task must be positive when provided")

    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    episode_count = 0
    frame_count = 0
    task_counts: dict[str, int] = {}

    for episode in iter_rmbench_episode_files(rmbench_root, tasks=tasks, setting=setting):
        if max_episodes_per_task is not None and task_counts.get(episode.task_name, 0) >= int(max_episodes_per_task):
            continue
        arrays = read_rmbench_state_action_arrays(episode.hdf5_path, action_key=action_key)
        if arrays.states.size == 0 or arrays.actions.size == 0:
            continue
        state_min = _running_min(state_min, arrays.states)
        state_max = _running_max(state_max, arrays.states)
        action_min = _running_min(action_min, arrays.actions)
        action_max = _running_max(action_max, arrays.actions)
        episode_count += 1
        frame_count += int(arrays.actions.shape[0])
        task_counts[episode.task_name] = task_counts.get(episode.task_name, 0) + 1

    if episode_count == 0 or state_min is None or state_max is None or action_min is None or action_max is None:
        raise ValueError("no RMBench episodes were found for normalization stats")

    stats = {
        robot_key: {
            "observation.state": {
                "min": state_min.astype(float).tolist(),
                "max": state_max.astype(float).tolist(),
            },
            "action": {
                "min": action_min.astype(float).tolist(),
                "max": action_max.astype(float).tolist(),
            },
        },
    }
    metadata = {
        "benchmark": "RMBench",
        "setting": setting,
        "tasks": sorted(task_counts),
        "episodes": episode_count,
        "frames": frame_count,
        "action_key": action_key,
        "state_dim": int(state_min.shape[0]),
        "action_dim": int(action_min.shape[0]),
        "max_episodes_per_task": max_episodes_per_task,
    }
    return RMBenchNormalizationResult(stats=stats, metadata=metadata)


def read_rmbench_instruction(path: str | Path | None) -> str:
    if path is None:
        return ""
    instruction_path = Path(path).expanduser()
    if not instruction_path.exists():
        return ""
    payload = json.loads(instruction_path.read_text(encoding="utf-8"))
    return _first_text(payload, preferred_keys=("seen", "instruction", "prompt", "language", "text", "unseen"))


def decode_rmbench_rgb(value: Any) -> Image.Image:
    array = np.asarray(value)
    if array.ndim == 3:
        return Image.fromarray(array.astype(np.uint8), mode="RGB").convert("RGB")

    if isinstance(value, np.void):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray, np.bytes_)):
        raw = bytes(value)
    elif isinstance(value, np.ndarray):
        if value.dtype == object and value.shape == ():
            return decode_rmbench_rgb(value.item())
        raw = bytes(value.astype(np.uint8).reshape(-1))
    else:
        raw = bytes(value)

    with Image.open(BytesIO(raw)) as image:
        return image.convert("RGB")


def build_rmbench_state_vector(
    endpose_by_arm: Mapping[str, np.ndarray],
    gripper_by_arm: Mapping[str, np.ndarray],
    *,
    arm_order: Sequence[str] = ("left", "right"),
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for arm_name in arm_order:
        endpose = endpose_by_arm.get(arm_name)
        if endpose is not None:
            parts.append(np.asarray(endpose, dtype=np.float32).reshape(-1))
        gripper = gripper_by_arm.get(arm_name)
        if gripper is not None:
            parts.append(np.asarray(gripper, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def build_rmbench_state_matrix(
    endpose_by_arm: Mapping[str, np.ndarray],
    gripper_by_arm: Mapping[str, np.ndarray],
    *,
    arm_order: Sequence[str] = ("left", "right"),
) -> np.ndarray:
    parts: list[np.ndarray] = []
    expected_length: int | None = None
    for arm_name in arm_order:
        endpose = endpose_by_arm.get(arm_name)
        if endpose is not None:
            endpose_array = np.asarray(endpose, dtype=np.float32)
            expected_length = _validate_time_length(expected_length, endpose_array, arm_name)
            parts.append(endpose_array.reshape(endpose_array.shape[0], -1))
        gripper = gripper_by_arm.get(arm_name)
        if gripper is not None:
            gripper_array = np.asarray(gripper, dtype=np.float32)
            expected_length = _validate_time_length(expected_length, gripper_array, arm_name)
            parts.append(gripper_array.reshape(gripper_array.shape[0], -1))
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _resolve_instruction_path(hdf5_path: Path, instruction_path: str | Path | None) -> Path | None:
    if instruction_path is not None:
        return Path(instruction_path).expanduser()
    candidate = hdf5_path.parent.parent / "instructions" / f"{hdf5_path.stem}.json"
    return candidate if candidate.exists() else None


def _discover_tasks(data_root: Path) -> tuple[str, ...]:
    if not data_root.exists():
        return ()
    task_names = [path.name for path in data_root.iterdir() if path.is_dir()]
    return tuple(sorted(task_names))


def _camera_rgb_key(camera_name: str) -> str:
    if camera_name == "third_view_rgb":
        return "third_view_rgb"
    return f"observation/{camera_name}/rgb"


def _dataset_length(handle: Any, key: str) -> int:
    if key not in handle:
        raise KeyError(f"dataset key {key!r} is missing")
    shape = handle[key].shape
    if not shape:
        raise ValueError(f"dataset key {key!r} must have a time dimension")
    return int(shape[0])


def _read_arm_arrays(handle: Any, index: int, key_aliases_by_arm: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    arrays = {}
    for arm_name, aliases in key_aliases_by_arm.items():
        for key in aliases:
            if key in handle:
                arrays[arm_name] = np.asarray(handle[key][index], dtype=np.float32).reshape(-1)
                break
    return arrays


def _running_min(current: np.ndarray | None, values: np.ndarray) -> np.ndarray:
    values_min = np.asarray(values, dtype=np.float32).min(axis=0)
    return values_min if current is None else np.minimum(current, values_min)


def _running_max(current: np.ndarray | None, values: np.ndarray) -> np.ndarray:
    values_max = np.asarray(values, dtype=np.float32).max(axis=0)
    return values_max if current is None else np.maximum(current, values_max)


def _validate_time_length(expected: int | None, values: np.ndarray, name: str) -> int:
    if values.ndim < 1:
        raise ValueError(f"{name} array must have a time dimension")
    length = int(values.shape[0])
    if expected is not None and length != expected:
        raise ValueError(f"{name} length {length} does not match expected length {expected}")
    return length


def _first_text(value: Any, *, preferred_keys: Sequence[str]) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            text = _first_text(item, preferred_keys=preferred_keys)
            if text:
                return text
        return ""
    if isinstance(value, Mapping):
        for key in preferred_keys:
            if key in value:
                text = _first_text(value[key], preferred_keys=preferred_keys)
                if text:
                    return text
        for item in value.values():
            text = _first_text(item, preferred_keys=preferred_keys)
            if text:
                return text
    return ""


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RMBenchEpisodeReader requires h5py to read HDF5 episodes") from exc
    return h5py
