from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from himem_bridge_vla.dataset.memory_replay_frames import MemoryReplayFrameSample, ReplayFrame
from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.training.stage2.libero.cli import build_arg_parser
from himem_bridge_vla.training.stage2.libero.config import build_stage2_config
from himem_bridge_vla.training.stage2.libero.validators import enforce_stage2_contract


REPO_ROOT = find_repo_root(__file__)
TRAIN_STAGE2_SCRIPT = REPO_ROOT / "scripts" / "train" / "stage2" / "libero.py"
STAGE2_CONFIG = "configs/training/stage2/libero_10_full_e2e_from_stage1_best.yaml"


def test_train_stage2_help_does_not_require_training_runtime():
    result = subprocess.run(
        [sys.executable, str(TRAIN_STAGE2_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--sequence_len" in result.stdout
    assert "--finetune_progress_planner" in result.stdout


def test_train_stage2_script_is_thin_launcher():
    source = TRAIN_STAGE2_SCRIPT.read_text(encoding="utf-8")

    assert "from himem_bridge_vla.cli.train.stage2.libero import main" in source
    assert "build_stage2_config" not in source
    assert "train_stage2(" not in source


def test_stage2_module_help_does_not_require_training_runtime():
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "himem_bridge_vla.training.stage2.libero.cli", "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--stage2_sampling_mode" in result.stdout
    assert "--reset_best_loss_on_resume" in result.stdout


def test_build_stage2_config_uses_full_e2e_profile_and_cli_override():
    args = build_arg_parser().parse_args(["--config", STAGE2_CONFIG, "--max_steps", "3000"])
    config = build_stage2_config(args, repo_root=REPO_ROOT)

    assert config["dataset_type"] == "libero_raw_episode"
    assert config["load_vlm"] is True
    assert config["finetune_vlm"] is True
    assert config["finetune_action_head"] is True
    assert config["progress_planner_enabled"] is True
    assert config["finetune_progress_planner"] is True
    assert config["enable_bridge_aux_loss"] is False
    assert config["sequence_len"] == 16
    assert config["stage2_sampling_mode"] == "group"
    assert config["max_steps"] == 3000
    assert config["resume_pretrain"] is True
    assert config["reset_best_loss_on_resume"] is False
    assert config.get("min_cuda_memory_gb") is None
    assert config["loss"] == {
        "action_fm": 1.0,
        "vlm_ce": 0.0,
        "planner_aux": 0.0,
        "gripper_bce": 0.0,
    }


def test_enforce_stage2_contract_rejects_token_cache_dataset():
    config = _minimal_stage2_config()
    config["dataset_type"] = "memory_token_cache"

    with pytest.raises(ValueError, match="dataset_type=libero_raw_episode"):
        enforce_stage2_contract(config)


def test_enforce_stage2_contract_rejects_frozen_progress_planner():
    config = _minimal_stage2_config()
    config["finetune_progress_planner"] = False

    with pytest.raises(ValueError, match="finetune_progress_planner=true"):
        enforce_stage2_contract(config)


def test_enforce_stage2_contract_rejects_cuda_memory_floor():
    config = _minimal_stage2_config()
    config["min_cuda_memory_gb"] = 21.0

    with pytest.raises(ValueError, match="real training workload"):
        enforce_stage2_contract(config)


def test_raw_episode_sequence_dataset_samples_sorted_16_steps_and_collates(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from himem_bridge_vla.training.stage2.common import dataset as stage2_dataset

    monkeypatch.setattr(stage2_dataset, "MemoryReplayFrameReader", FakeMemoryReplayFrameReader)
    index_path = _write_stage2_replay_index(tmp_path)
    normalization_path = _write_normalization_source(tmp_path)
    dataset = stage2_dataset.LiberoRawEpisodeSequenceDataset(
        index_path,
        sequence_len=16,
        action_horizon=4,
        action_dim=7,
        state_dim=8,
        normalization=stage2_dataset.load_stage2_normalization(normalization_path),
    )

    item = dataset[0]

    sampled_steps = item["sampled_steps"]
    assert len(sampled_steps) == 16
    assert sampled_steps == sorted(sampled_steps)
    assert min(sampled_steps) >= 0
    assert max(sampled_steps) <= 16
    assert len(item["steps"]) == 16
    first_step = item["steps"][0]
    assert first_step["states"].shape == (8,)
    assert first_step["actions"].shape == (4, 7)
    assert first_step["action_mask"].shape == (4, 7)
    assert first_step["executed_actions"].shape == (4, 7)
    assert torch.all(first_step["actions"] <= 1.0)
    assert torch.all(first_step["actions"] >= -1.0)

    batch = stage2_dataset.collate_libero_raw_episode_sequences([item])

    assert batch["batch_size"] == 1
    assert len(batch["trajectory_steps"]) == 16
    assert batch["trajectory_steps"][0]["states"].shape == (1, 8)
    assert batch["trajectory_steps"][0]["actions"].shape == (1, 4, 7)
    assert batch["trajectory_steps"][0]["loss_mask"].tolist() == [True]


def test_stage2_checkpoint_load_allows_missing_embedder_only(tmp_path):
    torch = pytest.importorskip("torch")
    from himem_bridge_vla.training.stage2.common.loop import load_stage2_training_checkpoint

    source = TinyStage2Model(torch)
    target = TinyStage2Model(torch)
    with torch.no_grad():
        source.head.weight.fill_(0.25)
        source.head.bias.fill_(0.5)
        target.head.weight.zero_()
        target.head.bias.zero_()
    state = {key: value.clone() for key, value in source.state_dict().items() if not key.startswith("embedder.")}
    _write_checkpoint(torch, tmp_path, state, next_step=4970)

    step, client_state = load_stage2_training_checkpoint(
        torch,
        target,
        load_dir=str(tmp_path),
        accelerator=FakeAccelerator(torch),
        tag="step_best",
        optimizer=None,
        load_optimizer_states=False,
        resume_pretrain=True,
    )

    assert step == 4970
    assert client_state["next_step"] == 4970
    assert torch.allclose(target.head.weight, source.head.weight)
    assert torch.allclose(target.head.bias, source.head.bias)


def test_stage2_resume_best_loss_can_reset_stage1_selection():
    from himem_bridge_vla.training.stage2.common.loop import _resume_best_loss

    client_state = {"best_loss": 0.0167}

    assert _resume_best_loss({"resume_pretrain": True}, client_state) == float("inf")
    assert _resume_best_loss({"reset_best_loss_on_resume": True}, client_state) == float("inf")
    assert _resume_best_loss({"reset_best_loss_on_resume": False}, client_state) == 0.0167


def test_stage2_episode_group_batches_active_rows_per_timestep():
    torch = pytest.importorskip("torch")
    from himem_bridge_vla.model.internvl3.internvl3_embedder import InternVL3EmbeddingOutput
    from himem_bridge_vla.model.planner.progress_state import ProgressState
    from himem_bridge_vla.training.stage2.common.loop import _run_stage2_episode_group_batch

    class FakePlanner:
        def initial_state(self, batch_size, *, device, dtype):
            return ProgressState(
                completed_events=torch.zeros(batch_size, 4, device=device, dtype=dtype),
                current_stage=torch.ones(batch_size, 4, device=device, dtype=dtype),
            )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.progress_state_planner = FakePlanner()
            self.use_direct_bridge = False
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.forward_batch_sizes = []
            self.last_progress_planner_output = None

        def get_vl_embeddings(self, *, images, image_mask, prompt, return_cls_only, return_hidden_states):
            del images, image_mask, prompt, return_cls_only, return_hidden_states
            base = self.weight.reshape(1, 1, 1)
            summary_base = self.weight.reshape(1, 1)
            return InternVL3EmbeddingOutput(
                fused_tokens=base.expand(1, 3, 4),
                hidden_states=[base.expand(1, 5, 4)],
                attention_mask=torch.ones(1, 5, dtype=torch.bool),
                visual_tokens=base.expand(1, 3, 4),
                planner_vl_summary=summary_base.expand(1, 4),
            )

        def forward(
            self,
            fused_tokens,
            *,
            state,
            actions_gt,
            action_mask,
            hidden_states,
            memory_context,
            memory_context_mask,
            short_memory_time_ids,
            executed_actions,
            executed_action_mask,
            planner_vl_summary,
            progress_state,
        ):
            del state, action_mask, hidden_states, memory_context, memory_context_mask
            del short_memory_time_ids, executed_actions, executed_action_mask, planner_vl_summary
            batch_size = int(fused_tokens.shape[0])
            self.forward_batch_sizes.append(batch_size)
            self.last_progress_planner_output = SimpleNamespace(
                progress_state=ProgressState(
                    completed_events=progress_state.completed_events + 1.0,
                    current_stage=progress_state.current_stage + 1.0,
                )
            )
            pred_velocity = self.weight.expand(batch_size, actions_gt.numel() // batch_size)
            noise = torch.zeros_like(actions_gt)
            return pred_velocity, noise

    model = FakeModel()
    batch = _stage2_group_batch(torch, batch_size=2, sequence_len=3, horizon=2, action_dim=3)

    loss, metrics, _last_tensors = _run_stage2_episode_group_batch(
        torch=torch,
        model=model,
        unwrapped_model=model,
        batch=batch,
        accelerator=FakeAccelerator(torch),
        backward_fn=None,
    )

    assert model.forward_batch_sizes == [2, 2, 2]
    assert metrics["stage2_sequence_len"] == 3.0
    assert metrics["stage2_loss_terms"] == 3.0
    assert metrics["stage2_active_samples"] == 6.0
    assert metrics["stage2_batch_rows_max"] == 2.0
    assert loss.item() >= 0.0


def test_stage2_checkpoint_load_rejects_missing_non_vlm_key(tmp_path):
    torch = pytest.importorskip("torch")
    from himem_bridge_vla.training.stage2.common.loop import load_stage2_training_checkpoint

    model = TinyStage2Model(torch)
    state = {key: value.clone() for key, value in model.state_dict().items() if key.startswith("embedder.")}
    _write_checkpoint(torch, tmp_path, state, next_step=1)

    with pytest.raises(RuntimeError, match="missing non-VLM keys"):
        load_stage2_training_checkpoint(
            torch,
            model,
            load_dir=str(tmp_path),
            accelerator=FakeAccelerator(torch),
            tag="step_best",
            optimizer=None,
            load_optimizer_states=False,
            resume_pretrain=True,
        )


class FakeMemoryReplayFrameReader:
    def __init__(self, **_kwargs):
        pass

    def read(self, row):
        current_step = int(row["current_step"])
        current = ReplayFrame(
            tau=current_step,
            images_by_view=_fake_images(current_step),
            state_vector=np.full(8, float(current_step), dtype=np.float32),
        )
        short_frames = tuple(
            None
            if step is None
            else ReplayFrame(
                tau=int(step),
                images_by_view=_fake_images(int(step)),
                state_vector=np.full(8, float(step), dtype=np.float32),
            )
            for step in row["short_steps"]
        )
        stride = int(row["executed_action_stride"])
        valid_executed = int(row["executed_action_end"]) - int(row["executed_action_start"])
        executed_actions = np.zeros((stride, 7), dtype=np.float32)
        executed_action_mask = np.zeros((stride,), dtype=bool)
        if valid_executed > 0:
            executed_actions[-valid_executed:] = float(current_step)
            executed_action_mask[-valid_executed:] = True
        future_count = int(row["action_end"]) - int(row["action_start"])
        future_actions = np.full((future_count, 7), float(current_step), dtype=np.float32)
        return MemoryReplayFrameSample(
            benchmark="LIBERO",
            episode_id=str(row["episode_id"]),
            prompt="reader prompt",
            current_step=current_step,
            current=current,
            short_frames=short_frames,
            short_mask=tuple(bool(value) for value in row["short_mask"]),
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            future_actions=future_actions,
            action_valid_count=int(row["action_valid_count"]),
        )


class FakeAccelerator:
    def __init__(self, torch):
        self.device = torch.device("cpu")
        self.is_main_process = True

    def unwrap_model(self, model):
        return model


class TinyStage2Model:
    def __init__(self, torch):
        self.module = torch.nn.Module()
        self.module.embedder = torch.nn.Linear(2, 2)
        self.module.head = torch.nn.Linear(2, 1)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)

    @property
    def head(self):
        return self.module.head


def _fake_images(step: int):
    color = int(step) % 255
    return {
        "agentview_rgb": Image.new("RGB", (4, 4), color=(color, 0, 0)),
        "eye_in_hand_rgb": Image.new("RGB", (4, 4), color=(0, color, 0)),
    }


def _write_stage2_replay_index(tmp_path: Path) -> Path:
    path = tmp_path / "episode_replay.json"
    payload = {
        "format": "libero_episode_replay_index",
        "benchmark": "LIBERO",
        "libero_root": ".",
        "action_horizon": 4,
        "short_offsets": [4, 2],
        "executed_action_stride": 4,
        "episodes": [
            {
                "episode_id": "libero_10:task:demo_0",
                "episode_key": "demo_0",
                "source_path": "libero_10/task.hdf5",
                "episode_length": 20,
                "prompt": "do the task",
                "task_name": "task",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_normalization_source(tmp_path: Path) -> Path:
    path = tmp_path / "norm_manifest.json"
    payload = {
        "normalization": {
            "enabled": True,
            "type": "train_split_minmax_to_minus_one_one",
            "robot_key": "libero",
            "clip_after_normalization": True,
            "stats": {
                "libero": {
                    "observation.state": {
                        "min": [0.0] * 8,
                        "max": [100.0] * 8,
                    },
                    "action": {
                        "min": [0.0] * 7,
                        "max": [100.0] * 7,
                    },
                }
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_checkpoint(torch, root: Path, model_state: dict, *, next_step: int) -> None:
    checkpoint_dir = root / "step_best"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "format": "stage1_torch_checkpoint",
            "model_state_dict": model_state,
            "client_state": {"next_step": next_step},
        },
        checkpoint_dir / "model.pt",
    )


def _stage2_group_batch(torch, *, batch_size: int, sequence_len: int, horizon: int, action_dim: int) -> dict:
    trajectory_steps = []
    for step in range(sequence_len):
        trajectory_steps.append(
            {
                "batch_indices": torch.arange(batch_size, dtype=torch.long),
                "loss_mask": torch.ones(batch_size, dtype=torch.bool),
                "images": [[object()] for _ in range(batch_size)],
                "image_mask": torch.ones(batch_size, 1, dtype=torch.bool),
                "prompts": [f"prompt {index}" for index in range(batch_size)],
                "states": torch.full((batch_size, 8), float(step), dtype=torch.float32),
                "actions": torch.ones(batch_size, horizon, action_dim, dtype=torch.float32),
                "action_mask": torch.ones(batch_size, horizon, action_dim, dtype=torch.bool),
                "short_images": [tuple() for _ in range(batch_size)],
                "short_image_masks": [tuple() for _ in range(batch_size)],
                "executed_actions": torch.zeros(batch_size, horizon, action_dim, dtype=torch.float32),
                "executed_action_mask": torch.zeros(batch_size, horizon, dtype=torch.bool),
                "current_steps": torch.full((batch_size,), step, dtype=torch.long),
            }
        )
    return {
        "batch_size": batch_size,
        "episode_ids": [f"episode_{index}" for index in range(batch_size)],
        "sampled_steps": [list(range(sequence_len)) for _ in range(batch_size)],
        "trajectory_steps": trajectory_steps,
    }


def _minimal_stage2_config() -> dict:
    return {
        "dataset_type": "libero_raw_episode",
        "load_vlm": True,
        "finetune_vlm": True,
        "finetune_action_head": True,
        "progress_planner_enabled": True,
        "finetune_progress_planner": True,
        "enable_bridge_aux_loss": False,
        "memory_token_cache_sequence_training": False,
        "sequence_len": 16,
        "stage2_sampling_mode": "group",
        "loss": {
            "action_fm": 1.0,
            "vlm_ce": 0.0,
            "planner_aux": 0.0,
            "gripper_bce": 0.0,
        },
    }
