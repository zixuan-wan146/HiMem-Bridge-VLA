from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import subprocess
import sys
import os

import pytest

from himem_bridge_vla.training.stage1.common.batch_contract import validate_stage1_window_batch
from himem_bridge_vla.training.common.optimizer import build_param_groups
from himem_bridge_vla.training.common.scheduler import get_lr_lambda
from himem_bridge_vla.training.stage1.libero.cli import build_arg_parser
from himem_bridge_vla.training.stage1.libero.config import build_stage1_config
from himem_bridge_vla.training.stage1.common.loss import stage1_flow_matching_loss
from himem_bridge_vla.training.stage1.common.loop import _scatter_progress_state
from himem_bridge_vla.training.stage1.libero.validators import enforce_stage1_contract, validate_stage1_cache_contract


REPO_ROOT = find_repo_root(__file__)
TRAIN_STAGE1_SCRIPT = REPO_ROOT / "scripts" / "train" / "stage1" / "libero.py"


def test_train_stage1_help_does_not_require_training_runtime():
    result = subprocess.run(
        [sys.executable, str(TRAIN_STAGE1_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--burnin_replan_steps" in result.stdout


def test_train_stage1_script_is_thin_launcher():
    source = TRAIN_STAGE1_SCRIPT.read_text(encoding="utf-8")

    assert "from himem_bridge_vla.cli.train.stage1.libero import main" in source
    assert "build_stage1_config" not in source
    assert "train_stage1(" not in source


def test_stage1_module_help_does_not_require_training_runtime():
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    result = subprocess.run(
        [sys.executable, "-m", "himem_bridge_vla.training.stage1.libero.cli", "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--loss_replan_steps" in result.stdout


def test_build_stage1_config_uses_active_profile_and_cli_override():
    args = build_arg_parser().parse_args(
        [
            "--config",
            "configs/training/stage1/libero/libero_10_direct_progress_w4.yaml",
            "--batch_size",
            "2",
        ]
    )
    config = build_stage1_config(args, repo_root=REPO_ROOT)

    assert config["dataset_type"] == "memory_token_cache"
    assert config["memory_token_cache_sequence_training"] is True
    assert config["batch_size"] == 2
    assert config["horizon"] == 32
    assert config["progress_planner_replan_stride"] == 16
    assert config["shuffle_trajectory_windows"] is False
    assert config["num_inference_timesteps"] == 15
    assert config["progress_planner_checkpoint"]


def test_build_stage1_config_strict_mode_requires_cache_manifest():
    args = build_arg_parser().parse_args(
        [
            "--config",
            "configs/training/stage1/libero/libero_10_direct_progress_w4.yaml",
            "--dataset_config_path",
            "local_data/token_caches/missing_stage1_test/manifest.json",
        ]
    )

    with pytest.raises(FileNotFoundError, match="Dataset config file not found"):
        build_stage1_config(args, repo_root=REPO_ROOT, validate_external_artifacts=True)


def test_enforce_stage1_contract_rejects_frame_level_cache_training():
    config = _minimal_stage1_config()
    config["memory_token_cache_sequence_training"] = False

    with pytest.raises(ValueError, match="episode-level fixed-replan-node"):
        enforce_stage1_contract(config)


def test_enforce_stage1_contract_rejects_missing_progress_checkpoint():
    config = _minimal_stage1_config()
    config["progress_planner_checkpoint"] = None

    with pytest.raises(ValueError, match="progress_planner_checkpoint"):
        enforce_stage1_contract(config)


def test_validate_stage1_cache_contract_checks_hidden_layers(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "format": "libero_episode_feature_cache",
  "hidden_dim": 896,
  "hidden_state_layers": [3, 6],
  "node_count": 1,
  "planner_vl_summary": {"enabled": true, "source": "vlm_last_valid_token"},
  "normalization": {
    "robot_key": "libero",
    "stats": {
      "libero": {
        "action": {"max": [1, 1, 1, 1, 1, 1, 1]},
        "observation.state": {"max": [1, 1, 1, 1, 1, 1, 1, 1]}
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    config = _minimal_stage1_config()
    config.update({"dataset_config_path": "manifest.json", "per_action_dim": 7, "state_dim": 8})

    with pytest.raises(ValueError, match="hidden_state_layers"):
        validate_stage1_cache_contract(config, repo_root=tmp_path)


def test_validate_stage1_cache_contract_rejects_legacy_visual_token_cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "format": "memory_replay_visual_token_cache",
  "hidden_dim": 896,
  "hidden_state_layers": [3, 6, 9, 12],
  "hidden_state_cache_entries": 1,
  "planner_vl_summary": {"enabled": true, "source": "vlm_last_valid_token"}
}
""",
        encoding="utf-8",
    )
    config = _minimal_stage1_config()
    config.update({"dataset_config_path": "manifest.json", "per_action_dim": 7, "state_dim": 8})

    with pytest.raises(ValueError, match="libero_episode_feature_cache"):
        validate_stage1_cache_contract(config, repo_root=tmp_path)


def test_validate_stage1_cache_contract_checks_action_state_dims(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "format": "libero_episode_feature_cache",
  "hidden_dim": 896,
  "hidden_state_layers": [3, 6, 9, 12],
  "node_count": 1,
  "planner_vl_summary": {"enabled": true, "source": "vlm_last_valid_token"},
  "normalization": {
    "robot_key": "libero",
    "stats": {
      "libero": {
        "action": {"max": [1, 1]},
        "observation.state": {"max": [1, 1, 1]}
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    config = _minimal_stage1_config()
    config.update({"dataset_config_path": "manifest.json", "per_action_dim": 7, "state_dim": 8})

    with pytest.raises(ValueError, match="action dimension"):
        validate_stage1_cache_contract(config, repo_root=tmp_path)


def test_validate_stage1_cache_contract_requires_planner_vl_summary(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "format": "libero_episode_feature_cache",
  "hidden_dim": 896,
  "hidden_state_layers": [3, 6, 9, 12],
  "node_count": 1,
  "normalization": {
    "robot_key": "libero",
    "stats": {
      "libero": {
        "action": {"max": [1, 1, 1, 1, 1, 1, 1]},
        "observation.state": {"max": [1, 1, 1, 1, 1, 1, 1, 1]}
      }
    }
  }
}
""",
        encoding="utf-8",
    )
    config = _minimal_stage1_config()
    config.update({"dataset_config_path": "manifest.json", "per_action_dim": 7, "state_dim": 8})

    with pytest.raises(ValueError, match="planner_vl_summary"):
        validate_stage1_cache_contract(config, repo_root=tmp_path)


def test_stage1_lr_lambda_honors_min_ratio():
    schedule = get_lr_lambda(warmup_steps=0, total_steps=10, min_lr_ratio=0.1)

    assert schedule(10) == pytest.approx(0.1)


def test_stage1_batch_contract_rejects_frame_level_batch():
    with pytest.raises(ValueError, match="trajectory_steps"):
        validate_stage1_window_batch({"states": object(), "actions": object()})


def test_stage1_batch_contract_rejects_missing_step_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_stage1_window_batch({"batch_size": 1, "trajectory_steps": [{"states": object()}]})


def test_stage1_flow_matching_loss_validates_shape():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="pred_velocity shape"):
        stage1_flow_matching_loss(
            pred_velocity=torch.zeros(2, 3),
            noise=torch.zeros(2, 4),
            actions_gt=torch.ones(2, 4),
            action_mask=torch.ones(2, 4, dtype=torch.bool),
        )


def test_stage1_scatter_progress_state_aligns_dtype():
    torch = pytest.importorskip("torch")
    from himem_bridge_vla.model.planner import ProgressState

    progress_state = ProgressState(
        completed_events=torch.zeros(2, 1, 4, dtype=torch.bfloat16),
        current_stage=torch.zeros(2, 1, 4, dtype=torch.bfloat16),
    )
    updated_state = ProgressState(
        completed_events=torch.ones(1, 1, 4, dtype=torch.float32),
        current_stage=torch.ones(1, 1, 4, dtype=torch.float32),
    )

    scattered = _scatter_progress_state(progress_state, torch.tensor([1]), updated_state)

    assert scattered.completed_events.dtype == torch.bfloat16
    assert scattered.current_stage.dtype == torch.bfloat16
    assert scattered.completed_events[1].float().sum().item() == pytest.approx(4.0)


def test_stage1_param_groups_keep_action_expert_separate_from_bridge_attention():
    torch = pytest.importorskip("torch")

    model = torch.nn.Module()
    model.action_head = torch.nn.Module()
    model.action_head.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.action_head.transformer_blocks[0].self_attn = torch.nn.Linear(4, 4)
    model.action_head.transformer_blocks[0].visual_attn = torch.nn.Linear(4, 4)
    model.action_head.transformer_blocks[0].action_attn = torch.nn.Linear(4, 4)

    groups = build_param_groups(
        model,
        0.001,
        base_lr=5e-5,
        lr_groups={
            "action_expert": 5e-5,
            "bridge_attention": 1e-4,
            "flow_matching_action_head": 5e-5,
        },
    )
    named = {group.get("name", ""): group for group in groups}

    assert named["action_expert.decay"]["lr"] == 5e-5
    assert named["bridge_attention.decay"]["lr"] == 1e-4


def _minimal_stage1_config() -> dict:
    return {
        "dataset_type": "memory_token_cache",
        "memory_token_cache_sequence_training": True,
        "load_vlm": False,
        "finetune_vlm": False,
        "finetune_action_head": True,
        "finetune_progress_planner": False,
        "enable_bridge_aux_loss": False,
        "progress_planner_enabled": True,
        "progress_planner_checkpoint": "local_data/runs/progress_warmup/best.pt",
        "horizon": 32,
        "progress_planner_replan_stride": 16,
        "num_inference_timesteps": 15,
        "inference_tau_schedule": "midpoint",
        "avoid_endpoint_tau": True,
    }
