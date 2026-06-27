from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def test_train_help_does_not_require_training_runtime():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--dataset_config_path" in result.stdout


def test_build_training_config_uses_defaults_and_cli_overrides():
    train_script = _load_train_script()
    args = train_script.build_arg_parser().parse_args(
        [
            "--batch_size",
            "3",
            "--dataset_config_path",
            "configs/datasets/simulation.yaml",
            "--save_dir",
            "run_outputs/training/from_cli",
        ]
    )

    config = train_script.build_training_config(args)

    assert config["batch_size"] == 3
    assert config["dataset_config_path"] == "configs/datasets/simulation.yaml"
    assert config["save_dir"] == "run_outputs/training/from_cli"
    assert config["num_inference_timesteps"] == 15
    assert config["inference_tau_schedule"] == "midpoint"
    assert config["avoid_endpoint_tau"] is True


def test_build_param_groups_rejects_model_with_no_trainable_parameters():
    train_script = _load_train_script()

    with pytest.raises(ValueError, match="No trainable parameters"):
        train_script.build_param_groups(_FrozenModel(), wd=0.01)


def test_build_param_groups_applies_stage1_lr_and_no_decay_rules():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = torch.nn.Module()
    model.action_head = torch.nn.Module()
    model.action_head.plan_gate_logits = torch.nn.Parameter(torch.zeros(2))
    model.action_head.action_encoder = torch.nn.Linear(3, 4)
    model.action_head.short_memory_adapter = torch.nn.Linear(4, 4)

    groups = train_script.build_param_groups(
        model,
        wd=0.001,
        base_lr=5e-5,
        lr_groups={
            "gates": 5e-5,
            "noisy_action_encoder": 1e-4,
            "short_memory_projector": 1e-4,
        },
    )

    by_name = {group.get("name", ""): group for group in groups}

    assert by_name["gates.no_decay"]["lr"] == 5e-5
    assert by_name["gates.no_decay"]["weight_decay"] == 0.0
    assert by_name["noisy_action_encoder.decay"]["lr"] == 1e-4
    assert by_name["short_memory_projector.decay"]["lr"] == 1e-4


def test_build_param_groups_keeps_action_expert_separate_from_bridge_attention():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = torch.nn.Module()
    model.action_head = torch.nn.Module()
    model.action_head.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.action_head.transformer_blocks[0].self_attn = torch.nn.Linear(4, 4)
    model.action_head.transformer_blocks[0].visual_attn = torch.nn.Linear(4, 4)
    model.action_head.transformer_blocks[0].action_attn = torch.nn.Linear(4, 4)

    groups = train_script.build_param_groups(
        model,
        wd=0.001,
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


def test_lr_lambda_honors_min_lr_ratio():
    train_script = _load_train_script()
    schedule = train_script.get_lr_lambda(warmup_steps=0, total_steps=10, min_lr_ratio=0.1)

    assert schedule(10) == pytest.approx(0.1)


def test_train_rejects_stage1_trajectory_window_route(monkeypatch):
    train_script = _load_train_script()
    monkeypatch.setattr(train_script, "_ensure_training_runtime", lambda: None)
    monkeypatch.setattr(train_script, "resolve_experiment_config", lambda config: dict(config))
    monkeypatch.setattr(train_script, "resolve_training_config_paths", lambda config, _repo_root: dict(config))
    monkeypatch.setattr(train_script, "validate_training_config", lambda config, **_kwargs: None)

    with pytest.raises(ValueError, match="scripts/train_stage1.py"):
        train_script.train({"memory_token_cache_sequence_training": True})


def test_unwrap_training_model_uses_accelerator_when_available(monkeypatch):
    train_script = _load_train_script()
    wrapper = object()
    unwrapped = object()

    class FakeAccelerator:
        def unwrap_model(self, model):
            assert model is wrapper
            return unwrapped

    monkeypatch.setattr(train_script, "accelerator", FakeAccelerator())

    assert train_script.unwrap_training_model(wrapper) is unwrapped


def test_validate_batch_image_masks_rejects_empty_sample():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()
    train_script.torch = torch

    image_masks = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="no active image"):
        train_script.validate_batch_image_masks(image_masks, step=3)


def test_compute_bridge_auxiliary_loss_uses_boundary_and_progress_labels():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()

    model = _ModelWithBridgeOutput(
        boundary_logits=torch.tensor([[0.0], [2.0]]),
        progress_logits=torch.tensor([[0.0], [2.0]]),
    )
    batch = {
        "boundary": torch.tensor([0.0, 1.0]),
        "progress": torch.tensor([0.5, 1.0]),
    }

    loss, metrics = train_script.compute_bridge_auxiliary_loss(
        model,
        batch,
        {"enable_bridge_aux_loss": True, "boundary_loss_weight": 1.0, "progress_loss_weight": 0.2},
    )

    assert loss is not None
    assert loss.item() > 0
    assert "boundary_loss" in metrics
    assert "progress_loss" in metrics
    assert "bridge_aux_loss" not in metrics


def test_custom_collate_includes_action_segment_targets():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()
    train_script.torch = torch

    item = {
        "prompt": "pick",
        "images": torch.zeros(1, 3, 4, 4),
        "state": torch.zeros(2),
        "action": torch.zeros(2, 2),
        "action_mask": torch.ones(2, 2, dtype=torch.bool),
        "image_mask": torch.ones(1, dtype=torch.bool),
        "embodiment_id": torch.tensor(0),
        "planner_prompt": "pick",
        "planner_images": torch.zeros(1, 3, 4, 4),
        "planner_image_mask": torch.ones(1, dtype=torch.bool),
        "planner_state": torch.zeros(2),
        "action_segments": torch.zeros(3, 4, 2),
        "action_segment_mask": torch.tensor([True, True, False]),
    }

    batch = train_script.custom_collate_fn([item, item])

    assert tuple(batch["action_segments"].shape) == (2, 3, 4, 2)
    assert tuple(batch["action_segment_mask"].shape) == (2, 3)
    assert tuple(batch["planner_states"].shape) == (2, 2)
    assert "plan_active_mask" not in batch


def test_custom_collate_includes_direct_bridge_optional_tensors():
    torch = pytest.importorskip("torch")
    train_script = _load_train_script()
    train_script.torch = torch

    item = {
        "prompt": "pick",
        "images": torch.zeros(1, 3, 4, 4),
        "state": torch.zeros(2),
        "action": torch.zeros(2, 2),
        "action_mask": torch.ones(2, 2, dtype=torch.bool),
        "image_mask": torch.ones(1, dtype=torch.bool),
        "embodiment_id": torch.tensor(0),
        "memory_context": torch.zeros(4, 8),
        "memory_context_mask": torch.ones(4, dtype=torch.bool),
        "short_memory_time_ids": torch.tensor([0, 0, 1, 1]),
        "plan_token_mask": torch.ones(1, dtype=torch.bool),
    }

    batch = train_script.custom_collate_fn([item, item])

    assert tuple(batch["memory_context"].shape) == (2, 4, 8)
    assert tuple(batch["memory_context_mask"].shape) == (2, 4)
    assert batch["short_memory_time_ids"].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert tuple(batch["plan_token_mask"].shape) == (2, 1)


class _FrozenParam:
    requires_grad = False

    def dim(self) -> int:
        return 2


class _FrozenModel:
    def named_parameters(self):
        return [("weight", _FrozenParam())]


class _BridgeOutput:
    def __init__(self, *, boundary_logits, progress_logits):
        self.boundary_logits = boundary_logits
        self.progress_logits = progress_logits


class _ModelWithBridgeOutput:
    def __init__(self, *, boundary_logits, progress_logits):
        self.last_bridge_output = _BridgeOutput(
            boundary_logits=boundary_logits,
            progress_logits=progress_logits,
        )


def _load_train_script():
    spec = importlib.util.spec_from_file_location("himem_train_script_for_tests", TRAIN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
