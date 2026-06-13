from pathlib import Path

import pytest

from himem_bridge_vla.reproducibility import (
    build_environment_metadata,
    build_reproducibility_metadata,
    build_torch_generator,
    write_experiment_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_metadata_uses_project_relative_paths(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)

    metadata = build_reproducibility_metadata({"repo_root": ".", "seed": 42})

    assert metadata["repo_root"] == "."
    assert metadata["cwd"] == "."
    assert metadata["git"]["commit"]
    assert metadata["git"]["branch"] == "main"
    assert metadata["seed"] == 42
    assert "numpy" in metadata["environment"]["packages"]


def test_environment_metadata_records_safe_relative_environment(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("HF_HOME", "run_outputs/hf-home")
    monkeypatch.setenv("HIMEM_TOKEN", "secret")

    metadata = build_environment_metadata(".")

    assert metadata["env"]["HF_HOME"] == "run_outputs/hf-home"
    assert "HIMEM_TOKEN" not in metadata["env"]
    assert metadata["torch"]["available"] in {True, False}


def test_write_experiment_snapshot_writes_environment_json(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    save_dir = tmp_path / "snapshot"

    write_experiment_snapshot(save_dir, {"repo_root": ".", "dataset_config_path": "configs/datasets/calvin.yaml"})

    assert (save_dir / "resolved_config.json").exists()
    assert (save_dir / "reproducibility.json").exists()
    assert (save_dir / "environment.json").exists()


def test_build_torch_generator_uses_requested_seed():
    torch = pytest.importorskip("torch")

    generator = build_torch_generator(123)

    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 123
