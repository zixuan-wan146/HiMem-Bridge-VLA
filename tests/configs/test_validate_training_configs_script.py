import subprocess
from himem_bridge_vla.path_utils import find_repo_root
from pathlib import Path
import sys


REPO_ROOT = find_repo_root(__file__)


def test_validate_training_configs_script_loads_default_profiles():
    result = subprocess.run(
        [sys.executable, "scripts/quality/validate_training_configs.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "configs/training/stage1/libero/libero_10_direct_progress_w4.yaml" in result.stdout


def test_validate_training_configs_script_rejects_absolute_profile_paths(tmp_path):
    config_path = tmp_path / "bad.yaml"
    absolute_dataset_path = Path("/").joinpath("datasets", "benchmark.yaml")
    config_path.write_text(
        "\n".join(
            [
                f"dataset_config_path: {absolute_dataset_path}",
                "max_steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/quality/validate_training_configs.py", str(config_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "project-relative" in result.stderr
