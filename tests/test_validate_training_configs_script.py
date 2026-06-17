import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_training_configs_script_loads_default_profiles():
    result = subprocess.run(
        [sys.executable, "scripts/validate_training_configs.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "validated 0 training config(s)" in result.stdout


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
        [sys.executable, "scripts/validate_training_configs.py", str(config_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "project-relative" in result.stderr
