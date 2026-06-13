import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_bridge_himem_configs_script_loads_default_configs():
    result = subprocess.run(
        ["python3", "scripts/validate_bridge_himem_configs.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "baseline_fused_only" in result.stdout
    assert "crosskv_clean" in result.stdout
    assert "mixed_latent_clean" in result.stdout
    assert "mixed_latent_skill" in result.stdout
