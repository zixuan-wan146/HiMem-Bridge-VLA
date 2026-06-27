from himem_bridge_vla.path_utils import find_repo_root

from himem_bridge_vla.experiment_config import resolve_experiment_config


def test_resolve_experiment_config_loads_bridge_yaml_and_sets_seed():
    config_path = (
        find_repo_root(__file__)
        / "configs"
        / "experiments"
        / "bridge_himem"
        / "crosskv_clean.yaml"
    )

    resolved = resolve_experiment_config({"bridge_himem_config": str(config_path), "seed": None})

    assert resolved["experiment_config_resolved"] is True
    assert resolved["seed"] == 42
    assert resolved["use_bridge"] is True
    assert resolved["use_memory"] is True
    assert resolved["bridge_variant"] == "crosskv"
    assert resolved["bridge_himem"]["experiment_name"] == "crosskv_clean"
    assert resolved["bridge_himem_config_path"] == str(config_path)


def test_resolve_experiment_config_cli_seed_overrides_yaml_seed():
    config_path = (
        find_repo_root(__file__)
        / "configs"
        / "experiments"
        / "bridge_himem"
        / "crosskv_clean.yaml"
    )

    resolved = resolve_experiment_config({"bridge_himem_config": str(config_path), "seed": 7})

    assert resolved["seed"] == 7


def test_resolve_experiment_config_is_idempotent():
    resolved = resolve_experiment_config({"seed": 3})

    assert resolve_experiment_config(resolved) == resolved
