from __future__ import annotations

from himem_bridge_vla.benchmarks.libero.config import (
    DEFAULT_TASK_SUITES,
    LiberoClientConfig,
    align_max_steps,
    configure_mujoco_environment,
)


def test_default_config_matches_documented_smoke_server():
    config = LiberoClientConfig.from_env({})

    assert config.server_url == "ws://127.0.0.1:9000"
    assert config.task_suites == DEFAULT_TASK_SUITES
    assert config.max_steps == [25, 25, 25, 95]
    assert config.horizon == 32
    assert config.mujoco_gl == "osmesa"
    assert config.result_file == "./log_file/HiMem_libero_all_results.json"


def test_single_max_steps_value_expands_to_all_task_suites():
    config = LiberoClientConfig.from_env(
        {
            "HIMEM_LIBERO_TASK_SUITES": "libero_spatial,libero_goal",
            "HIMEM_LIBERO_MAX_STEPS": "7",
        }
    )

    assert config.task_suites == ["libero_spatial", "libero_goal"]
    assert config.max_steps == [7, 7]


def test_server_url_prefers_shared_env_var():
    config = LiberoClientConfig.from_env(
        {
            "HIMEM_SERVER_URI": "ws://server-uri:9000",
            "HIMEM_LIBERO_SERVER_URL": "ws://libero-only:9000",
        }
    )

    assert config.server_url == "ws://server-uri:9000"


def test_result_file_can_be_overridden():
    config = LiberoClientConfig.from_env({"HIMEM_LIBERO_RESULT_FILE": "run_outputs/results.json"})

    assert config.result_file == "run_outputs/results.json"


def test_invalid_max_steps_count_is_rejected():
    try:
        align_max_steps([1, 2], ["libero_spatial", "libero_goal", "libero_10"])
    except ValueError as exc:
        assert "one integer per task suite" in str(exc)
    else:
        raise AssertionError("Expected max_steps mismatch to raise ValueError")


def test_negative_task_limit_is_rejected():
    try:
        LiberoClientConfig.from_env({"HIMEM_LIBERO_TASK_LIMIT": "-1"})
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative task limit to raise ValueError")


def test_configure_mujoco_environment_sets_egl_platform():
    config = LiberoClientConfig.from_env({"HIMEM_MUJOCO_GL": "egl"})
    environ = {}

    configure_mujoco_environment(config, environ)

    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"
