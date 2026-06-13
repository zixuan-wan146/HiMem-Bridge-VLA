from __future__ import annotations

from evaluations.calvin.calvin_client_config import (
    DEFAULT_CALVIN_ROOT,
    CalvinClientConfig,
    configure_calvin_environment,
)


def test_default_calvin_config_matches_full_eval_defaults():
    config = CalvinClientConfig.from_env({})

    assert config.server_url == "ws://127.0.0.1:9000"
    assert config.calvin_root == DEFAULT_CALVIN_ROOT
    assert config.dataset_path == f"{DEFAULT_CALVIN_ROOT}/dataset/task_ABC_D"
    assert config.num_sequences == 1000
    assert config.horizon == 14
    assert config.max_steps_per_subtask == 360
    assert config.gripper_mode == "openvla"
    assert config.reset_memory_scope == "sequence"
    assert config.save_video is False


def test_calvin_config_prefers_shared_server_uri():
    config = CalvinClientConfig.from_env(
        {
            "HIMEM_SERVER_URI": "ws://shared:9000",
            "HIMEM_CALVIN_SERVER_URL": "ws://calvin-only:9000",
        }
    )

    assert config.server_url == "ws://shared:9000"


def test_calvin_config_can_override_paths_and_counts():
    config = CalvinClientConfig.from_env(
        {
            "HIMEM_CALVIN_ROOT": "/data/calvin",
            "HIMEM_CALVIN_DATASET_PATH": "/datasets/task_D_D",
            "HIMEM_CALVIN_NUM_SEQUENCES": "3",
            "HIMEM_CALVIN_SEQUENCE_OFFSET": "7",
            "HIMEM_CALVIN_SAVE_VIDEO": "true",
            "HIMEM_CALVIN_GRIPPER_MODE": "passthrough",
        }
    )

    assert config.calvin_root == "/data/calvin"
    assert config.dataset_path == "/datasets/task_D_D"
    assert config.num_sequences == 3
    assert config.sequence_offset == 7
    assert config.save_video is True
    assert config.gripper_mode == "passthrough"


def test_invalid_reset_memory_scope_is_rejected():
    try:
        CalvinClientConfig.from_env({"HIMEM_CALVIN_RESET_MEMORY_SCOPE": "episode"})
    except ValueError as exc:
        assert "HIMEM_CALVIN_RESET_MEMORY_SCOPE" in str(exc)
    else:
        raise AssertionError("Expected invalid memory reset scope to raise ValueError")


def test_configure_calvin_environment_sets_calvin_root_and_egl_platform():
    config = CalvinClientConfig.from_env({"HIMEM_MUJOCO_GL": "egl", "HIMEM_CALVIN_ROOT": "/data/calvin"})
    environ = {}

    configure_calvin_environment(config, environ)

    assert environ["CALVIN_ROOT"] == "/data/calvin"
    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"
