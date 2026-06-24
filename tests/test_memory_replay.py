from __future__ import annotations

from himem_bridge_vla.dataset.memory_replay import build_memory_replay_manifest
from himem_bridge_vla.dataset.memory_replay import build_memory_replay_samples
from himem_bridge_vla.dataset.memory_replay import read_memory_replay_jsonl
from himem_bridge_vla.dataset.memory_replay import write_memory_replay_jsonl


def test_build_memory_replay_samples_uses_low_level_offsets_and_masks_missing_history(tmp_path):
    samples = build_memory_replay_samples(
        episode_id="episode0",
        episode_length=80,
        action_horizon=32,
        stride=16,
        short_offsets=(32, 16),
        long_candidate_steps=(8, 24, 40),
        long_capacity=2,
        include_tail=False,
        benchmark="synthetic",
        task_name="task",
        source_path="episode0.hdf5",
    )

    assert [sample.current_step for sample in samples] == [0, 16, 32, 48]
    assert samples[0].short_steps == (None, None)
    assert samples[0].short_mask == (False, False)
    assert samples[2].short_steps == (0, 16)
    assert samples[2].short_mask == (True, True)
    assert samples[3].long_steps == (24, 40)
    assert samples[-1].action_valid_count == 32

    output = write_memory_replay_jsonl(tmp_path / "index.jsonl", samples)
    rows = read_memory_replay_jsonl(output)

    assert rows[0]["action_start"] == 0
    assert rows[0]["action_end"] == 32
    assert rows[0]["source_path"] == "episode0.hdf5"
    assert rows[2]["short_steps"] == [0, 16]


def test_build_memory_replay_samples_can_include_tail_with_valid_count():
    samples = build_memory_replay_samples(
        episode_id="episode0",
        episode_length=40,
        action_horizon=32,
        stride=16,
        include_tail=True,
    )

    assert [sample.current_step for sample in samples] == [0, 16, 32]
    assert [sample.action_valid_count for sample in samples] == [32, 24, 8]


def test_build_memory_replay_manifest_records_generation_policy():
    manifest = build_memory_replay_manifest(
        benchmark="RMBench",
        action_horizon=32,
        stride=1,
        short_offsets=(16, 32),
        long_capacity=4,
        include_tail=False,
        sample_count=10,
        episode_count=2,
        task_counts={"b": 4, "a": 6},
    )

    assert manifest["format"] == "memory_replay_index"
    assert manifest["short_offsets"] == [32, 16]
    assert manifest["task_counts"] == {"a": 6, "b": 4}
