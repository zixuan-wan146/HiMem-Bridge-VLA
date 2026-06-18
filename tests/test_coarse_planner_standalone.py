import pytest

torch = pytest.importorskip("torch")


def test_build_planner_feature_cache_and_dataset(tmp_path):
    from coarse_planner.config import load_config
    from coarse_planner.data import PlannerFeatureDataset, build_planner_feature_cache

    source = tmp_path / "source.pt"
    torch.save({"episodes": [_episode("ep0"), _episode("ep1")]}, source)
    config = load_config(None)
    config["seed"] = 1
    config["data"].update(
        {
            "root": str(tmp_path / "cache"),
            "input_paths": [str(source)],
            "include_tail": False,
            "max_samples_per_shard": 3,
            "val_fraction": 0.5,
        }
    )
    config["target"].update({"num_plan_steps": 2, "planning_horizon": 4, "gripper_indices": [-1]})

    manifest = build_planner_feature_cache(config)
    assert manifest["num_samples"] == 10
    assert manifest["split_counts"]["train"] > 0
    assert manifest["split_counts"]["eval"] > 0

    train_dataset = PlannerFeatureDataset(config["data"]["root"], split="train")
    sample = train_dataset[0]
    assert tuple(sample["vlm_tokens"].shape) == (3, 4)
    assert tuple(sample["state"].shape) == (2,)
    assert tuple(sample["coarse_actions"].shape) == (2, 3)
    assert tuple(sample["coarse_action_mask"].shape) == (2,)


def test_standalone_model_config_uses_dataset_shapes(tmp_path):
    from coarse_planner.config import load_config
    from coarse_planner.data import build_synthetic_feature_cache, build_datasets
    from coarse_planner.train import resolve_model_config

    config = load_config(None)
    config["data"].update({"root": str(tmp_path / "cache"), "include_tail": False, "val_fraction": 0.5})
    config["target"].update({"num_plan_steps": 2, "planning_horizon": 4})
    config["model"].update({"num_heads": 2})
    build_synthetic_feature_cache(
        config,
        tmp_path / "cache",
        num_episodes=2,
        episode_length=6,
        hidden_dim=4,
        state_dim=2,
        action_dim=3,
        num_tokens=3,
    )

    train_dataset, _ = build_datasets(config)
    model_config = resolve_model_config(config, train_dataset.sample_shapes)

    assert model_config.hidden_dim == 4
    assert model_config.state_dim == 2
    assert model_config.action_dim == 3
    assert model_config.num_plan_steps == 2
    assert model_config.num_layers == 3


def test_standalone_evaluate_runs_on_synthetic_cache(tmp_path):
    from torch.utils.data import DataLoader

    from coarse_planner.config import load_config
    from coarse_planner.data import build_synthetic_feature_cache, build_datasets
    from coarse_planner.evaluate import evaluate_planner
    from coarse_planner.train import resolve_model_config
    from himem_bridge_vla.model.planner import CoarsePlanner

    config = load_config(None)
    config["data"].update({"root": str(tmp_path / "cache"), "include_tail": False, "val_fraction": 0.5})
    config["target"].update({"num_plan_steps": 2, "planning_horizon": 4})
    config["model"].update({"num_heads": 2})
    build_synthetic_feature_cache(
        config,
        tmp_path / "cache",
        num_episodes=2,
        episode_length=6,
        hidden_dim=4,
        state_dim=2,
        action_dim=3,
        num_tokens=3,
    )
    train_dataset, _ = build_datasets(config)
    model = CoarsePlanner(resolve_model_config(config, train_dataset.sample_shapes))

    metrics = evaluate_planner(model, DataLoader(train_dataset, batch_size=2), config, device="cpu")

    assert metrics["loss"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_simulation_source_validation_reports_missing_videos(tmp_path):
    from coarse_planner.build_from_simulation import validate_simulation_sources

    dataset_root = tmp_path / "dataset"
    parquet_dir = dataset_root / "data" / "chunk-000"
    meta_dir = dataset_root / "meta"
    parquet_dir.mkdir(parents=True)
    meta_dir.mkdir()
    (parquet_dir / "episode_000000.parquet").write_text("placeholder")
    (meta_dir / "tasks.jsonl").write_text("{}\n")
    (meta_dir / "stats.json").write_text("{}\n")
    config = {
        "data_groups": {
            "arm": {
                "demo": {
                    "path": str(dataset_root),
                    "view_map": {"image_1": "image"},
                }
            }
        }
    }

    summary = validate_simulation_sources(config)

    assert summary["total_parquet_files"] == 1
    assert summary["missing_video_files"] == 1


def test_simulation_feature_split_is_stable():
    from coarse_planner.build_from_simulation import split_for_episode

    first = split_for_episode("episode-1", seed=3, val_fraction=0.2, train_split="train", eval_split="eval")
    second = split_for_episode("episode-1", seed=3, val_fraction=0.2, train_split="train", eval_split="eval")

    assert first == second


def _episode(episode_id):
    return {
        "episode_id": episode_id,
        "vlm_tokens": torch.arange(8 * 3 * 4, dtype=torch.float32).reshape(8, 3, 4),
        "states": torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2),
        "actions": torch.ones(8, 3),
        "frame_index": torch.arange(8),
    }
