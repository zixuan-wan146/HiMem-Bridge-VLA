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
    assert tuple(sample["action_segments"].shape) == (2, 2, 3)
    assert tuple(sample["action_segment_mask"].shape) == (2,)


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
    model_config = resolve_model_config(config, train_dataset.sample_shapes, latent_dim=5)

    assert model_config.hidden_dim == 4
    assert model_config.state_dim == 2
    assert model_config.latent_dim == 5
    assert model_config.num_plan_steps == 2
    assert model_config.num_layers == 4


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
    model = CoarsePlanner(resolve_model_config(config, train_dataset.sample_shapes, latent_dim=5))
    segment_autoencoder = _FakeSegmentAutoencoder(latent_dim=5)

    metrics = evaluate_planner(model, segment_autoencoder, DataLoader(train_dataset, batch_size=2), config, device="cpu")

    assert metrics["loss"] >= 0.0
    assert metrics["latent_mse"] >= 0.0
    assert metrics["normalized_latent_mse"] >= 0.0
    assert metrics["raw_latent_mse"] >= 0.0
    assert metrics["decoded_chunk_loss"] >= 0.0
    assert -1.0 <= metrics["latent_cosine_similarity"] <= 1.0


def test_standalone_evaluate_accepts_latent_normalizer(tmp_path):
    from torch.utils.data import DataLoader

    from coarse_planner.config import load_config
    from coarse_planner.data import build_synthetic_feature_cache, build_datasets
    from coarse_planner.evaluate import evaluate_planner
    from coarse_planner.latent_normalization import LatentNormalizer
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
    model = CoarsePlanner(resolve_model_config(config, train_dataset.sample_shapes, latent_dim=5))
    segment_autoencoder = _FakeSegmentAutoencoder(latent_dim=5)
    normalizer = LatentNormalizer(mean=torch.ones(5), std=torch.full((5,), 2.0), count=8)

    metrics = evaluate_planner(
        model,
        segment_autoencoder,
        DataLoader(train_dataset, batch_size=2),
        config,
        device="cpu",
        latent_normalizer=normalizer,
    )

    assert metrics["normalized_latent_mse"] != metrics["raw_latent_mse"]
    assert "normalized_latent_mse_u0" in metrics


def test_convert_raw_latent_head_preserves_unnormalized_output():
    from coarse_planner.latent_normalization import LatentNormalizer
    from coarse_planner.train import _convert_raw_latent_head_to_normalized_output
    from himem_bridge_vla.model.planner import CoarsePlanner, CoarsePlannerConfig

    model = CoarsePlanner(
        CoarsePlannerConfig(
            hidden_dim=4,
            state_dim=2,
            latent_dim=3,
            num_plan_steps=2,
            planning_horizon=4,
            num_layers=3,
            num_heads=2,
            dropout=0.0,
        )
    )
    model.eval()
    vlm_tokens = torch.randn(2, 3, 4)
    state = torch.randn(2, 2)
    normalizer = LatentNormalizer(
        mean=torch.tensor([0.5, -0.25, 1.0]),
        std=torch.tensor([2.0, 0.5, 4.0]),
        count=16,
    )

    with torch.no_grad():
        raw_before = model(vlm_tokens, state).predicted_latents.clone()
        _convert_raw_latent_head_to_normalized_output(model, normalizer)
        normalized_after = model(vlm_tokens, state).predicted_latents
        raw_after = normalizer.unnormalize(normalized_after)

    assert torch.allclose(raw_after, raw_before, atol=1.0e-5)


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


class _FakeSegmentAutoencoder(torch.nn.Module):
    def __init__(self, *, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def encode(self, action_segments):
        return action_segments.new_zeros((*action_segments.shape[:2], self.latent_dim))

    def decode(self, latents):
        return latents.new_zeros((*latents.shape[:2], 2, 3))
