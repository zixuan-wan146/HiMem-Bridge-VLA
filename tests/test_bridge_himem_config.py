from pathlib import Path
import unittest


class BridgeHiMemConfigTests(unittest.TestCase):
    def test_load_crosskv_yaml_maps_to_legacy_model_config(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "bridge_himem"
            / "experiments"
            / "crosskv_clean.yaml"
        )

        config = config_module.load_bridge_himem_config(config_path)
        legacy = config.to_legacy_model_config()

        self.assertEqual(config.experiment_name, "crosskv_clean")
        self.assertTrue(legacy["use_bridge"])
        self.assertTrue(legacy["use_himem"])
        self.assertEqual(legacy["bridge_variant"], "crosskv")
        self.assertEqual(legacy["memory_placement"], "crosskv")
        self.assertEqual(legacy["bridge_raw_layers"], [3, 7, 11, 14])
        self.assertFalse(legacy["allow_image_token_truncation"])
        self.assertEqual(legacy["bridge_context_mode"], "bridge_clean")
        self.assertEqual(legacy["memory_write_tokens"], 4)

    def test_all_repository_bridge_himem_yamls_are_valid(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_root = Path(__file__).resolve().parents[1] / "configs" / "bridge_himem"

        loaded_names = []
        for config_path in sorted(config_root.glob("**/*.yaml")):
            loaded_names.append(config_module.load_bridge_himem_config(config_path).experiment_name)

        self.assertIn("baseline_fused_only", loaded_names)
        self.assertIn("coarse_planner_crosskv", loaded_names)
        self.assertIn("coarse_planner_plan_only", loaded_names)
        self.assertIn("crosskv_clean", loaded_names)
        self.assertIn("mixed_latent_clean", loaded_names)
        self.assertIn("mixed_latent_skill", loaded_names)

    def test_yaml_extends_merges_base_and_overlay(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "bridge_himem"
            / "experiments"
            / "mixed_latent_skill.yaml"
        )

        config = config_module.load_bridge_himem_config(config_path)

        self.assertEqual(config.experiment_name, "mixed_latent_skill")
        self.assertEqual(config.bridge.variant, "mixed_latent")
        self.assertTrue(config.memory.enabled)
        self.assertTrue(config.skill.enabled)
        self.assertEqual(config.memory.writer.num_tokens, 4)

    def test_nested_mapping_is_supported(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        config = config_module.BridgeHiMemConfig.from_mapping(
            {
                "bridge_himem": {
                    "vlm": {"hidden_dim": 8, "raw_layers": ["shallow", "deep"]},
                    "action_query": {"num_tokens": 3},
                    "bridge": {
                        "enabled": True,
                        "variant": "mixed_latent",
                        "num_layers": 1,
                        "num_heads": 2,
                        "num_action_tokens": 2,
                    },
                    "context": {"mode": "bridge_clean"},
                    "memory": {
                        "enabled": True,
                        "placement": "mixed_latent",
                        "token_dim": 8,
                        "bank_max_tokens": 4,
                        "read_top_k": 2,
                        "writer": {"num_tokens": 2, "num_heads": 2},
                    },
                }
            }
        )

        self.assertEqual(config.vlm.raw_layers, ("shallow", "deep"))
        self.assertEqual(config.to_legacy_model_config()["bridge_num_action_queries"], 3)

    def test_coarse_planner_maps_to_legacy_model_config(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "bridge_himem"
            / "experiments"
            / "coarse_planner_crosskv.yaml"
        )

        config = config_module.load_bridge_himem_config(config_path)
        legacy = config.to_legacy_model_config()

        self.assertTrue(config.coarse_planner.enabled)
        self.assertEqual(config.coarse_planner.num_layers, 4)
        self.assertEqual(config.coarse_planner.latent_dim, 128)
        self.assertEqual(config.coarse_planner.latent_head_hidden_dim, 512)
        self.assertEqual(config.coarse_planner.planning_horizon, 64)
        self.assertEqual(config.coarse_planner.num_plan_steps, 8)
        self.assertEqual(config.coarse_planner.execution_horizon, 16)
        self.assertEqual(config.coarse_planner.suffix_stride_tokens, 2)
        self.assertFalse(config.coarse_planner.input_memory)
        self.assertTrue(legacy["coarse_planner_enabled"])
        self.assertEqual(legacy["coarse_planner_latent_dim"], 128)
        self.assertEqual(legacy["coarse_planner_latent_head_hidden_dim"], 512)
        self.assertEqual(legacy["coarse_planner_execution_horizon"], 16)
        self.assertEqual(legacy["coarse_planner_suffix_stride_tokens"], 2)
        self.assertEqual(legacy["coarse_planner_refresh_policy"], "transition_or_queue")
        self.assertEqual(legacy["coarse_planner_placement"], "bridge_crosskv")

    def test_coarse_planner_rejects_memory_input(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "input_memory"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "bridge": {"enabled": True, "num_heads": 2},
                    "context": {"mode": "bridge_clean"},
                    "coarse_planner": {
                        "enabled": True,
                        "hidden_dim": 8,
                        "num_heads": 2,
                        "num_layers": 3,
                        "planning_horizon": 12,
                        "num_plan_steps": 3,
                        "input_memory": True,
                    },
                }
            )

    def test_coarse_planner_rejects_out_of_range_action_indices(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "gripper_indices"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "coarse_planner": {
                        "segment_action_dim": 3,
                        "gripper_indices": [3],
                    }
                }
            )

    def test_memory_requires_bridge(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "requires bridge"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "bridge": {"enabled": False, "num_heads": 2},
                    "memory": {"enabled": True, "token_dim": 8},
                }
            )

    def test_memory_placement_must_match_variant(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "memory.placement must match"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "bridge": {"enabled": True, "variant": "crosskv", "num_heads": 2},
                    "context": {"mode": "bridge_clean"},
                    "memory": {"enabled": True, "placement": "mixed_latent", "token_dim": 8},
                }
            )

    def test_context_mode_is_validated(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "context.mode"):
            config_module.BridgeHiMemConfig.from_mapping({"context": {"mode": "unclear"}})

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
