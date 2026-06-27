from himem_bridge_vla.path_utils import find_repo_root
import unittest


class BridgeHiMemConfigTests(unittest.TestCase):
    def test_load_crosskv_yaml_maps_to_legacy_model_config(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_path = (
            find_repo_root(__file__)
            / "configs"
            / "experiments"
            / "bridge_himem"
            / "crosskv_clean.yaml"
        )

        config = config_module.load_bridge_himem_config(config_path)
        legacy = config.to_legacy_model_config()

        self.assertEqual(config.experiment_name, "crosskv_clean")
        self.assertTrue(legacy["use_bridge"])
        self.assertTrue(legacy["use_memory"])
        self.assertEqual(legacy["bridge_variant"], "crosskv")
        self.assertEqual(legacy["memory_kind"], "fixed_recent_visual")
        self.assertEqual(legacy["memory_short_offsets"], [16, 8])
        self.assertEqual(legacy["memory_entry_tokens"], 16)
        self.assertEqual(legacy["bridge_raw_layers"], [3, 6, 9, 12])
        self.assertFalse(legacy["allow_image_token_truncation"])
        self.assertEqual(legacy["bridge_context_mode"], "bridge_clean")

    def test_all_repository_bridge_himem_yamls_are_valid(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        repo_root = find_repo_root(__file__)
        config_paths = [
            *sorted((repo_root / "configs" / "models" / "bridge_himem").glob("*.yaml")),
            *sorted((repo_root / "configs" / "experiments" / "bridge_himem").glob("*.yaml")),
        ]

        loaded_names = [config_module.load_bridge_himem_config(config_path).experiment_name for config_path in config_paths]

        self.assertIn("baseline_fused_only", loaded_names)
        self.assertIn("crosskv_clean", loaded_names)
        self.assertIn("mixed_latent_clean", loaded_names)
        self.assertIn("mixed_latent_skill", loaded_names)

    def test_yaml_extends_merges_base_and_overlay(self):
        self._import_or_skip("yaml")
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")
        config_path = (
            find_repo_root(__file__)
            / "configs"
            / "experiments"
            / "bridge_himem"
            / "mixed_latent_skill.yaml"
        )

        config = config_module.load_bridge_himem_config(config_path)

        self.assertEqual(config.experiment_name, "mixed_latent_skill")
        self.assertEqual(config.bridge.variant, "mixed_latent")
        self.assertTrue(config.memory.enabled)
        self.assertTrue(config.skill.enabled)
        self.assertEqual(config.memory.compression.entry_tokens, 16)

    def test_nested_mapping_is_supported(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        config = config_module.BridgeHiMemConfig.from_mapping(
            {
                "bridge_himem": {
                    "vlm": {"hidden_dim": 8, "raw_layers": ["shallow", "deep"]},
                    "bridge": {
                        "enabled": True,
                        "variant": "mixed_latent",
                        "num_layers": 1,
                        "num_heads": 2,
                        "num_bridge_tokens": 2,
                        "num_action_queries": 3,
                    },
                    "context": {"mode": "bridge_clean"},
                    "memory": {
                        "enabled": True,
                        "hidden_dim": 8,
                        "views": ["base", "wrist"],
                        "short": {"capacity": 2, "offsets": [32, 16]},
                        "long": {"capacity": 0},
                        "compression": {"entry_tokens": 2, "num_heads": 2},
                    },
                }
            }
        )

        self.assertEqual(config.vlm.raw_layers, ("shallow", "deep"))
        self.assertEqual(config.to_legacy_model_config()["bridge_num_action_queries"], 3)

    def test_progress_planner_maps_to_legacy_model_config(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        config = config_module.BridgeHiMemConfig.from_mapping(
            {
                "vlm": {"hidden_dim": 8},
                "bridge": {"enabled": True, "variant": "direct", "num_heads": 2},
                "progress_planner": {
                    "enabled": True,
                    "hidden_dim": 8,
                    "state_dim": 5,
                    "action_dim": 3,
                    "replan_stride": 4,
                    "planner_layers": 1,
                    "num_heads": 2,
                },
            }
        )
        legacy = config.to_legacy_model_config()

        self.assertTrue(config.progress_planner.enabled)
        self.assertTrue(legacy["progress_planner_enabled"])
        self.assertEqual(legacy["progress_planner_hidden_dim"], 8)
        self.assertEqual(legacy["progress_planner_state_dim"], 5)
        self.assertEqual(legacy["progress_planner_action_dim"], 3)
        self.assertEqual(legacy["progress_planner_replan_stride"], 4)

    def test_memory_hidden_dim_must_match_vlm_hidden_dim(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "memory.hidden_dim"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "memory": {"enabled": True, "hidden_dim": 16},
                }
            )

    def test_memory_short_offsets_must_match_capacity(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "offsets length"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "memory": {"enabled": True, "hidden_dim": 8, "short": {"capacity": 2, "offsets": [16]}},
                }
            )

    def test_memory_long_capacity_must_remain_zero(self):
        config_module = self._import_or_skip("himem_bridge_vla.bridge_himem_config")

        with self.assertRaisesRegex(ValueError, "memory.long.capacity"):
            config_module.BridgeHiMemConfig.from_mapping(
                {
                    "vlm": {"hidden_dim": 8},
                    "memory": {"enabled": True, "hidden_dim": 8, "long": {"capacity": 1}},
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
