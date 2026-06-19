import unittest


class CoarsePlannerTests(unittest.TestCase):
    def test_coarse_planner_outputs_plan_tokens_and_actions(self):
        torch = self._import_or_skip("torch")
        planner = self._import_or_skip("himem_bridge_vla.model.planner")

        config = planner.CoarsePlannerConfig(
            hidden_dim=8,
            state_dim=4,
            latent_dim=6,
            num_plan_steps=5,
            planning_horizon=20,
            num_layers=3,
            num_heads=2,
        )
        module = planner.CoarsePlanner(config)

        output = module(torch.randn(2, 6, 8), torch.randn(2, 4))

        self.assertEqual(tuple(output.plan_tokens.shape), (2, 5, 8))
        self.assertEqual(tuple(output.predicted_latents.shape), (2, 5, 6))

    def test_coarse_planner_requires_three_layers(self):
        planner = self._import_or_skip("himem_bridge_vla.model.planner")

        with self.assertRaisesRegex(ValueError, "at least 3"):
            planner.CoarsePlanner(
                planner.CoarsePlannerConfig(
                    hidden_dim=8,
                    state_dim=4,
                    latent_dim=6,
                    num_plan_steps=5,
                    planning_horizon=20,
                    num_layers=2,
                    num_heads=2,
                )
            )

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
