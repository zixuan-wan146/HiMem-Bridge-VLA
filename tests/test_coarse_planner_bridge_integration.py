import unittest


class CoarsePlannerBridgeIntegrationTests(unittest.TestCase):
    def test_planner_tokens_enter_bridge_adapter_condition_path(self):
        torch = self._import_or_skip("torch")
        bridge = self._import_or_skip("himem_bridge_vla.model.bridge")
        planner = self._import_or_skip("himem_bridge_vla.model.planner")

        planner_module = planner.CoarsePlanner(
            planner.CoarsePlannerConfig(
                hidden_dim=8,
                action_dim=3,
                state_dim=4,
                num_plan_steps=5,
                planning_horizon=20,
                num_layers=3,
                num_heads=2,
            )
        )
        bridge_adapter = bridge.BridgeAdapter(
            bridge.BridgeAdapterConfig(
                embed_dim=8,
                raw_dim=8,
                state_dim=4,
                num_layers=2,
                num_heads=2,
                num_bridge_tokens=6,
                num_action_queries=7,
            )
        )

        fused_tokens = torch.randn(2, 9, 8)
        state = torch.randn(2, 4)
        planner_output = planner_module(fused_tokens, state)
        bridge_output = bridge_adapter(
            fused_tokens,
            hidden_states=[fused_tokens, fused_tokens],
            state=state,
            plan_tokens=planner_output.plan_tokens,
            memory_context=torch.randn(2, 3, 8),
        )

        self.assertEqual(tuple(planner_output.plan_tokens.shape), (2, 5, 8))
        self.assertEqual(tuple(planner_output.coarse_actions.shape), (2, 5, 3))
        self.assertEqual(tuple(bridge_output.bridge_tokens.shape), (2, 6, 8))

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
