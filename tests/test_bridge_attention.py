import unittest


class BridgeAttentionTests(unittest.TestCase):
    def test_bridge_attention_block_shapes_and_zero_raw_gate(self):
        torch = self._import_or_skip("torch")
        bridge = self._import_or_skip("himem_bridge_vla.model.bridge")

        block = bridge.BridgeAttentionBlock(hidden_dim=8, raw_dim=10, query_dim=8, num_heads=2)
        action_tokens = torch.randn(2, 4, 8)
        raw_features = torch.randn(2, 6, 10)
        action_queries = torch.randn(2, 5, 8)
        proprio = torch.randn(2, 1, 8)

        output = block(action_tokens, raw_features, action_queries, proprio)

        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertAlmostEqual(block.raw_gate_value.item(), 0.0, places=6)

    def test_bridge_adapter_outputs_context_tokens(self):
        torch = self._import_or_skip("torch")
        bridge = self._import_or_skip("himem_bridge_vla.model.bridge")

        config = bridge.BridgeAdapterConfig(
            embed_dim=8,
            raw_dim=8,
            state_dim=3,
            num_layers=2,
            num_heads=2,
            num_bridge_tokens=4,
            num_action_queries=5,
        )
        adapter = bridge.BridgeAdapter(config)
        fused_tokens = torch.randn(2, 6, 8)
        hidden_states = [torch.randn(2, 6, 8), torch.randn(2, 6, 8)]
        state = torch.randn(2, 3)
        memory_context = torch.randn(2, 2, 8)

        output = adapter(
            fused_tokens,
            hidden_states=hidden_states,
            state=state,
            memory_context=memory_context,
        )

        self.assertEqual(tuple(output.bridge_tokens.shape), (2, 4, 8))
        self.assertEqual(tuple(output.boundary_logits.shape), (2, 1))
        self.assertEqual(tuple(output.progress_logits.shape), (2, 1))
        self.assertEqual(tuple(output.raw_gate_values.shape), (2,))
        self.assertTrue(torch.allclose(output.raw_gate_values, torch.zeros_like(output.raw_gate_values)))

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
