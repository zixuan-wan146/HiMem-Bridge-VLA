import unittest


class FlowMatchingConfigTests(unittest.TestCase):
    def test_action_head_can_be_constructed_without_config(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
        )

        self.assertIsInstance(head, torch.nn.Module)
        self.assertEqual(head.horizon, 2)
        self.assertEqual(head.per_action_dim, 3)
        self.assertEqual(head.action_dim, 6)

    def test_action_encoder_rejects_wrong_horizon(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        encoder = flow_matching.MultiEmbodimentActionEncoder(
            action_dim=3,
            embed_dim=8,
            hidden_dim=8,
            horizon=2,
            num_categories=1,
        )

        action_seq = torch.zeros(1, 3, 3)
        category_id = torch.zeros(1, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "must match horizon"):
            encoder(action_seq, category_id)

    def test_action_head_rejects_wrong_training_action_mask_shape(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
        )
        fused_tokens = torch.zeros(1, 1, 8)
        actions_gt = torch.zeros(1, 2, 3)
        action_mask = torch.ones(1, 6)

        with self.assertRaisesRegex(ValueError, "action_mask shape"):
            head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
