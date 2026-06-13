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

    def test_single_step_action_head_registers_projection_before_forward(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=3,
            horizon=1,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=1,
        )

        param_names_before = set(dict(head.named_parameters()))
        self.assertIn("single_action_proj.weight", param_names_before)

        fused_tokens = torch.zeros(2, 1, 8)
        actions_gt = torch.zeros(2, 1, 3)
        action_mask = torch.ones(2, 1, 3)
        pred_velocity, noise = head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

        self.assertEqual(tuple(pred_velocity.shape), (2, 3))
        self.assertEqual(tuple(noise.shape), (2, 1, 3))
        self.assertEqual(set(dict(head.named_parameters())), param_names_before)

    def test_inference_keeps_masked_action_dimensions_zero_after_final_step(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        torch.manual_seed(0)
        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
            num_inference_timesteps=2,
        )
        fused_tokens = torch.zeros(1, 1, 8)
        action_mask = torch.tensor([[1.0, 0.0, 1.0]])

        action = head.get_action(fused_tokens, action_mask=action_mask).view(1, 2, 3)

        self.assertTrue(torch.equal(action[:, :, 1], torch.zeros_like(action[:, :, 1])))

    def test_action_head_training_shapes_for_common_horizons(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("himem_bridge_vla.model.action_head.flow_matching")

        for horizon, per_action_dim in ((1, 7), (14, 7), (16, 8)):
            with self.subTest(horizon=horizon, per_action_dim=per_action_dim):
                action_dim = horizon * per_action_dim
                head = flow_matching.FlowmatchingActionHead(
                    embed_dim=8,
                    hidden_dim=16,
                    action_dim=action_dim,
                    horizon=horizon,
                    per_action_dim=per_action_dim,
                    num_heads=2,
                    num_layers=1,
                    num_inference_timesteps=1,
                )
                fused_tokens = torch.zeros(2, 3, 8)
                actions_gt = torch.zeros(2, horizon, per_action_dim)
                action_mask = torch.ones(2, horizon, per_action_dim)

                pred_velocity, noise = head(fused_tokens, actions_gt=actions_gt, action_mask=action_mask)

                self.assertEqual(tuple(pred_velocity.shape), (2, action_dim))
                self.assertEqual(tuple(noise.shape), (2, horizon, per_action_dim))

    def test_multi_category_action_head_training_shape(self):
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
            num_categories=2,
        )
        fused_tokens = torch.zeros(2, 3, 8)
        actions_gt = torch.zeros(2, 2, 3)
        action_mask = torch.ones(2, 2, 3)
        embodiment_id = torch.tensor([0, 1], dtype=torch.long)

        pred_velocity, noise = head(
            fused_tokens,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
        )

        self.assertEqual(tuple(pred_velocity.shape), (2, 6))
        self.assertEqual(tuple(noise.shape), (2, 2, 3))

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
