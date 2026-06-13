import unittest


class TrainingLossTests(unittest.TestCase):
    def test_masked_flow_matching_mse_ignores_inactive_dimensions(self):
        torch = self._import_or_skip("torch")
        training_loss = self._import_or_skip("himem_bridge_vla.training_loss")

        pred = torch.tensor([[1.0, 10.0, 3.0, 10.0]])
        target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

        loss = training_loss.masked_flow_matching_mse(pred, target, mask)

        self.assertEqual(loss.item(), 2.5)

    def test_masked_flow_matching_mse_rejects_empty_mask(self):
        torch = self._import_or_skip("torch")
        training_loss = self._import_or_skip("himem_bridge_vla.training_loss")

        pred = torch.zeros(1, 2)
        target = torch.zeros(1, 2)
        mask = torch.zeros(1, 1, 2)

        with self.assertRaisesRegex(ValueError, "action_mask.sum"):
            training_loss.masked_flow_matching_mse(pred, target, mask)

    def test_boundary_bce_loss_accepts_batch_labels(self):
        torch = self._import_or_skip("torch")
        training_loss = self._import_or_skip("himem_bridge_vla.training_loss")

        logits = torch.tensor([[0.0], [2.0]])
        labels = torch.tensor([0.0, 1.0])

        loss = training_loss.boundary_bce_loss(logits, labels)

        self.assertEqual(tuple(loss.shape), ())

    def test_progress_smooth_l1_loss_accepts_batch_labels(self):
        torch = self._import_or_skip("torch")
        training_loss = self._import_or_skip("himem_bridge_vla.training_loss")

        logits = torch.tensor([[0.0], [2.0]])
        labels = torch.tensor([0.5, 1.0])

        loss = training_loss.progress_smooth_l1_loss(logits, labels)

        self.assertEqual(tuple(loss.shape), ())

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
