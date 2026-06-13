import unittest


class InternVL3HiddenSelectionTests(unittest.TestCase):
    def test_select_hidden_states_supports_named_and_numeric_layers(self):
        torch = self._import_or_skip("torch")
        internvl3 = self._import_or_skip("himem_bridge_vla.model.internvl3.internvl3_embedder")

        hidden_states = [torch.full((1, 2, 3), float(index)) for index in range(8)]

        selected = internvl3.select_hidden_states(hidden_states, ("shallow", "mid", "deep", -1))

        self.assertEqual([tensor[0, 0, 0].item() for tensor in selected], [2.0, 4.0, 7.0, 7.0])

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
