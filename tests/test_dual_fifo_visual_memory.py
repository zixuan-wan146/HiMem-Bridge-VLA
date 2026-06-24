import unittest


class DualFifoVisualMemoryTests(unittest.TestCase):
    def test_short_memory_reads_offsets_oldest_to_newest_with_trailing_padding(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        memory = himem.DualFifoVisualMemory(short_offsets=(32, 16), long_capacity=4)
        memory.write_observation(64, {"base": torch.ones(2, 4)})
        memory.write_observation(80, {"base": torch.full((2, 4), 2.0)})

        result = memory.read(96)

        self.assertEqual([entry.tau if entry is not None else None for entry in result.entries[:2]], [64, 80])
        self.assertEqual(result.entry_mask.tolist(), [True, True, False, False, False, False])
        self.assertEqual(result.token_mask(tokens_per_entry=1).tolist(), [True, True, False, False, False, False])

    def test_short_memory_compacts_available_entries_when_older_offset_is_missing(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        memory = himem.DualFifoVisualMemory(short_offsets=(32, 16), long_capacity=0)
        memory.write_observation(80, {"base": torch.ones(2, 4)})

        result = memory.read(96)

        self.assertEqual([entry.tau if entry is not None else None for entry in result.entries], [80, None])
        self.assertEqual(result.entry_mask.tolist(), [True, False])

    def test_long_memory_keeps_fifo_capacity(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        memory = himem.DualFifoVisualMemory(short_offsets=(16,), long_capacity=3)
        for tau in (8, 16, 24, 32):
            memory.write_long(tau, {"base": torch.full((1, 4), float(tau))})

        self.assertEqual([entry.tau for entry in memory.long_entries()], [16, 24, 32])
        result = memory.read(48)
        self.assertEqual([entry.tau if entry is not None else None for entry in result.entries], [None, 16, 24, 32])
        self.assertEqual(result.entry_mask.tolist(), [False, True, True, True])

    def test_entry_mask_expands_to_token_mask(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        mask = torch.tensor([True, False, True])

        self.assertEqual(himem.expand_entry_mask(mask, tokens_per_entry=2).tolist(), [True, True, False, False, True, True])

    def test_visual_memory_compressor_outputs_shape_and_masks_padding(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        memory = himem.DualFifoVisualMemory(short_offsets=(32, 16), long_capacity=4)
        memory.write_observation(64, {"base": torch.randn(3, 8), "wrist": torch.randn(2, 8)})
        result = memory.read(96)
        compressor = himem.VisualMemoryCompressor(
            hidden_dim=8,
            view_names=("base", "wrist"),
            tokens_per_entry=2,
            num_heads=2,
            dropout=0.0,
            max_age_steps=128,
        )

        compressed = compressor(result, current_step=96)

        self.assertEqual(tuple(compressed.tokens.shape), (12, 8))
        self.assertEqual(compressed.mask.tolist(), [True, True, False, False, False, False, False, False, False, False, False, False])
        self.assertTrue(torch.allclose(compressed.tokens[2:], torch.zeros_like(compressed.tokens[2:])))
        self.assertFalse(torch.allclose(compressed.tokens[:2], torch.zeros_like(compressed.tokens[:2])))

    def test_visual_memory_compressor_rejects_unaligned_hidden_dim(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        entry = himem.VisualMemoryEntry({"base": torch.randn(2, 7)}, tau=0, eta=himem.SHORT_MEMORY)
        compressor = himem.VisualMemoryCompressor(hidden_dim=8, view_names=("base",), tokens_per_entry=1, num_heads=2)

        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            compressor([entry], current_step=16)

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
