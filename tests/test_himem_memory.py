import unittest


class HiMemMemoryTests(unittest.TestCase):
    def test_memory_bank_write_read_and_reset(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        bank = himem.EpisodeMemoryBank(max_tokens=3, token_dim=4)
        self.assertEqual(bank.write("episode-a", torch.ones(2, 4)), 2)
        self.assertEqual(bank.write("episode-a", torch.arange(8, dtype=torch.float32).view(2, 4)), 2)
        self.assertEqual(bank.episode_length("episode-a"), 3)

        query = torch.randn(2, 4)
        memory = bank.read("episode-a", query, top_k=2)
        self.assertEqual(tuple(memory.shape), (2, 2, 4))

        bank.reset("episode-a")
        self.assertEqual(bank.episode_length("episode-a"), 0)
        empty = bank.read("episode-a", query)
        self.assertEqual(tuple(empty.shape), (2, 0, 4))

    def test_memory_bank_gate_filters_writes(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        bank = himem.EpisodeMemoryBank(max_tokens=4, token_dim=2)
        tokens = torch.ones(3, 2)
        written = bank.write("episode-b", tokens, gate=torch.tensor([0.1, 0.8, 0.2]), threshold=0.5)

        self.assertEqual(written, 1)
        self.assertEqual(bank.episode_length("episode-b"), 1)

    def test_himem_token_writer_shape(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        writer = himem.HiMemTokenWriter(hidden_dim=8, num_tokens=3, num_heads=2)
        output = writer(torch.randn(2, 5, 8))

        self.assertEqual(tuple(output.shape), (2, 3, 8))

    def test_hierarchical_memory_accumulates_segment_before_write(self):
        torch = self._import_or_skip("torch")
        himem = self._import_or_skip("himem_bridge_vla.model.himem")

        bank = himem.EpisodeMemoryBank(max_tokens=4, token_dim=2)
        memory = himem.HierarchicalEpisodeMemory(
            bank=bank,
            read_top_k=2,
            write_threshold=0.5,
            segment_accumulator="ema",
            segment_ema_decay=0.5,
        )

        self.assertEqual(memory.write("episode-c", torch.ones(2, 2), gate=0.1), 0)
        self.assertEqual(memory.segment_length("episode-c"), 2)
        self.assertEqual(bank.episode_length("episode-c"), 0)

        written = memory.write("episode-c", torch.full((2, 2), 3.0), gate=0.9)

        self.assertEqual(written, 2)
        self.assertEqual(memory.segment_length("episode-c"), 0)
        self.assertEqual(bank.episode_length("episode-c"), 2)
        read = memory.read("episode-c", torch.randn(1, 2))
        self.assertEqual(tuple(read.shape), (1, 2, 2))

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
