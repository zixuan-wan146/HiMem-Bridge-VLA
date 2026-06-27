import unittest

import numpy as np

from himem_bridge_vla.runtime_config import TARGET_STATE_DIM, build_action_mask, normalize_mask, pad_1d


class RuntimeConfigTests(unittest.TestCase):
    def test_pad_1d_pads_short_vector(self):
        padded = pad_1d([1.0, 2.0], target_dim=4, fill_value=-1.0)

        np.testing.assert_array_equal(padded, np.array([1.0, 2.0, -1.0, -1.0], dtype=np.float32))

    def test_pad_1d_rejects_long_vector(self):
        with self.assertRaisesRegex(ValueError, "exceeds target dimension"):
            pad_1d(range(TARGET_STATE_DIM + 1), target_dim=TARGET_STATE_DIM)

    def test_build_action_mask(self):
        self.assertEqual(build_action_mask(3, target_dim=5), [1, 1, 1, 0, 0])

    def test_normalize_mask_pads_and_casts(self):
        self.assertEqual(normalize_mask([1, 0], target_dim=4), [1, 0, 0, 0])

    def test_normalize_mask_rejects_long_mask(self):
        with self.assertRaisesRegex(ValueError, "exceeds target dimension"):
            normalize_mask([1, 0, 1], target_dim=2)


if __name__ == "__main__":
    unittest.main()
