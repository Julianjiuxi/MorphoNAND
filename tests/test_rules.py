import unittest
import numpy as np

from morphonand.rules import BinaryAttractionForce, NearestTwoNAND


class RuleTests(unittest.TestCase):
    def test_force_signs(self):
        rule = BinaryAttractionForce(2.0, 0.2, 1.0, 1.0, 5.0)
        bits = np.array([0, 1, 1], dtype=np.uint8)
        d = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        s = rule.pair_scalar(bits, d)
        self.assertGreater(s[0, 1], 0.0)   # unlike attract
        self.assertLess(s[1, 2], 0.0)      # like repel

    def test_nand_receiver(self):
        rule = NearestTwoNAND()
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0]])
        bits = np.array([0, 1, 1], dtype=np.uint8)
        out, flips = rule.update(pos, bits, 10.0, 1.0, 2, np.random.default_rng(0), 1.0)
        self.assertEqual(int(out[0]), 0)  # NAND(1,1)=0
        self.assertGreaterEqual(flips, 0)


if __name__ == "__main__":
    unittest.main()
