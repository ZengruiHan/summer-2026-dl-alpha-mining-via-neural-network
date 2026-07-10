from __future__ import annotations

import unittest

import numpy as np

from models.M0.plot_oos_results import compute_fold_contributions


class M0PlottingTest(unittest.TestCase):
    def test_fold_contributions_add_to_total_compounded_return(self) -> None:
        net_return = np.array([0.10, -0.05, 0.02, 0.03], dtype=np.float64)
        fold_index = np.array([0, 0, 1, 1], dtype=np.int8)

        standalone, contribution = compute_fold_contributions(
            net_return, fold_index, fold_count=2
        )
        total = np.prod(1.0 + net_return) - 1.0

        self.assertAlmostEqual(standalone[0], 1.10 * 0.95 - 1.0)
        self.assertAlmostEqual(standalone[1], 1.02 * 1.03 - 1.0)
        self.assertAlmostEqual(contribution.sum(), total)

    def test_missing_daily_return_is_skipped_within_fold(self) -> None:
        net_return = np.array([0.01, np.nan, 0.02], dtype=np.float64)
        fold_index = np.zeros(3, dtype=np.int8)

        standalone, contribution = compute_fold_contributions(
            net_return, fold_index, fold_count=1
        )

        expected = 1.01 * 1.02 - 1.0
        self.assertAlmostEqual(standalone[0], expected)
        self.assertAlmostEqual(contribution[0], expected)


if __name__ == "__main__":
    unittest.main()
