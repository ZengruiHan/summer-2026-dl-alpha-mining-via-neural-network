from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.portfolio_construction import (
    construct_daily_portfolios,
    probability_to_score,
)


class PortfolioConstructionTest(unittest.TestCase):
    def test_probability_to_expected_class_score(self) -> None:
        probabilities = np.array(
            [[[0.1, 0.2, 0.3, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        scores = probability_to_score(probabilities)
        np.testing.assert_allclose(scores, [[0.2, 2.0]], atol=1e-7)

    def test_portfolio_has_exact_exposures_and_tail_sizes(self) -> None:
        scores = np.arange(500, dtype=np.float32)[None, :]
        permno = np.arange(10_000, 10_500, dtype=np.int32)[None, :]
        mask = np.ones((1, 500), dtype=bool)

        result = construct_daily_portfolios(scores, permno, mask)

        self.assertEqual(result["long_leg_size"].tolist(), [100])
        self.assertEqual(result["short_leg_size"].tolist(), [100])
        self.assertAlmostEqual(float(result["weights"].sum()), 0.0, places=7)
        self.assertAlmostEqual(float(np.abs(result["weights"]).sum()), 1.0, places=6)
        self.assertTrue(np.all(result["positions"][0, :100] == -1))
        self.assertTrue(np.all(result["positions"][0, -100:] == 1))

    def test_mask_and_floor_leg_rule(self) -> None:
        scores = np.arange(500, dtype=np.float32)[None, :]
        permno = np.arange(20_000, 20_500, dtype=np.int32)[None, :]
        mask = np.ones((1, 500), dtype=bool)
        mask[0, 250] = False
        scores[0, 250] = np.nan

        result = construct_daily_portfolios(scores, permno, mask)

        self.assertEqual(result["eligible_count"].tolist(), [499])
        self.assertEqual(result["long_leg_size"].tolist(), [99])
        self.assertEqual(result["short_leg_size"].tolist(), [99])
        self.assertEqual(result["ordinal_rank"][0, 250], -1)
        self.assertEqual(result["weights"][0, 250], 0.0)

    def test_permno_breaks_exact_score_ties_deterministically(self) -> None:
        scores = np.zeros((1, 500), dtype=np.float32)
        permno = np.arange(500, 0, -1, dtype=np.int32)[None, :]
        mask = np.ones((1, 500), dtype=bool)

        result = construct_daily_portfolios(scores, permno, mask)
        ranks_by_permno = result["ordinal_rank"][0, np.argsort(permno[0])]

        np.testing.assert_array_equal(ranks_by_permno, np.arange(1, 501))


if __name__ == "__main__":
    unittest.main()
