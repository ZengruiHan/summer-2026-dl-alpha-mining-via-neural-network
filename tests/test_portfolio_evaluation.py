from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.portfolio_evaluation import (
    aligned_turnover,
    performance_summary,
    strict_portfolio_return,
    zero_contribution_portfolio_return,
)


class PortfolioEvaluationTest(unittest.TestCase):
    def test_turnover_aligns_by_permno_not_column(self) -> None:
        previous_permno = np.array([10, 20, 30])
        previous_weight = np.array([0.5, -0.5, 0.0])
        current_permno = np.array([20, 30, 10])
        current_weight = np.array([-0.5, 0.0, 0.5])

        turnover = aligned_turnover(
            current_permno,
            current_weight,
            previous_permno,
            previous_weight,
        )

        self.assertEqual(turnover, 0.0)

    def test_first_day_turnover_starts_from_cash(self) -> None:
        turnover = aligned_turnover(
            np.array([10, 20]),
            np.array([0.5, -0.5]),
            None,
            None,
        )
        self.assertEqual(turnover, 1.0)

    def test_missing_held_return_invalidates_day_without_reweighting(self) -> None:
        gross, missing = strict_portfolio_return(
            np.array([0.5, -0.5, 0.0]),
            np.array([0.1, np.nan, np.nan]),
        )
        self.assertTrue(np.isnan(gross))
        self.assertEqual(missing, 1)

    def test_zero_contribution_policy_keeps_original_weights(self) -> None:
        gross, missing = zero_contribution_portfolio_return(
            np.array([0.5, -0.5, 0.0]),
            np.array([0.1, np.nan, np.nan]),
        )
        self.assertAlmostEqual(gross, 0.05)
        self.assertEqual(missing, 1)

    def test_performance_summary_uses_sample_sharpe_and_compounding(self) -> None:
        gross = np.array([0.01, -0.02, 0.03], dtype=np.float64)
        turnover = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        summary, arrays = performance_summary(
            gross, turnover, transaction_cost_bps=0.0
        )
        expected_sharpe = np.sqrt(252.0) * gross.mean() / gross.std(ddof=1)
        expected_cumulative = np.prod(1.0 + gross) - 1.0
        self.assertAlmostEqual(summary["sharpe"], expected_sharpe)
        self.assertAlmostEqual(summary["cumulative_net_return"], expected_cumulative)
        self.assertAlmostEqual(summary["mean_daily_turnover"], turnover.mean())
        np.testing.assert_allclose(arrays["net_return"], gross)


if __name__ == "__main__":
    unittest.main()
