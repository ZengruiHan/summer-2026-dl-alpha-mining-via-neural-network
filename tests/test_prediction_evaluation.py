from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.prediction_evaluation import (
    average_ranks,
    evaluate_one_day,
    pearson_correlation,
)


class PredictionEvaluationTest(unittest.TestCase):
    def test_average_ranks_handles_ties(self) -> None:
        ranks = average_ranks(np.array([3.0, 1.0, 1.0, 2.0]))
        np.testing.assert_allclose(ranks, [3.0, 0.5, 0.5, 2.0])

    def test_pearson_correlation_extremes(self) -> None:
        values = np.arange(10, dtype=np.float64)
        self.assertAlmostEqual(pearson_correlation(values, values), 1.0)
        self.assertAlmostEqual(pearson_correlation(values, -values), -1.0)

    def test_daily_metrics_are_exact_for_perfect_prediction(self) -> None:
        scores = np.arange(5, dtype=np.float64)
        returns = np.arange(5, dtype=np.float64)
        classes = np.arange(5, dtype=np.int8)
        rank_ic, ic, accuracy = evaluate_one_day(
            scores, returns, classes, classes
        )
        self.assertAlmostEqual(rank_ic, 1.0)
        self.assertAlmostEqual(ic, 1.0)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_accuracy_is_daily_mean(self) -> None:
        rank_ic, ic, accuracy = evaluate_one_day(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 1.0, 0.0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 2]),
        )
        self.assertAlmostEqual(rank_ic, -1.0)
        self.assertAlmostEqual(ic, -1.0)
        self.assertAlmostEqual(accuracy, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
