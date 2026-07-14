from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_mining_neural_network.features_labels import (
    compute_labels,
    compute_return_features,
)


class FeaturesLabelsTest(unittest.TestCase):
    def test_proposal_return_blocks_and_no_current_return_leakage(self) -> None:
        calendar = pd.date_range("2020-01-01", periods=22, freq="D")
        returns = np.arange(22, dtype=np.float64) / 100.0
        matrix = returns.reshape(1, -1)
        member_asset = np.array([0], dtype=np.int64)
        member_date = np.array([20], dtype=np.int16)

        first = compute_return_features(
            matrix.copy(), np.linspace(-0.02, 0.02, 22), member_asset, member_date
        )
        changed = matrix.copy()
        changed[0, 20] = 999.0
        second = compute_return_features(
            changed, np.linspace(-0.02, 0.02, 22), member_asset, member_date
        )

        self.assertAlmostEqual(first["return_t_minus_1"][0], 0.19)
        self.assertAlmostEqual(
            first["mean_return_t_minus_6_to_t_minus_2"][0],
            np.mean(returns[14:19]),
        )
        self.assertAlmostEqual(
            first["mean_return_t_minus_20_to_t_minus_7"][0],
            np.mean(returns[0:14]),
        )
        for name in first:
            if name != "beta_60":
                self.assertTrue(np.allclose(first[name], second[name], equal_nan=True))

    def test_labels_use_next_date_cross_section_quantiles(self) -> None:
        calendar = pd.date_range("2020-01-01", periods=2, freq="D")
        matrix = np.array(
            [
                [0.0, -0.2],
                [0.0, -0.1],
                [0.0, 0.0],
                [0.0, 0.1],
                [0.0, 0.2],
            ]
        )
        member_asset = np.arange(5)
        member_date = np.zeros(5, dtype=np.int16)

        target_date, target_return, label = compute_labels(
            matrix, member_asset, member_date, calendar
        )

        self.assertTrue((target_date == np.datetime64("2020-01-02")).all())
        self.assertTrue(np.allclose(target_return, [-0.2, -0.1, 0.0, 0.1, 0.2]))
        self.assertEqual(label.tolist(), [-2.0, -1.0, 0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()

