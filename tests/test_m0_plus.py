from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.m0 import PreparedSplit
from alpha_mining_neural_network.m0_plus import (
    build_sparse_features,
    daily_equal_weights,
    place_probabilities,
    validate_config,
)


class M0PlusTest(unittest.TestCase):
    def _split(self) -> PreparedSplit:
        x = np.array(
            [
                [[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]],
                [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
            ],
            dtype=np.float32,
        )
        sector = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int16)
        y = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int8)
        mask = np.ones_like(y, dtype=bool)
        dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        return PreparedSplit("train", [2020], dates, x, sector, y, mask)

    def test_sparse_features_keep_numeric_zeros_and_sector_one_hot(self) -> None:
        split = self._split()
        mask = np.array([[True, False, True], [False, False, False]])

        matrix = build_sparse_features(split, mask, sector_category_count=4)

        self.assertEqual(matrix.shape, (2, 6))
        self.assertEqual(matrix.nnz, 6)
        np.testing.assert_allclose(
            matrix.toarray(),
            np.array(
                [[0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                 [3.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )

    def test_daily_equal_weights_equalize_dates(self) -> None:
        mask = np.array([[True, False, False], [True, True, True]])

        weights = daily_equal_weights(mask)

        self.assertAlmostEqual(float(weights[:1].sum()), 2.0)
        self.assertAlmostEqual(float(weights[1:].sum()), 2.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_place_probabilities_uses_nan_outside_mask(self) -> None:
        mask = np.array([[True, False], [False, True]])
        selected = np.array(
            [[0.1, 0.2, 0.3, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]],
            dtype=np.float32,
        )

        probabilities = place_probabilities(mask, selected)

        np.testing.assert_allclose(probabilities[mask], selected)
        self.assertTrue(np.isnan(probabilities[~mask]).all())

    def test_config_requires_five_classes(self) -> None:
        config = {
            "model": "M0-Plus",
            "model_family": "xgboost_multiclass",
            "selection_metric": "validation_daily_mean_cross_entropy",
            "numeric_feature_count": 8,
            "sector_category_count": 690,
            "class_count": 4,
            "num_boost_round": 10,
            "early_stopping_rounds": 2,
            "tree_method": "hist",
            "max_bin": 64,
            "candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "five output classes"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
