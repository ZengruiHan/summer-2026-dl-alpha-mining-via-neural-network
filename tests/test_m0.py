from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.m0 import (
    PreparedSplit,
    daily_loss_accuracy,
    fit_candidate,
    initialize_model,
    model_logits,
    predict_probabilities,
    softmax_and_true_class_nll,
    stable_softmax,
)


class M0Test(unittest.TestCase):
    def test_sparse_sector_lookup_equals_dense_one_hot(self) -> None:
        rng = np.random.default_rng(7)
        parameters = initialize_model(3, 4)
        parameters.numeric_weight[:] = rng.normal(size=(3, 5))
        parameters.sector_weight[:] = rng.normal(size=(4, 5))
        parameters.bias[:] = rng.normal(size=5)
        x = rng.normal(size=(11, 3)).astype(np.float32)
        sector = rng.integers(0, 4, size=11)

        sparse = model_logits(parameters, x, sector)
        dense_x = np.concatenate((x, np.eye(4, dtype=np.float32)[sector]), axis=1)
        dense_weight = np.concatenate(
            (parameters.numeric_weight, parameters.sector_weight), axis=0
        )
        dense = dense_x @ dense_weight + parameters.bias

        np.testing.assert_allclose(sparse, dense, rtol=1e-6, atol=1e-6)

    def test_softmax_is_finite_and_normalized_for_large_logits(self) -> None:
        logits = np.array([[1000.0, 999.0, -1000.0, 0.0, 1.0]], dtype=np.float32)
        probabilities = stable_softmax(logits)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)

    def test_true_class_nll_is_not_clipped_for_extreme_error(self) -> None:
        logits = np.array([[1000.0, -1000.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        probabilities, nll = softmax_and_true_class_nll(
            logits, np.array([1], dtype=np.int64)
        )

        self.assertEqual(probabilities[0, 1], 0.0)
        self.assertAlmostEqual(float(nll[0]), 2000.0, places=3)

    def test_loss_is_date_equal_not_flat_node_equal(self) -> None:
        y = np.zeros((2, 3), dtype=np.int8)
        mask = np.array([[True, False, False], [True, True, True]])
        probabilities = np.zeros((2, 3, 5), dtype=np.float32)
        probabilities[..., 0] = np.array([[0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
        probabilities[..., 1:] = (1.0 - probabilities[..., :1]) / 4.0

        loss, _, nodes = daily_loss_accuracy(y, mask, probabilities)
        expected = (-np.log(0.9) - np.log(0.5)) / 2.0
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(nodes, 4)

    def test_candidate_training_improves_validation_cross_entropy(self) -> None:
        rng = np.random.default_rng(11)
        date_count = 50
        node_count = 20
        x = rng.normal(size=(date_count, node_count, 2)).astype(np.float32)
        sector = rng.integers(0, 3, size=(date_count, node_count), dtype=np.int16)
        latent = 2.5 * x[..., 0] - 1.5 * x[..., 1]
        y = np.digitize(latent, [-1.0, -0.25, 0.25, 1.0]).astype(np.int8)
        mask = np.ones_like(y, dtype=bool)
        dates = np.arange(date_count).astype("datetime64[D]")
        train = PreparedSplit("train", [2000], dates, x, sector, y, mask)
        validation = PreparedSplit("validation", [2001], dates, x, sector, y, mask)
        config = {
            "numeric_feature_count": 2,
            "sector_category_count": 3,
            "class_count": 5,
            "max_epochs": 12,
            "patience": 12,
            "min_delta": 0.0,
            "batch_dates": 10,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
        }
        candidate = {"id": "test", "learning_rate": 0.05, "l2": 0.0}

        model, record = fit_candidate(
            train, validation, config=config, candidate=candidate, seed=9
        )
        probabilities = predict_probabilities(model, validation, chunk_dates=25)
        final_loss, _, _ = daily_loss_accuracy(y, mask, probabilities)

        self.assertLess(final_loss, np.log(5.0) - 0.2)
        self.assertEqual(record["best_epoch"], 12)


if __name__ == "__main__":
    unittest.main()
