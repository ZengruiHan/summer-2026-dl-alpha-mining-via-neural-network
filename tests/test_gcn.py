from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alpha_mining_neural_network.gcn import (
    GCNParameters,
    PreparedGraphSplit,
    adam_update,
    batch_objective_and_gradients,
    daily_cross_entropy_error,
    fit_candidate,
    gcn_forward,
    initialize_adam,
    initialize_gcn,
    normalized_graph,
    predict_probabilities,
    predict_probabilities_with_metrics,
    sector_graph,
    strict_loss_accuracy,
)
from alpha_mining_neural_network.m0 import PreparedSplit
from alpha_mining_neural_network.m2_gcn import train_all_folds, validate_config
from alpha_mining_neural_network.probability_export import export_test_probabilities


class GraphOperatorTest(unittest.TestCase):
    def test_sparse_forward_and_transpose_match_dense_matrix(self) -> None:
        graph = normalized_graph(
            4,
            sender=np.array([0, 2]),
            receiver=np.array([1, 1]),
            weight=np.array([1.0, -1.0]),
            dtype=np.float64,
        )
        values = np.arange(12, dtype=np.float64).reshape(4, 3) / 7.0
        gradient = np.flip(values, axis=0).copy()
        dense = graph.to_dense()
        expected_dense = np.array(
            [
                [1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0],
                [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(3.0), -1.0 / np.sqrt(6.0), 0.0],
                [0.0, 0.0, 1.0 / np.sqrt(2.0), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(dense, expected_dense)
        np.testing.assert_allclose(graph.aggregate(values), dense @ values)
        np.testing.assert_allclose(
            graph.aggregate(gradient, transpose=True), dense.T @ gradient
        )
        self.assertTrue(np.isfinite(graph.weight).all())
        self.assertTrue((graph.weight < 0).any())

    def test_directed_edge_sends_only_source_message_to_destination(self) -> None:
        graph = normalized_graph(3, sender=[0], receiver=[1], weight=[1.0])
        values = np.array([2.0, 3.0, 0.0], dtype=np.float32)

        output = graph.aggregate(values)

        self.assertAlmostEqual(float(output[0]), np.sqrt(2.0), places=6)
        self.assertAlmostEqual(float(output[1]), 1.0 + 3.0 / np.sqrt(2.0), places=6)
        self.assertEqual(float(output[2]), 0.0)

    def test_normalization_rejects_integer_output_dtype(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating"):
            normalized_graph(2, [0], [1], dtype=np.int32)

    def test_empty_graph_is_identity_after_self_loops(self) -> None:
        graph = normalized_graph(3, sender=[], receiver=[])
        np.testing.assert_allclose(graph.to_dense(), np.eye(3), atol=0.0)

    def test_sector_graph_averages_each_sector_clique(self) -> None:
        graph = sector_graph(np.array([7, 7, 9], dtype=np.int16))
        expected = np.array(
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(graph.to_dense(), expected, atol=1e-7)


class GCNMathTest(unittest.TestCase):
    def test_sector_lookup_equals_materialized_one_hot(self) -> None:
        parameters = initialize_gcn(2, 4, 3, seed=12, dtype=np.float64)
        parameters.hidden_bias[:] = np.array([0.2, 0.3, 0.4])
        x = np.array([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.4]])
        sector = np.array([0, 3, 1], dtype=np.int16)
        graph = normalized_graph(3, [], [], dtype=np.float64)

        _, cache = gcn_forward(parameters, x, sector, graph)
        dense_x = np.concatenate((x, np.eye(4)[sector]), axis=1)
        dense_weight = np.concatenate(
            (parameters.numeric_weight, parameters.sector_weight), axis=0
        )
        expected = dense_x @ dense_weight + parameters.hidden_bias

        np.testing.assert_allclose(cache.hidden_pre_activation, expected)

    def test_strict_sector_clique_intentionally_ties_member_logits(self) -> None:
        parameters = initialize_gcn(2, 2, 4, seed=3, dtype=np.float64)
        parameters.hidden_bias[:] = 1.0
        x = np.array([[2.0, -1.0], [-3.0, 4.0], [0.5, 0.1]])
        sector = np.array([0, 0, 1], dtype=np.int16)
        graph = sector_graph(sector)

        logits, _ = gcn_forward(parameters, x, sector, graph)

        np.testing.assert_allclose(logits[0], logits[1], atol=1e-12)

    def test_daily_equal_loss_weights_dates_not_flat_nodes(self) -> None:
        logits_one = np.array([[2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        logits_three = np.zeros((3, 5), dtype=np.float64)
        loss_one, _, _, _ = daily_cross_entropy_error(
            logits_one,
            np.array([0, -100]),
            np.array([True, False]),
            date_weight=0.5,
        )
        loss_three, _, _, _ = daily_cross_entropy_error(
            logits_three,
            np.array([0, 1, 2]),
            np.ones(3, dtype=bool),
            date_weight=0.5,
        )
        expected_one = -np.log(np.exp(2.0) / (np.exp(2.0) + 4.0))
        expected = 0.5 * expected_one + 0.5 * np.log(5.0)

        self.assertAlmostEqual(loss_one + loss_three, expected, places=12)

    def test_cross_entropy_rejects_integer_logits(self) -> None:
        with self.assertRaisesRegex(TypeError, "floating"):
            daily_cross_entropy_error(
                np.zeros((2, 5), dtype=np.int32),
                np.array([0, 1]),
                np.ones(2, dtype=bool),
            )

    def test_full_backward_matches_central_differences(self) -> None:
        rng = np.random.default_rng(4)
        date_count, node_count = 2, 3
        x = rng.normal(scale=0.2, size=(date_count, node_count, 2))
        sector = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int16)
        y = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int8)
        mask = np.array([[True, True, False], [True, False, True]])
        dates = np.arange(date_count).astype("datetime64[D]")
        data = PreparedSplit("train", [2000], dates, x, sector, y, mask)
        graphs = (
            normalized_graph(node_count, [0, 1], [1, 2], [0.7, -0.4], dtype=np.float64),
            normalized_graph(node_count, [2], [0], [0.5], dtype=np.float64),
        )
        split = PreparedGraphSplit(data, graphs, relation="test")
        parameters = initialize_gcn(2, 3, 3, seed=8, dtype=np.float64)
        parameters.hidden_bias[:] = 1.0
        objective, gradients = batch_objective_and_gradients(
            parameters, split, [0, 1], l2=0.03
        )
        self.assertTrue(np.isfinite(objective))

        epsilon = 1e-6
        max_error = 0.0
        for parameter, analytic in zip(
            parameters.arrays(), gradients.arrays(), strict=True
        ):
            for index in np.ndindex(parameter.shape):
                original = float(parameter[index])
                parameter[index] = original + epsilon
                plus, _ = batch_objective_and_gradients(
                    parameters, split, [0, 1], l2=0.03
                )
                parameter[index] = original - epsilon
                minus, _ = batch_objective_and_gradients(
                    parameters, split, [0, 1], l2=0.03
                )
                parameter[index] = original
                numerical = (plus - minus) / (2.0 * epsilon)
                max_error = max(max_error, abs(numerical - float(analytic[index])))

        self.assertLess(max_error, 2e-6)

    def test_adam_uses_one_shared_step_per_batch(self) -> None:
        parameters = initialize_gcn(1, 2, 2, seed=5, dtype=np.float64)
        original = [value.copy() for value in parameters.arrays()]
        gradients = GCNParameters(
            *(np.ones_like(value) for value in parameters.arrays())
        )
        state = initialize_adam(parameters)

        for _ in range(2):
            adam_update(
                parameters,
                gradients,
                state,
                learning_rate=0.1,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-12,
            )

        self.assertEqual(state.step, 2)
        for value, start in zip(parameters.arrays(), original, strict=True):
            np.testing.assert_allclose(value, start - 0.2, atol=3e-12)


class GCNTrainingTest(unittest.TestCase):
    def _synthetic_split(
        self, rng: np.random.Generator, name: str, date_count: int
    ) -> PreparedGraphSplit:
        node_count = 12
        x = rng.normal(size=(date_count, node_count, 2)).astype(np.float32)
        latent = 2.2 * x[..., 0] - 1.3 * x[..., 1]
        y = np.digitize(latent, [-1.0, -0.3, 0.3, 1.0]).astype(np.int8)
        sector = np.broadcast_to(
            np.arange(node_count, dtype=np.int16), (date_count, node_count)
        ).copy()
        mask = np.ones_like(y, dtype=bool)
        dates = np.arange(date_count).astype("datetime64[D]")
        data = PreparedSplit(name, [2000], dates, x, sector, y, mask)
        graphs = tuple(normalized_graph(node_count, [], []) for _ in range(date_count))
        return PreparedGraphSplit(data, graphs, relation="identity-test")

    def test_training_learns_synthetic_node_labels(self) -> None:
        rng = np.random.default_rng(22)
        train = self._synthetic_split(rng, "train", 32)
        validation = self._synthetic_split(rng, "validation", 12)
        config = {
            "numeric_feature_count": 2,
            "sector_category_count": 12,
            "class_count": 5,
            "max_epochs": 35,
            "patience": 35,
            "min_delta": 0.0,
            "batch_dates": 8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "gradient_clip_norm": 5.0,
        }
        candidate = {
            "id": "test",
            "hidden_dim": 16,
            "learning_rate": 0.02,
            "l2": 0.0,
        }

        model, record = fit_candidate(
            train, validation, config=config, candidate=candidate, seed=9
        )
        probabilities = predict_probabilities(model, validation)
        combined_probabilities, combined_metrics = predict_probabilities_with_metrics(
            model, validation
        )
        strict_metrics = strict_loss_accuracy(model, validation)

        self.assertLess(
            record["best_validation_daily_mean_cross_entropy"], np.log(5.0) - 0.35
        )
        self.assertEqual(probabilities.shape, (12, 12, 5))
        np.testing.assert_array_equal(probabilities, combined_probabilities)
        np.testing.assert_allclose(combined_metrics[:2], strict_metrics[:2], atol=1e-12)
        self.assertEqual(combined_metrics[2], strict_metrics[2])
        np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=2e-6)

    def test_m2c_config_requires_five_classes(self) -> None:
        config = {
            "model": "M2-C",
            "model_family": "gcn_multiclass",
            "graph_relation": "sector",
            "optimizer": "adam",
            "selection_metric": "validation_daily_mean_cross_entropy",
            "numeric_feature_count": 8,
            "sector_category_count": 690,
            "class_count": 4,
            "max_epochs": 1,
            "patience": 1,
            "batch_dates": 1,
        }
        with self.assertRaisesRegex(ValueError, "five output classes"):
            validate_config(config)


class M2WorkflowTest(unittest.TestCase):
    def _write_split(
        self, split_dir: Path, name: str, seed: int, years: list[int]
    ) -> None:
        node_count, feature_count = 500, 2
        for offset, year in enumerate(years):
            rng = np.random.default_rng(seed + offset)
            dates = np.array([f"{year:04d}-01-03"], dtype="datetime64[D]")
            x = rng.normal(size=(1, node_count, feature_count)).astype(np.float32)
            sector = np.arange(node_count, dtype=np.int16)[None, :]
            y = np.digitize(x[..., 0], [-1.0, -0.3, 0.3, 1.0]).astype(np.int8)
            mask = np.ones((1, node_count), dtype=bool)
            permno = np.arange(10_000, 10_000 + node_count, dtype=np.int32)[None, :]
            target_return = x[..., 0].astype(np.float32)
            target_date = np.broadcast_to(
                (dates + np.timedelta64(1, "D"))[:, None],
                (1, node_count),
            ).copy()
            feature_dir = split_dir / "features" / f"year={year}"
            supervision_dir = split_dir / "supervision" / f"year={year}"
            feature_dir.mkdir(parents=True)
            supervision_dir.mkdir(parents=True)
            for filename, values in {
                "dates.npy": dates,
                "x_numeric.npy": x,
                "sector_index.npy": sector,
                "complete_numeric_mask.npy": mask,
                "permno.npy": permno,
            }.items():
                np.save(feature_dir / filename, values, allow_pickle=False)
            for filename, values in {
                "y.npy": y,
                "loss_mask.npy": mask,
                "day_model_mask.npy": mask.any(axis=1),
                "y_signed.npy": (y - 2).astype(np.int8),
                "target_return.npy": target_return,
                "target_date.npy": target_date,
            }.items():
                np.save(supervision_dir / filename, values, allow_pickle=False)
        (split_dir / "ready_split_manifest.json").write_text(
            json.dumps({"split": name, "years": years}) + "\n",
            encoding="utf-8",
        )

    def test_fold_workflow_writes_expected_prediction_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "ready"
            fold_dir = input_dir / "fold_00"
            split_years = {
                "train": [2000],
                "validation": [2001],
                "refit": [2000, 2001],
                "test": [2002],
            }
            for index, (split_name, years) in enumerate(split_years.items()):
                self._write_split(fold_dir / split_name, split_name, 20 + index, years)
            (fold_dir / "ready_fold_manifest.json").write_text(
                json.dumps(
                    {
                        "fold": {
                            "index": 0,
                            "name": "fold_00",
                            "train_start_year": 2000,
                            "train_end_year": 2000,
                            "validation_year": 2001,
                            "test_year": 2002,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            config = {
                "model": "M2-C",
                "model_family": "gcn_multiclass",
                "graph_relation": "sector",
                "seed": 7,
                "numeric_feature_count": 2,
                "sector_category_count": 500,
                "class_count": 5,
                "optimizer": "adam",
                "selection_metric": "validation_daily_mean_cross_entropy",
                "max_epochs": 1,
                "patience": 1,
                "min_delta": 0.0,
                "batch_dates": 2,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "gradient_clip_norm": 5.0,
                "candidates": [
                    {
                        "id": "smoke",
                        "hidden_dim": 4,
                        "learning_rate": 0.01,
                        "l2": 0.0,
                    }
                ],
            }
            (input_dir / "ready_for_use_manifest.json").write_text(
                json.dumps(
                    {
                        "numeric_feature_count": 2,
                        "sector_category_count": 500,
                        "fold_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "feature_spec.json").write_text(
                json.dumps(
                    {
                        "numeric_columns": ["f0", "f1"],
                        "combined_fixed_feature_dimension": 502,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

            manifest = train_all_folds(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config_path,
            )
            resumed = train_all_folds(
                input_dir=input_dir,
                output_dir=output_dir,
                config_path=config_path,
            )
            published = export_test_probabilities(
                source_dir=output_dir,
                output_dir=root / "published",
            )

            probabilities = np.load(
                output_dir / "fold_00" / "predictions" / "test" / "probabilities.npy",
                allow_pickle=False,
            )
            record = manifest["folds"][0]
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(resumed["run_fingerprint"], manifest["run_fingerprint"])
            self.assertEqual(record["status"], "complete")
            self.assertTrue(record["test"]["probabilities_generated_once"])
            self.assertEqual(published["model"], "M2-C")
            self.assertEqual(probabilities.shape, (1, 500, 5))
            np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
