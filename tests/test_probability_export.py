from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.probability_export import (
    validate_probability_arrays,
)


class ProbabilityExportTest(unittest.TestCase):
    def _arrays(self) -> dict[str, np.ndarray]:
        dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        permno = np.tile(np.arange(10_000, 10_500, dtype=np.int32), (2, 1))
        probabilities = np.full((2, 500, 5), 0.2, dtype=np.float32)
        inference_mask = np.ones((2, 500), dtype=bool)
        evaluation_mask = inference_mask.copy()
        return {
            "dates": dates,
            "permno": permno,
            "probabilities": probabilities,
            "inference_mask": inference_mask,
            "evaluation_mask": evaluation_mask,
        }

    def test_probability_contract_passes(self) -> None:
        record = validate_probability_arrays(**self._arrays())
        self.assertEqual(record["probability_shape"], [2, 500, 5])
        self.assertEqual(record["evaluation_nodes"], 1000)

    def test_masked_probability_must_be_nan(self) -> None:
        arrays = self._arrays()
        arrays["inference_mask"][0, 0] = False
        arrays["evaluation_mask"][0, 0] = False
        with self.assertRaisesRegex(RuntimeError, "must be NaN"):
            validate_probability_arrays(**arrays)

    def test_evaluation_mask_must_be_subset(self) -> None:
        arrays = self._arrays()
        arrays["inference_mask"][0, 0] = False
        arrays["probabilities"][0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "subset"):
            validate_probability_arrays(**arrays)


if __name__ == "__main__":
    unittest.main()
