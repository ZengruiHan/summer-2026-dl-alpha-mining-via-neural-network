from __future__ import annotations

import unittest

import numpy as np

from alpha_mining_neural_network.cellular import (
    correlation_matrix,
    directed_threshold,
    undirected_threshold,
    undirected_top_k,
)


class CellularTest(unittest.TestCase):
    def test_correlation_and_top_k_are_deterministic(self) -> None:
        values = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        corr = correlation_matrix(values)
        source, target, weight = undirected_top_k(corr, 1)

        edges = {tuple(pair) for pair in zip(source, target, strict=True)}
        self.assertEqual(len(edges), 4)
        self.assertTrue(all((target, source) in edges for source, target in edges))
        self.assertTrue(np.allclose(np.abs(weight), 1.0))

    def test_threshold_edges_and_direction(self) -> None:
        matrix = np.array([[0.0, 0.5, -0.2], [0.1, 0.0, 0.8], [0.3, -0.9, 0.0]])
        source, target, weight = undirected_threshold(matrix, 0.5)
        self.assertEqual(len(source), 4)
        self.assertEqual(
            set(zip(source, target, strict=True)),
            {(0, 1), (1, 0), (1, 2), (2, 1)},
        )
        source, target, _ = directed_threshold(matrix, 0.5)
        self.assertEqual(set(zip(source, target, strict=True)), {(0, 1), (1, 2), (2, 1)})


if __name__ == "__main__":
    unittest.main()
