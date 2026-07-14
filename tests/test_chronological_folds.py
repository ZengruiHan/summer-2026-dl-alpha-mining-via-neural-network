from __future__ import annotations

import unittest

import pandas as pd

from alpha_mining_neural_network.chronological_folds import (
    generate_fold_specs,
    partition_labels,
)


class ChronologicalFoldsTest(unittest.TestCase):
    def test_2000_2025_produces_seventeen_rolling_folds(self) -> None:
        folds = generate_fold_specs(2000, 2025)

        self.assertEqual(len(folds), 17)
        self.assertEqual(folds[0].years("train"), list(range(2000, 2008)))
        self.assertEqual(folds[0].validation_year, 2008)
        self.assertEqual(folds[0].years("refit"), list(range(2000, 2009)))
        self.assertEqual(folds[0].test_year, 2009)
        self.assertEqual(folds[-1].years("train"), list(range(2016, 2024)))
        self.assertEqual(folds[-1].validation_year, 2024)
        self.assertEqual(folds[-1].test_year, 2025)

    def test_target_crossing_year_boundary_is_purged(self) -> None:
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"]),
                "permno": [1, 1, 1],
                "rank": [1, 1, 1],
                "target_date": pd.to_datetime(
                    ["2020-12-31", "2021-01-04", "2021-01-05"]
                ),
                "target_return": [0.1, 0.2, 0.3],
                "label": pd.Series([1, 2, 0], dtype="Int8"),
            }
        )
        complete = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-12-30", "2020-12-31"]),
                "permno": [1, 1],
                "complete_numeric_features": [True, True],
            }
        )

        partition, stats = partition_labels(
            labels, year=2020, complete_features=complete
        )

        self.assertEqual(partition["date"].tolist(), [pd.Timestamp("2020-12-30")])
        self.assertEqual(stats["boundary_purged_rows"], 1)


if __name__ == "__main__":
    unittest.main()
