from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_mining_neural_network.standardize_features import (
    NUMERIC_FEATURES,
    apply_proposal_transforms,
    cross_sectional_winsor_zscore,
    sector_encoding,
)


class StandardizeFeaturesTest(unittest.TestCase):
    def test_feature_specific_transforms(self) -> None:
        frame = pd.DataFrame(
            {
                feature: [1.0]
                for feature in NUMERIC_FEATURES
            }
        )
        transformed = apply_proposal_transforms(frame)
        index = {feature: position for position, feature in enumerate(NUMERIC_FEATURES)}

        self.assertAlmostEqual(transformed[0, index["return_t_minus_1"]], 1.0)
        self.assertAlmostEqual(
            transformed[0, index["volatility_20"]], np.log(1.0 + 1e-12)
        )
        self.assertAlmostEqual(
            transformed[0, index["mean_dollar_volume_20"]], np.log(2.0)
        )
        self.assertAlmostEqual(
            transformed[0, index["mean_turnover_20"]], np.log(2.0)
        )

    def test_daily_winsor_zscore_has_zero_mean_unit_population_std(self) -> None:
        values = np.array(
            [[0.0], [1.0], [2.0], [1000.0], [10.0], [11.0], [12.0], [13.0]]
        )
        dates = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        standardized, stats = cross_sectional_winsor_zscore(values, dates)

        for group in (standardized[:4], standardized[4:]):
            self.assertAlmostEqual(float(np.mean(group)), 0.0)
            self.assertAlmostEqual(float(np.std(group, ddof=0)), 1.0)
        self.assertGreater(stats["upper_tail_values_clipped"], 0)

    def test_sector_encoding_is_stable_and_zero_based(self) -> None:
        indices, codes, columns = sector_encoding(pd.Series([7370, 2834, 7370]))

        self.assertEqual(codes, [2834, 7370])
        self.assertEqual(columns, ["sector_sic_2834", "sector_sic_7370"])
        self.assertEqual(indices.tolist(), [1, 0, 1])


if __name__ == "__main__":
    unittest.main()

