from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alpha_mining_neural_network.top500 import compute_membership


class Top500Test(unittest.TestCase):
    def test_signal_excludes_current_date_and_uses_permno_tie_break(self) -> None:
        calendar = pd.date_range("2020-01-01", periods=4, freq="D")
        rows = []
        for permno, volumes in [(1, [10, 10, 1000, 10]), (2, [10, 10, 1, 10])]:
            for date, volume in zip(calendar, volumes, strict=True):
                rows.append(
                    {"permno": permno, "date": date.strftime("%Y-%m-%d"), "prc": 1.0, "vol": volume}
                )
        panel = pd.DataFrame(rows)

        membership = compute_membership(panel, calendar, window=2, top_n=2)
        third = membership[membership["date"] == calendar[2]]
        fourth = membership[membership["date"] == calendar[3]]

        self.assertEqual(third["permno"].tolist(), [1, 2])
        self.assertTrue(np.allclose(third["trailing_21d_dollar_volume"], [10, 10]))
        self.assertEqual(fourth["permno"].tolist(), [1, 2])
        self.assertTrue(np.allclose(fourth["trailing_21d_dollar_volume"], [505, 5.5]))

    def test_missing_market_date_invalidates_strict_window(self) -> None:
        calendar = pd.date_range("2020-01-01", periods=4, freq="D")
        panel = pd.DataFrame(
            {
                "permno": [1, 1, 1],
                "date": ["2020-01-01", "2020-01-03", "2020-01-04"],
                "prc": [1.0, 1.0, 1.0],
                "vol": [10.0, 10.0, 10.0],
            }
        )

        membership = compute_membership(panel, calendar, window=2, top_n=1)

        self.assertNotIn(calendar[2], membership["date"].tolist())
        self.assertNotIn(calendar[3], membership["date"].tolist())


if __name__ == "__main__":
    unittest.main()

