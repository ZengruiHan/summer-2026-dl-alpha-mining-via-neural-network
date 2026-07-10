from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from alpha_mining_neural_network.crsp_download import (
    OUTPUT_COLUMNS,
    download_crsp_daily,
    month_intervals,
)


class FakeConnection:
    def raw_sql(self, sql: str, **kwargs: object) -> pd.DataFrame:
        if "MAX(dlycaldt)" in sql:
            return pd.DataFrame({"max_date": [date(2026, 1, 2)]})
        params = kwargs["params"]
        assert isinstance(params, dict)
        start = pd.Timestamp(params["start_date"])
        values: dict[str, list[object]] = {column: [pd.NA] for column in OUTPUT_COLUMNS}
        values.update(
            {
                "permno": [10001],
                "permco": [1000],
                "date": [start],
                "prc": [10.0],
                "ret": [0.01],
                "retx": [0.01],
                "vol": [100.0],
                "mktcap": [1000.0],
            }
        )
        return pd.DataFrame(values)

    def close(self) -> None:
        pass


class CrspDownloadTest(unittest.TestCase):
    def test_month_intervals_are_clipped(self) -> None:
        self.assertEqual(
            month_intervals(date(2025, 11, 15), date(2026, 1, 5)),
            [
                (date(2025, 11, 15), date(2025, 11, 30)),
                (date(2025, 12, 1), date(2025, 12, 31)),
                (date(2026, 1, 1), date(2026, 1, 5)),
            ],
        )

    def test_month_intervals_reject_reverse_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "after"):
            month_intervals(date(2026, 1, 1), date(2025, 1, 1))

    def test_download_writes_partition_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crsp"
            first = download_crsp_daily(
                FakeConnection(),
                start_date=date(2025, 12, 1),
                end_date=date(2025, 12, 31),
                output_dir=output,
            )
            second = download_crsp_daily(
                FakeConnection(),
                start_date=date(2025, 12, 1),
                end_date=date(2025, 12, 31),
                output_dir=output,
            )

            self.assertEqual(first, second)
            self.assertEqual(first[0].rows, 1)
            self.assertTrue(
                (output / "year=2025/month=12/crsp_daily.parquet").exists()
            )
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
