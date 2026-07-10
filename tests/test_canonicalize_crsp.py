from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alpha_mining_neural_network.canonicalize_crsp import canonicalize_crsp
from alpha_mining_neural_network.crsp_download import OUTPUT_COLUMNS


def row(permno: int, date: str) -> dict[str, object]:
    values: dict[str, object] = {column: pd.NA for column in OUTPUT_COLUMNS}
    values.update(
        {
            "permno": permno,
            "permco": permno + 100,
            "date": date,
            "prc": 10.0,
            "ret": 0.01,
            "retx": 0.01,
            "vol": 100.0,
            "mktcap": 1000.0,
        }
    )
    return values


class CanonicalizeCrspTest(unittest.TestCase):
    def test_sorts_and_publishes_unique_key_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.csv"
            output = root / "canonical" / "data.csv"
            manifest = root / "canonical" / "manifest.json"
            pd.DataFrame(
                [
                    row(2, "2020-01-02"),
                    row(1, "2020-01-02"),
                    row(1, "2020-01-01"),
                ],
                columns=OUTPUT_COLUMNS,
            ).to_csv(source, index=False)

            summary = canonicalize_crsp(
                input_path=source,
                output_path=output,
                manifest_path=manifest,
            )
            canonical = pd.read_csv(output)

            self.assertEqual(summary.rows, 3)
            self.assertEqual(summary.assets, 2)
            self.assertEqual(
                canonical[["permno", "date"]].to_records(index=False).tolist(),
                [(1, "2020-01-01"), (1, "2020-01-02"), (2, "2020-01-02")],
            )
            self.assertTrue(manifest.exists())

    def test_duplicate_key_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.csv"
            output = root / "canonical.csv"
            manifest = root / "manifest.json"
            pd.DataFrame(
                [row(1, "2020-01-01"), row(1, "2020-01-01")],
                columns=OUTPUT_COLUMNS,
            ).to_csv(source, index=False)

            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                canonicalize_crsp(
                    input_path=source,
                    output_path=output,
                    manifest_path=manifest,
                )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()

