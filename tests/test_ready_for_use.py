from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_mining_neural_network.ready_for_use import (
    build_feature_arrays,
    date_pointer,
    map_local_ranks,
)


class ReadyForUseTest(unittest.TestCase):
    def test_feature_arrays_have_daily_node_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "features.parquet"
            output = root / "arrays"
            rows = []
            for date in pd.to_datetime(["2020-01-02", "2020-01-03"]):
                for rank in range(1, 501):
                    rows.append(
                        {
                            "date": date,
                            "permno": 10_000 + rank,
                            "rank": rank,
                            "f_z": float(rank),
                            "sector_index": rank % 10,
                        }
                    )
            pd.DataFrame(rows).to_parquet(source, index=False)
            source_frame = pd.read_parquet(source)
            source_frame.loc[0, "f_z"] = np.nan
            source_frame.to_parquet(source, index=False)

            record = build_feature_arrays(source, ["f_z"], output)

            self.assertEqual(record["dates"], 2)
            self.assertEqual(np.load(output / "x_numeric.npy").shape, (2, 500, 1))
            self.assertEqual(np.load(output / "x_numeric.npy")[0, 0, 0], 0.0)
            self.assertFalse(np.load(output / "complete_numeric_mask.npy")[0, 0])
            self.assertEqual(np.load(output / "permno.npy")[0, 0], 10001)

    def test_graph_permnos_map_to_local_ranks_and_pointer(self) -> None:
        dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        permno = np.vstack((np.arange(1000, 1500), np.arange(2000, 2500))).astype(
            np.int32
        )
        edge_dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03"]))
        date_index, source, target = map_local_ranks(
            dates,
            permno,
            edge_dates,
            np.array([1003, 2004], dtype=np.int32),
            np.array([1007, 2009], dtype=np.int32),
        )

        self.assertEqual(source.tolist(), [3, 4])
        self.assertEqual(target.tolist(), [7, 9])
        self.assertEqual(date_pointer(date_index, 2).tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
