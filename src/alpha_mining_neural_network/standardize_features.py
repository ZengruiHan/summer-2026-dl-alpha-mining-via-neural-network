"""Cross-sectionally standardize proposal-defined daily features."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "features" / "daily_features.csv"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "standard"
DEFAULT_DATASET_DIR = DEFAULT_OUTPUT_ROOT / "standardized_features"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "standardization_manifest.json"
DEFAULT_SECTOR_MAPPING = DEFAULT_OUTPUT_ROOT / "sector_mapping.json"

LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99
EPSILON = 1e-12

DIRECT_FEATURES = (
    "return_t_minus_1",
    "mean_return_t_minus_6_to_t_minus_2",
    "mean_return_t_minus_20_to_t_minus_7",
    "momentum_20",
    "beta_60",
)
LOG_EPSILON_FEATURES = ("volatility_20",)
LOG1P_FEATURES = ("mean_dollar_volume_20", "mean_turnover_20")
NUMERIC_FEATURES = DIRECT_FEATURES + LOG_EPSILON_FEATURES + LOG1P_FEATURES


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    dtypes: dict[str, str] = {
        "date": "string",
        "permno": "int32",
        "rank": "int16",
        "sector": "Int32",
    }
    dtypes.update({feature: "float64" for feature in NUMERIC_FEATURES})
    frame = pd.read_csv(path, dtype=dtypes)
    expected = ["date", "permno", "rank", *NUMERIC_FEATURES, "sector"]
    missing = set(expected).difference(frame.columns)
    if missing:
        raise RuntimeError(f"Feature input is missing columns: {sorted(missing)}")
    frame = frame.loc[:, expected]
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    frame = frame.sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["date", "permno"]).any():
        raise RuntimeError("Feature input contains duplicate (date, permno) keys")
    return frame


def apply_proposal_transforms(frame: pd.DataFrame) -> np.ndarray:
    """Apply the feature-specific transform before winsorization."""
    transformed = np.empty((len(frame), len(NUMERIC_FEATURES)), dtype=np.float64)
    for index, feature in enumerate(NUMERIC_FEATURES):
        values = frame[feature].to_numpy(dtype=np.float64, copy=False)
        if feature in DIRECT_FEATURES:
            transformed[:, index] = values
        elif feature in LOG_EPSILON_FEATURES:
            invalid = np.isfinite(values) & (values + EPSILON <= 0)
            if invalid.any():
                raise RuntimeError(f"{feature} has values outside log(x + epsilon) domain")
            transformed[:, index] = np.log(values + EPSILON)
        elif feature in LOG1P_FEATURES:
            invalid = np.isfinite(values) & (values <= -1)
            if invalid.any():
                raise RuntimeError(f"{feature} has values outside log1p domain")
            transformed[:, index] = np.log1p(values)
        else:  # pragma: no cover - guarded by the constant definitions
            raise AssertionError(f"No transform configured for {feature}")
    return transformed


def cross_sectional_winsor_zscore(
    values: np.ndarray,
    date_codes: np.ndarray,
    *,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> tuple[np.ndarray, dict[str, int]]:
    """Winsorize and z-score each feature independently within each date."""
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if len(values) != len(date_codes):
        raise ValueError("values and date_codes must have the same row count")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Invalid winsorization quantiles")

    unique_dates, starts, counts = np.unique(
        date_codes, return_index=True, return_counts=True
    )
    standardized = np.full_like(values, np.nan, dtype=np.float64)
    lower_clipped = 0
    upper_clipped = 0
    degenerate_groups = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _, start, count in zip(unique_dates, starts, counts, strict=True):
            stop = start + count
            cross_section = values[start:stop]
            lower = np.nanquantile(cross_section, lower_quantile, axis=0)
            upper = np.nanquantile(cross_section, upper_quantile, axis=0)
            lower_clipped += int(
                np.count_nonzero(np.isfinite(cross_section) & (cross_section < lower))
            )
            upper_clipped += int(
                np.count_nonzero(np.isfinite(cross_section) & (cross_section > upper))
            )
            winsorized = np.clip(cross_section, lower, upper)
            mean = np.nanmean(winsorized, axis=0)
            std = np.nanstd(winsorized, axis=0, ddof=0)
            valid = np.isfinite(winsorized)
            nondegenerate = np.isfinite(std) & (std > 0)
            standardized[start:stop] = np.divide(
                winsorized - mean,
                std,
                out=np.full_like(winsorized, np.nan),
                where=valid & nondegenerate,
            )
            degenerate = np.isfinite(std) & (std == 0)
            if degenerate.any():
                standardized[start:stop, degenerate] = np.where(
                    valid[:, degenerate], 0.0, np.nan
                )
                degenerate_groups += int(degenerate.sum())

    return standardized, {
        "lower_tail_values_clipped": lower_clipped,
        "upper_tail_values_clipped": upper_clipped,
        "degenerate_date_feature_groups_set_to_zero": degenerate_groups,
    }


def sector_encoding(
    sector: pd.Series,
) -> tuple[np.ndarray, list[int], list[str]]:
    if sector.isna().any():
        raise RuntimeError("Sector contains missing values; cannot form one-hot encoding")
    sector_codes = sorted(int(value) for value in sector.unique())
    raw = sector.astype("int32").to_numpy()
    indices = np.searchsorted(np.asarray(sector_codes, dtype=np.int32), raw).astype(
        np.int16
    )
    columns = [f"sector_sic_{code:04d}" for code in sector_codes]
    return indices, sector_codes, columns


def write_partitioned_dataset(
    frame: pd.DataFrame,
    standardized: np.ndarray,
    sector_indices: np.ndarray,
    sector_codes: list[int],
    sector_columns: list[str],
    output_dir: Path,
) -> list[dict[str, int | str]]:
    """Atomically write a schema-consistent yearly Parquet dataset."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace standardized dataset: {output_dir}")
    temporary = output_dir.parent / f".{output_dir.name}.part"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, int | str]] = []
    standardized_columns = [f"{feature}_z" for feature in NUMERIC_FEATURES]
    years = frame["date"].dt.year.to_numpy()

    try:
        for year in sorted(np.unique(years)):
            positions = np.flatnonzero(years == year)
            part = frame.iloc[positions][["date", "permno", "rank"]].reset_index(drop=True)
            for column_index, column in enumerate(standardized_columns):
                part[column] = standardized[positions, column_index]
            part["sector_index"] = sector_indices[positions]

            categorical = pd.Categorical(
                frame.iloc[positions]["sector"].astype("int32"),
                categories=sector_codes,
            )
            dummies = pd.get_dummies(
                categorical, prefix="sector_sic", prefix_sep="_", dtype=np.uint8
            )
            dummies.columns = [
                f"sector_sic_{int(column.rsplit('_', 1)[1]):04d}"
                for column in dummies.columns
            ]
            dummies = dummies.reindex(columns=sector_columns, fill_value=0)
            if not (dummies.sum(axis=1).to_numpy() == 1).all():
                raise RuntimeError(f"Sector one-hot integrity failed for year {year}")
            part = pd.concat([part, dummies.reset_index(drop=True)], axis=1)

            year_dir = temporary / f"year={int(year):04d}"
            year_dir.mkdir(parents=True, exist_ok=False)
            path = year_dir / "standardized_features.parquet"
            part.to_parquet(path, index=False, compression="zstd")
            records.append(
                {
                    "year": int(year),
                    "rows": len(part),
                    "bytes": path.stat().st_size,
                    "relative_path": str(path.relative_to(temporary)),
                }
            )
            print(
                f"Standardized output: {year}, {len(part):,} rows, "
                f"{path.stat().st_size / 1_048_576:.1f} MiB",
                flush=True,
            )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return records


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def standardize_features(
    *,
    input_path: Path,
    dataset_dir: Path,
    manifest_path: Path,
    sector_mapping_path: Path,
) -> dict[str, object]:
    conflicts = [
        path
        for path in (dataset_dir, manifest_path, sector_mapping_path)
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(f"Refusing to replace existing outputs: {conflicts}")

    print("Loading unstandardized features...", flush=True)
    frame = load_features(input_path)
    transformed = apply_proposal_transforms(frame)
    date_codes = pd.factorize(frame["date"], sort=False)[0]
    print("Applying daily 1%/99% winsorization and cross-sectional z-scores...", flush=True)
    standardized, winsor_stats = cross_sectional_winsor_zscore(
        transformed, date_codes
    )
    if np.isinf(standardized).any():
        raise RuntimeError("Standardized features contain infinite values")

    sector_indices, sector_codes, sector_columns = sector_encoding(frame["sector"])
    print(
        f"Encoding {len(sector_codes):,} point-in-time SICCD categories...",
        flush=True,
    )
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    partitions = write_partitioned_dataset(
        frame,
        standardized,
        sector_indices,
        sector_codes,
        sector_columns,
        dataset_dir,
    )

    standardized_columns = [f"{feature}_z" for feature in NUMERIC_FEATURES]
    mapping = {
        "encoding": "one-hot SICCD with stable zero-based sector_index",
        "learned_embedding_note": (
            "sector_index is the input to a learned embedding during model training; "
            "embedding weights are not fitted during static preprocessing"
        ),
        "categories": [
            {
                "sector_index": index,
                "siccd": code,
                "one_hot_column": column,
            }
            for index, (code, column) in enumerate(zip(sector_codes, sector_columns, strict=True))
        ],
    }
    write_json_atomic(mapping, sector_mapping_path)

    missing_by_column = {
        column: int(np.isnan(standardized[:, index]).sum())
        for index, column in enumerate(standardized_columns)
    }
    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.resolve()),
        "dataset_dir": dataset_dir.name,
        "format": "year-partitioned parquet",
        "primary_key": ["date", "permno"],
        "rows": len(frame),
        "dates": frame["date"].nunique(),
        "date_range": [
            frame["date"].min().date().isoformat(),
            frame["date"].max().date().isoformat(),
        ],
        "winsorization": {
            "scope": "cross-sectional within each date and feature",
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "quantile_method": "linear",
            **winsor_stats,
        },
        "zscore": {
            "scope": "cross-sectional within each date and feature",
            "standard_deviation_ddof": 0,
        },
        "criteria": {
            "direct_then_winsor_z": list(DIRECT_FEATURES),
            "log_x_plus_epsilon_then_winsor_z": list(LOG_EPSILON_FEATURES),
            "epsilon": EPSILON,
            "log1p_then_winsor_z": list(LOG1P_FEATURES),
            "turnover_assumption": (
                "Proposal does not list turnover separately; positive turnover uses "
                "the dollar-volume log1p criterion"
            ),
            "sector": "one-hot SICCD plus stable sector_index for learned embedding",
        },
        "standardized_columns": standardized_columns,
        "sector_categories": len(sector_codes),
        "one_hot_columns": len(sector_columns),
        "missing_by_standardized_column": missing_by_column,
        "complete_numeric_rows": int(np.isfinite(standardized).all(axis=1).sum()),
        "partitions": partitions,
        "total_bytes": sum(int(record["bytes"]) for record in partitions),
    }
    write_json_atomic(payload, manifest_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply proposal-defined cross-sectional feature standardization."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sector-mapping", type=Path, default=DEFAULT_SECTOR_MAPPING)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = standardize_features(
        input_path=args.input.expanduser().resolve(),
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        sector_mapping_path=args.sector_mapping.expanduser().resolve(),
    )
    print(f"Standardized rows: {payload['rows']:,}", flush=True)
    print(f"Dataset: {args.dataset_dir.expanduser().resolve()}", flush=True)
    print(f"Manifest: {args.manifest.expanduser().resolve()}", flush=True)
    return 0

