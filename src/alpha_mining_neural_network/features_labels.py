"""Build proposal-defined features and labels on the daily TOP500 universe."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "canonical"
    / "crsp_common_stocks_daily_20000101_20251231.csv"
)
DEFAULT_UNIVERSE_INPUT = (
    REPOSITORY_ROOT / "data" / "TOP500" / "top500_universe_21d.csv"
)
DEFAULT_MARKET_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "crsp_market_returns_daily_20000101_20251231.csv"
)
DEFAULT_FEATURE_DIR = REPOSITORY_ROOT / "data" / "features"
DEFAULT_LABEL_DIR = REPOSITORY_ROOT / "data" / "labels"
DEFAULT_FEATURE_OUTPUT = DEFAULT_FEATURE_DIR / "daily_features.csv"
DEFAULT_LABEL_OUTPUT = DEFAULT_LABEL_DIR / "daily_labels.csv"
DEFAULT_FEATURE_MANIFEST = DEFAULT_FEATURE_DIR / "features_manifest.json"
DEFAULT_LABEL_MANIFEST = DEFAULT_LABEL_DIR / "labels_manifest.json"

FEATURE_COLUMNS = (
    "return_t_minus_1",
    "mean_return_t_minus_6_to_t_minus_2",
    "mean_return_t_minus_20_to_t_minus_7",
    "momentum_20",
    "volatility_20",
    "mean_dollar_volume_20",
    "mean_turnover_20",
    "beta_60",
    "sector",
)


def load_market(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=["date", "vwretd"],
        dtype={"date": "string", "vwretd": "float64"},
    )
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RuntimeError("Market dates must be unique and sorted")
    returns = frame["vwretd"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(returns).all():
        raise RuntimeError("vwretd contains missing or non-finite values")
    return pd.DatetimeIndex(dates), returns


def load_canonical_panel(path: Path) -> pd.DataFrame:
    columns = ["permno", "date", "ret", "prc", "vol", "mktcap", "siccd"]
    panel = pd.read_csv(
        path,
        usecols=columns,
        dtype={
            "permno": "int32",
            "date": "string[pyarrow]",
            "ret": "float64",
            "prc": "float64",
            "vol": "float64",
            "mktcap": "float64",
            "siccd": "float64",
        },
    )
    if panel[["permno", "date"]].isna().any().any():
        raise RuntimeError("Canonical panel contains missing keys")
    if panel.duplicated(["permno", "date"]).any():
        raise RuntimeError("Canonical panel contains duplicate keys")
    return panel


def load_universe(path: Path) -> pd.DataFrame:
    universe = pd.read_csv(
        path,
        usecols=["date", "permno", "rank"],
        dtype={"date": "string", "permno": "int32", "rank": "int16"},
    )
    universe["date"] = pd.to_datetime(
        universe["date"], format="%Y-%m-%d", errors="raise"
    )
    universe = universe.sort_values(["date", "rank"], kind="mergesort").reset_index(
        drop=True
    )
    if universe.duplicated(["date", "permno"]).any():
        raise RuntimeError("TOP500 universe contains duplicate keys")
    return universe


def encode_panel_and_universe(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assets = np.sort(panel["permno"].unique().astype(np.int32, copy=False))
    calendar_strings = calendar.strftime("%Y-%m-%d")
    panel_asset = np.searchsorted(assets, panel["permno"].to_numpy())
    panel_date = pd.Categorical(
        panel["date"], categories=calendar_strings, ordered=True
    ).codes.astype(np.int16, copy=False)
    member_asset = np.searchsorted(assets, universe["permno"].to_numpy())
    member_date = calendar.get_indexer(universe["date"]).astype(np.int16, copy=False)
    if (panel_date < 0).any() or (member_date < 0).any():
        raise RuntimeError("Panel or universe contains dates outside the market calendar")
    if (member_asset >= len(assets)).any() or not np.array_equal(
        assets[member_asset], universe["permno"].to_numpy()
    ):
        raise RuntimeError("Universe contains PERMNO absent from canonical data")
    return assets, panel_asset, panel_date, member_asset, member_date


def sparse_matrix(
    values: np.ndarray,
    panel_asset: np.ndarray,
    panel_date: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    matrix = np.full(shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    matrix[panel_asset[finite], panel_date[finite]] = values[finite]
    return matrix


def prefix_sum(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    filled = np.nan_to_num(values, copy=True, nan=0.0)
    np.cumsum(filled, axis=1, out=prefix[:, 1:])
    return prefix


def prefix_count(mask: np.ndarray) -> np.ndarray:
    prefix = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=np.uint16)
    np.cumsum(mask, axis=1, dtype=np.uint16, out=prefix[:, 1:])
    return prefix


def selected_window(
    prefix: np.ndarray,
    member_asset: np.ndarray,
    member_date: np.ndarray,
    *,
    far_lag: int,
    near_lag: int,
) -> np.ndarray:
    """Select inclusive offsets ``t-far_lag`` through ``t-near_lag``."""
    if far_lag < near_lag or near_lag < 1:
        raise ValueError("Require far_lag >= near_lag >= 1")
    start = member_date.astype(np.int64) - far_lag
    end = member_date.astype(np.int64) - near_lag + 1
    result = np.full(len(member_date), np.nan, dtype=np.float64)
    valid = start >= 0
    result[valid] = (
        prefix[member_asset[valid], end[valid]]
        - prefix[member_asset[valid], start[valid]]
    )
    return result


def require_full_window(
    values: np.ndarray, counts: np.ndarray, expected: int
) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[counts != expected] = np.nan
    return result


def compute_return_features(
    return_matrix: np.ndarray,
    market_returns: np.ndarray,
    member_asset: np.ndarray,
    member_date: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(member_date)
    lag_one = np.full(row_count, np.nan, dtype=np.float64)
    lag_valid = member_date >= 1
    lag_one[lag_valid] = return_matrix[
        member_asset[lag_valid], member_date[lag_valid] - 1
    ]

    finite = np.isfinite(return_matrix)
    count_prefix = prefix_count(finite)
    return_prefix = prefix_sum(return_matrix)

    count_5 = selected_window(
        count_prefix, member_asset, member_date, far_lag=6, near_lag=2
    )
    sum_5 = selected_window(
        return_prefix, member_asset, member_date, far_lag=6, near_lag=2
    )
    mean_5 = require_full_window(sum_5 / 5.0, count_5, 5)

    count_14 = selected_window(
        count_prefix, member_asset, member_date, far_lag=20, near_lag=7
    )
    sum_14 = selected_window(
        return_prefix, member_asset, member_date, far_lag=20, near_lag=7
    )
    mean_14 = require_full_window(sum_14 / 14.0, count_14, 14)

    count_20 = selected_window(
        count_prefix, member_asset, member_date, far_lag=20, near_lag=1
    )
    sum_20 = selected_window(
        return_prefix, member_asset, member_date, far_lag=20, near_lag=1
    )
    count_60 = selected_window(
        count_prefix, member_asset, member_date, far_lag=60, near_lag=1
    )
    sum_60 = selected_window(
        return_prefix, member_asset, member_date, far_lag=60, near_lag=1
    )
    del return_prefix

    squared_prefix = prefix_sum(np.square(return_matrix))
    sum_squared_20 = selected_window(
        squared_prefix, member_asset, member_date, far_lag=20, near_lag=1
    )
    variance_20 = (sum_squared_20 - np.square(sum_20) / 20.0) / 19.0
    variance_20 = np.maximum(variance_20, 0.0)
    volatility = require_full_window(np.sqrt(variance_20), count_20, 20)
    del squared_prefix

    cross_prefix = prefix_sum(return_matrix * market_returns[np.newaxis, :])
    sum_cross_60 = selected_window(
        cross_prefix, member_asset, member_date, far_lag=60, near_lag=1
    )
    del cross_prefix

    market_prefix = np.concatenate(([0.0], np.cumsum(market_returns)))
    market_squared_prefix = np.concatenate(
        ([0.0], np.cumsum(np.square(market_returns)))
    )
    starts_60 = member_date.astype(np.int64) - 60
    valid_60_date = starts_60 >= 0
    sum_market_60 = np.full(row_count, np.nan)
    sum_market_squared_60 = np.full(row_count, np.nan)
    ends = member_date.astype(np.int64)
    sum_market_60[valid_60_date] = (
        market_prefix[ends[valid_60_date]] - market_prefix[starts_60[valid_60_date]]
    )
    sum_market_squared_60[valid_60_date] = (
        market_squared_prefix[ends[valid_60_date]]
        - market_squared_prefix[starts_60[valid_60_date]]
    )
    covariance_numerator = sum_cross_60 - sum_60 * sum_market_60 / 60.0
    market_variance_numerator = (
        sum_market_squared_60 - np.square(sum_market_60) / 60.0
    )
    beta = np.full(row_count, np.nan)
    beta_valid = (
        (count_60 == 60)
        & np.isfinite(covariance_numerator)
        & (market_variance_numerator > 0)
    )
    beta[beta_valid] = (
        covariance_numerator[beta_valid] / market_variance_numerator[beta_valid]
    )

    momentum_valid = finite & (return_matrix >= -1.0)
    momentum_count_prefix = prefix_count(momentum_valid)
    log_values = np.zeros_like(return_matrix)
    above_minus_one = momentum_valid & (return_matrix > -1.0)
    log_values[above_minus_one] = np.log1p(return_matrix[above_minus_one])
    log_prefix = prefix_sum(log_values)
    minus_one_prefix = prefix_count(return_matrix == -1.0)
    momentum_count = selected_window(
        momentum_count_prefix,
        member_asset,
        member_date,
        far_lag=20,
        near_lag=1,
    )
    log_sum = selected_window(
        log_prefix, member_asset, member_date, far_lag=20, near_lag=1
    )
    minus_one_count = selected_window(
        minus_one_prefix,
        member_asset,
        member_date,
        far_lag=20,
        near_lag=1,
    )
    momentum = np.expm1(log_sum)
    momentum[minus_one_count > 0] = -1.0
    momentum[momentum_count != 20] = np.nan

    return {
        "return_t_minus_1": lag_one,
        "mean_return_t_minus_6_to_t_minus_2": mean_5,
        "mean_return_t_minus_20_to_t_minus_7": mean_14,
        "momentum_20": momentum,
        "volatility_20": volatility,
        "beta_60": beta,
    }


def selected_prior_mean(
    matrix: np.ndarray,
    member_asset: np.ndarray,
    member_date: np.ndarray,
    window: int,
) -> np.ndarray:
    finite = np.isfinite(matrix)
    counts = selected_window(
        prefix_count(finite),
        member_asset,
        member_date,
        far_lag=window,
        near_lag=1,
    )
    sums = selected_window(
        prefix_sum(matrix),
        member_asset,
        member_date,
        far_lag=window,
        near_lag=1,
    )
    return require_full_window(sums / float(window), counts, window)


def point_in_time_sector(
    panel_asset: np.ndarray,
    panel_date: np.ndarray,
    sector_values: np.ndarray,
    member_asset: np.ndarray,
    member_date: np.ndarray,
    asset_count: int,
) -> np.ndarray:
    """Use the latest nonmissing SICCD available on or before date t."""
    result = np.full(len(member_date), np.nan, dtype=np.float64)
    member_order = np.argsort(member_asset, kind="stable")
    ordered_member_assets = member_asset[member_order]
    member_starts = np.searchsorted(ordered_member_assets, np.arange(asset_count), side="left")
    member_ends = np.searchsorted(ordered_member_assets, np.arange(asset_count), side="right")
    panel_starts = np.searchsorted(panel_asset, np.arange(asset_count), side="left")
    panel_ends = np.searchsorted(panel_asset, np.arange(asset_count), side="right")

    for asset_index in range(asset_count):
        if member_starts[asset_index] == member_ends[asset_index]:
            continue
        raw_slice = slice(panel_starts[asset_index], panel_ends[asset_index])
        raw_sectors = sector_values[raw_slice]
        valid_sector = np.isfinite(raw_sectors)
        if not valid_sector.any():
            continue
        raw_dates = panel_date[raw_slice][valid_sector]
        raw_values = raw_sectors[valid_sector]
        selected_positions = member_order[
            member_starts[asset_index] : member_ends[asset_index]
        ]
        locations = np.searchsorted(
            raw_dates, member_date[selected_positions], side="right"
        ) - 1
        available = locations >= 0
        result[selected_positions[available]] = raw_values[locations[available]]
    return result


def compute_labels(
    return_matrix: np.ndarray,
    member_asset: np.ndarray,
    member_date: np.ndarray,
    calendar: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = len(member_date)
    target_return = np.full(row_count, np.nan, dtype=np.float64)
    label = np.full(row_count, np.nan, dtype=np.float64)
    target_date = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    has_next_date = member_date < len(calendar) - 1
    next_dates = member_date[has_next_date] + 1
    target_return[has_next_date] = return_matrix[
        member_asset[has_next_date], next_dates
    ]
    target_date[has_next_date] = calendar.to_numpy()[next_dates]

    unique_dates, starts, counts = np.unique(
        member_date, return_index=True, return_counts=True
    )
    for _, start, count in zip(unique_dates, starts, counts, strict=True):
        positions = np.arange(start, start + count)
        available = positions[np.isfinite(target_return[positions])]
        if len(available) == 0:
            continue
        quantiles = np.quantile(
            target_return[available], [0.2, 0.4, 0.6, 0.8], method="linear"
        )
        label[available] = np.searchsorted(
            quantiles, target_return[available], side="left"
        ) - 2
    return target_date, target_return, label


def atomic_to_csv(frame: pd.DataFrame, path: Path, chunk_size: int = 250_000) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        for start in range(0, len(frame), chunk_size):
            chunk = frame.iloc[start : start + chunk_size]
            chunk.to_csv(
                temporary,
                mode="w" if start == 0 else "a",
                header=start == 0,
                index=False,
                date_format="%Y-%m-%d",
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_features_and_labels(
    *,
    canonical_input: Path,
    universe_input: Path,
    market_input: Path,
    feature_output: Path,
    label_output: Path,
    feature_manifest: Path,
    label_manifest: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflicts = [
        path
        for path in (feature_output, label_output, feature_manifest, label_manifest)
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(f"Refusing to replace existing outputs: {conflicts}")

    print("Loading canonical data, TOP500 membership, and market returns...", flush=True)
    calendar, market_returns = load_market(market_input)
    panel = load_canonical_panel(canonical_input)
    universe = load_universe(universe_input)
    assets, panel_asset, panel_date, member_asset, member_date = encode_panel_and_universe(
        panel, universe, calendar
    )
    shape = (len(assets), len(calendar))

    print("Computing return, momentum, volatility, beta, and labels...", flush=True)
    return_matrix = sparse_matrix(
        panel["ret"].to_numpy(), panel_asset, panel_date, shape
    )
    target_date, target_return, label = compute_labels(
        return_matrix, member_asset, member_date, calendar
    )
    feature_values = compute_return_features(
        return_matrix, market_returns, member_asset, member_date
    )
    del return_matrix
    gc.collect()

    print("Computing trailing dollar volume and turnover...", flush=True)
    price = panel["prc"].to_numpy(dtype=np.float64, copy=False)
    volume = panel["vol"].to_numpy(dtype=np.float64, copy=False)
    market_cap = panel["mktcap"].to_numpy(dtype=np.float64, copy=False)
    dollar_volume = np.abs(price) * volume
    dollar_volume_matrix = sparse_matrix(dollar_volume, panel_asset, panel_date, shape)
    feature_values["mean_dollar_volume_20"] = selected_prior_mean(
        dollar_volume_matrix, member_asset, member_date, 20
    )
    del dollar_volume_matrix
    gc.collect()

    turnover = np.full(len(panel), np.nan, dtype=np.float64)
    turnover_valid = (
        np.isfinite(dollar_volume) & np.isfinite(market_cap) & (market_cap > 0)
    )
    # CRSP CIZ dlycap is reported in thousands of dollars.
    turnover[turnover_valid] = (
        dollar_volume[turnover_valid] / (market_cap[turnover_valid] * 1000.0)
    )
    turnover_matrix = sparse_matrix(turnover, panel_asset, panel_date, shape)
    feature_values["mean_turnover_20"] = selected_prior_mean(
        turnover_matrix, member_asset, member_date, 20
    )
    del turnover_matrix
    gc.collect()

    print("Resolving point-in-time SIC sector codes...", flush=True)
    feature_values["sector"] = point_in_time_sector(
        panel_asset,
        panel_date,
        panel["siccd"].to_numpy(dtype=np.float64, copy=False),
        member_asset,
        member_date,
        len(assets),
    )

    features = universe[["date", "permno", "rank"]].copy()
    for column in FEATURE_COLUMNS:
        features[column] = feature_values[column]
    features["sector"] = pd.array(features["sector"], dtype="Int32")

    labels = universe[["date", "permno", "rank"]].copy()
    labels["target_date"] = pd.to_datetime(target_date)
    labels["target_return"] = target_return
    labels["label"] = pd.array(label, dtype="Int8")

    if features.duplicated(["date", "permno"]).any() or labels.duplicated(
        ["date", "permno"]
    ).any():
        raise RuntimeError("Feature or label output contains duplicate keys")
    numeric_features = [column for column in FEATURE_COLUMNS if column != "sector"]
    if np.isinf(features[numeric_features].to_numpy()).any():
        raise RuntimeError("Feature output contains infinite values")
    if not features[["date", "permno", "rank"]].equals(
        labels[["date", "permno", "rank"]]
    ):
        raise RuntimeError("Feature and label keys are not aligned")

    print("Writing feature and label CSV files...", flush=True)
    atomic_to_csv(features, feature_output)
    atomic_to_csv(labels, label_output)

    created_at = datetime.now(timezone.utc).isoformat()
    common = {
        "created_at_utc": created_at,
        "canonical_input": str(canonical_input.resolve()),
        "universe_input": str(universe_input.resolve()),
        "market_input": str(market_input.resolve()),
        "primary_key": ["date", "permno"],
        "rows": len(universe),
        "dates": universe["date"].nunique(),
        "min_date": universe["date"].min().date().isoformat(),
        "max_date": universe["date"].max().date().isoformat(),
    }
    feature_payload = {
        **common,
        "output": feature_output.name,
        "bytes": feature_output.stat().st_size,
        "feature_columns": list(FEATURE_COLUMNS),
        "market_return": "CRSP vwretd",
        "turnover_definition": "abs(prc) * vol / (mktcap * 1000)",
        "sector_definition": "latest nonmissing CRSP SICCD available on or before t",
        "anti_lookahead": "all numeric features use dates strictly before t",
        "missing_by_feature": {
            column: int(features[column].isna().sum()) for column in FEATURE_COLUMNS
        },
        "complete_feature_rows": int(
            features[list(FEATURE_COLUMNS)].notna().all(axis=1).sum()
        ),
    }
    label_payload = {
        **common,
        "output": label_output.name,
        "bytes": label_output.stat().st_size,
        "target": "next global CRSP trading-date return for each member of A_t",
        "quantiles": [0.2, 0.4, 0.6, 0.8],
        "classes": [-2, -1, 0, 1, 2],
        "missing_target_returns": int(labels["target_return"].isna().sum()),
        "missing_labels": int(labels["label"].isna().sum()),
        "class_counts": {
            str(int(key)): int(value)
            for key, value in labels["label"].value_counts().sort_index().items()
        },
    }
    write_json_atomic(feature_payload, feature_manifest)
    write_json_atomic(label_payload, label_manifest)
    return features, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build proposal-defined x_i,t features and y_i,t labels."
    )
    parser.add_argument("--canonical-input", type=Path, default=DEFAULT_CANONICAL_INPUT)
    parser.add_argument("--universe-input", type=Path, default=DEFAULT_UNIVERSE_INPUT)
    parser.add_argument("--market-input", type=Path, default=DEFAULT_MARKET_INPUT)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--label-output", type=Path, default=DEFAULT_LABEL_OUTPUT)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--label-manifest", type=Path, default=DEFAULT_LABEL_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features, labels = build_features_and_labels(
        canonical_input=args.canonical_input.expanduser().resolve(),
        universe_input=args.universe_input.expanduser().resolve(),
        market_input=args.market_input.expanduser().resolve(),
        feature_output=args.feature_output.expanduser().resolve(),
        label_output=args.label_output.expanduser().resolve(),
        feature_manifest=args.feature_manifest.expanduser().resolve(),
        label_manifest=args.label_manifest.expanduser().resolve(),
    )
    print(f"Feature rows: {len(features):,}", flush=True)
    print(f"Label rows: {len(labels):,}", flush=True)
    print(f"Features: {args.feature_output.expanduser().resolve()}", flush=True)
    print(f"Labels: {args.label_output.expanduser().resolve()}", flush=True)
    return 0

