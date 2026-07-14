"""Construct the daily top-500 universe by lagged 21-day dollar volume."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_mining_neural_network.crsp_download import OUTPUT_COLUMNS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "canonical"
    / "crsp_common_stocks_daily_20000101_20251231.csv"
)
DEFAULT_MARKET_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "crsp_market_returns_daily_20000101_20251231.csv"
)
DEFAULT_RAW_PARQUET_DIR = REPOSITORY_ROOT / "data" / "raw" / "crsp_daily"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "TOP500"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "top500_universe_21d.csv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "top500_universe_21d_manifest.json"

DEFAULT_WINDOW = 21
DEFAULT_TOP_N = 500
TRAILING_COLUMN = "trailing_21d_dollar_volume"


def load_market_calendar(path: Path) -> pd.DatetimeIndex:
    """Load the strict global CRSP trading calendar."""
    if not path.exists():
        raise FileNotFoundError(f"Market-return calendar not found: {path}")
    dates = pd.read_csv(path, usecols=["date"], dtype={"date": "string"})["date"]
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if parsed.duplicated().any():
        raise RuntimeError("Market calendar contains duplicate dates")
    if not parsed.is_monotonic_increasing:
        raise RuntimeError("Market calendar is not sorted")
    return pd.DatetimeIndex(parsed)


def load_ranking_panel(path: Path) -> pd.DataFrame:
    """Load only the four canonical columns needed for ranking."""
    if not path.exists():
        raise FileNotFoundError(f"Canonical CRSP CSV not found: {path}")
    panel = pd.read_csv(
        path,
        usecols=["permno", "date", "prc", "vol"],
        dtype={
            "permno": "int32",
            "date": "string[pyarrow]",
            "prc": "float64",
            "vol": "float64",
        },
    )
    if panel[["permno", "date"]].isna().any().any():
        raise RuntimeError("Canonical ranking panel contains missing keys")
    if (panel["vol"].dropna() < 0).any():
        raise RuntimeError("Canonical ranking panel contains negative volume")
    if panel.duplicated(["permno", "date"]).any():
        raise RuntimeError("Canonical ranking panel contains duplicate keys")
    return panel


def build_dollar_volume_matrix(
    panel: pd.DataFrame, calendar: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """Map the sparse stock panel to asset-by-market-date dollar volume."""
    assets = np.sort(panel["permno"].unique().astype(np.int32, copy=False))
    calendar_strings = calendar.strftime("%Y-%m-%d")
    date_codes = pd.Categorical(
        panel["date"], categories=calendar_strings, ordered=True
    ).codes
    if (date_codes < 0).any():
        bad = panel.loc[date_codes < 0, "date"].head().tolist()
        raise RuntimeError(f"Stock dates absent from market calendar: {bad}")
    asset_codes = np.searchsorted(assets, panel["permno"].to_numpy())

    matrix = np.full((len(assets), len(calendar)), np.nan, dtype=np.float64)
    dollar_volume = (
        panel["prc"].abs().to_numpy(dtype=np.float64, copy=False)
        * panel["vol"].to_numpy(dtype=np.float64, copy=False)
    )
    finite = np.isfinite(dollar_volume)
    matrix[asset_codes[finite], date_codes[finite]] = dollar_volume[finite]
    return assets, matrix


def trailing_mean_strict_calendar(
    dollar_volume: np.ndarray, window: int
) -> np.ndarray:
    """Compute the prior-window mean, requiring every market date to be present."""
    if window <= 0:
        raise ValueError("window must be positive")
    if dollar_volume.ndim != 2:
        raise ValueError("dollar_volume must be a two-dimensional matrix")
    asset_count, date_count = dollar_volume.shape
    trailing = np.full((asset_count, date_count), np.nan, dtype=np.float64)
    if date_count <= window:
        return trailing

    valid = np.isfinite(dollar_volume)
    cumulative_sum = np.nan_to_num(dollar_volume, copy=True, nan=0.0)
    np.cumsum(cumulative_sum, axis=1, out=cumulative_sum)
    cumulative_count = np.cumsum(valid, axis=1, dtype=np.uint16)

    # At date index t, use exactly [t-window, ..., t-1]. Current-date
    # dollar volume is therefore never part of its own ranking signal.
    trailing[:, window:] = cumulative_sum[:, window - 1 : -1]
    if date_count > window + 1:
        trailing[:, window + 1 :] -= cumulative_sum[:, : -window - 1]

    rolling_count = np.zeros_like(cumulative_count, dtype=np.uint16)
    rolling_count[:, window:] = cumulative_count[:, window - 1 : -1]
    if date_count > window + 1:
        rolling_count[:, window + 1 :] -= cumulative_count[:, : -window - 1]
    trailing[rolling_count != window] = np.nan
    trailing /= float(window)
    return trailing


def select_daily_top_n(
    trailing: np.ndarray,
    assets: np.ndarray,
    calendar: pd.DatetimeIndex,
    top_n: int,
) -> pd.DataFrame:
    """Rank each date with deterministic PERMNO tie-breaking."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if trailing.shape != (len(assets), len(calendar)):
        raise ValueError("Trailing matrix dimensions do not match assets/calendar")

    date_parts: list[np.ndarray] = []
    permno_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    for date_index in range(len(calendar)):
        values = trailing[:, date_index]
        eligible = np.flatnonzero(np.isfinite(values))
        keep_count = min(top_n, len(eligible))
        if keep_count == 0:
            continue

        if len(eligible) > keep_count:
            eligible_values = values[eligible]
            partial = np.argpartition(eligible_values, -keep_count)[-keep_count:]
            cutoff = eligible_values[partial].min()
            greater = eligible[eligible_values > cutoff]
            equal = eligible[eligible_values == cutoff]
            needed = keep_count - len(greater)
            equal = equal[np.argsort(assets[equal], kind="stable")[:needed]]
            selected = np.concatenate((greater, equal))
        else:
            selected = eligible

        order = np.lexsort((assets[selected], -values[selected]))
        selected = selected[order]
        date_parts.append(np.full(len(selected), date_index, dtype=np.int16))
        permno_parts.append(assets[selected].astype(np.int32, copy=False))
        rank_parts.append(np.arange(1, len(selected) + 1, dtype=np.int16))
        value_parts.append(values[selected])

    if not date_parts:
        return pd.DataFrame(columns=["date", "permno", "rank", TRAILING_COLUMN])
    date_indices = np.concatenate(date_parts)
    return pd.DataFrame(
        {
            "date": calendar.take(date_indices),
            "permno": np.concatenate(permno_parts),
            "rank": np.concatenate(rank_parts),
            TRAILING_COLUMN: np.concatenate(value_parts),
        }
    )


def compute_membership(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    window: int = DEFAULT_WINDOW,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Compute daily membership from a canonical stock panel."""
    assets, dollar_volume = build_dollar_volume_matrix(panel, calendar)
    trailing = trailing_mean_strict_calendar(dollar_volume, window)
    return select_daily_top_n(trailing, assets, calendar, top_n)


def enrich_and_write_csv(
    membership: pd.DataFrame,
    *,
    raw_parquet_dir: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    """Attach current-date CRSP fields and atomically write the universe CSV."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace TOP500 output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    rows_written = 0
    missing_current_rows = 0
    missing_current_dollar_volume_rows = 0
    write_header = True
    extra_columns = [column for column in OUTPUT_COLUMNS if column not in {"date", "permno"}]
    output_columns = [
        "date",
        "permno",
        "rank",
        "dollar_volume",
        TRAILING_COLUMN,
        *extra_columns,
    ]

    membership = membership.copy()
    membership["date"] = pd.to_datetime(membership["date"], errors="raise")
    membership["year"] = membership["date"].dt.year
    membership["month"] = membership["date"].dt.month
    groups = list(membership.groupby(["year", "month"], sort=True))

    try:
        for index, ((year, month), members) in enumerate(groups, start=1):
            source = (
                raw_parquet_dir
                / f"year={int(year):04d}"
                / f"month={int(month):02d}"
                / "crsp_daily.parquet"
            )
            if not source.exists():
                raise FileNotFoundError(f"Raw CRSP partition not found: {source}")
            current = pd.read_parquet(source, columns=list(OUTPUT_COLUMNS))
            current["date"] = pd.to_datetime(current["date"], errors="raise")
            current["dollar_volume"] = current["prc"].abs() * current["vol"]
            members = members.drop(columns=["year", "month"])
            output = members.merge(
                current,
                on=["date", "permno"],
                how="left",
                validate="one_to_one",
                indicator=True,
            )
            missing_current_rows += int((output["_merge"] == "left_only").sum())
            output = output.drop(columns="_merge")
            missing_current_dollar_volume_rows += int(
                output["dollar_volume"].isna().sum()
            )
            output = output.sort_values(["date", "rank"], kind="mergesort")
            output = output.loc[:, output_columns]
            output.to_csv(
                temporary,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
                date_format="%Y-%m-%d",
            )
            write_header = False
            rows_written += len(output)
            if index == 1 or index % 12 == 0 or index == len(groups):
                print(
                    f"TOP500 output: {index:03d}/{len(groups):03d} months, "
                    f"{rows_written:,} rows",
                    flush=True,
                )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return rows_written, missing_current_rows, missing_current_dollar_volume_rows


def validate_membership(
    membership: pd.DataFrame, *, top_n: int
) -> dict[str, int | str]:
    if membership.empty:
        raise RuntimeError("TOP500 membership is empty")
    if membership[["date", "permno", "rank", TRAILING_COLUMN]].isna().any().any():
        raise RuntimeError("TOP500 membership contains missing key/rank/signal values")
    duplicate_keys = int(membership.duplicated(["date", "permno"]).sum())
    if duplicate_keys:
        raise RuntimeError(f"TOP500 membership has {duplicate_keys} duplicate keys")
    counts = membership.groupby("date", sort=True).size()
    if (counts > top_n).any():
        raise RuntimeError("TOP500 membership exceeds top_n on at least one date")
    expected_ranks = membership.groupby("date", sort=False).cumcount() + 1
    if not np.array_equal(membership["rank"].to_numpy(), expected_ranks.to_numpy()):
        raise RuntimeError("TOP500 ranks are not contiguous within date")
    return {
        "rows": len(membership),
        "dates": membership["date"].nunique(),
        "min_date": pd.Timestamp(membership["date"].min()).date().isoformat(),
        "max_date": pd.Timestamp(membership["date"].max()).date().isoformat(),
        "duplicate_keys": duplicate_keys,
        "min_assets_per_date": int(counts.min()),
        "max_assets_per_date": int(counts.max()),
    }


def write_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    canonical_input: Path,
    market_input: Path,
    window: int,
    top_n: int,
    validation: dict[str, int | str],
    missing_current_rows: int,
    missing_current_dollar_volume_rows: int,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_input": str(canonical_input.resolve()),
        "market_calendar_input": str(market_input.resolve()),
        "output": output_path.name,
        "primary_key": ["date", "permno"],
        "sort_order": ["date", "rank"],
        "window": window,
        "top_n": top_n,
        "dollar_volume_formula": "abs(prc) * vol",
        "ranking_signal": f"mean of prior {window} global market dates, excluding t",
        "tie_breaker": "permno ascending",
        "missing_current_rows": missing_current_rows,
        "missing_current_dollar_volume_rows": missing_current_dollar_volume_rows,
        "validation": validation,
        "bytes": output_path.stat().st_size,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def build_top500(
    *,
    canonical_input: Path,
    market_input: Path,
    raw_parquet_dir: Path,
    output_path: Path,
    manifest_path: Path,
    window: int = DEFAULT_WINDOW,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, int | str]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to replace existing TOP500 output or manifest")
    print("Loading canonical ranking columns...", flush=True)
    panel = load_ranking_panel(canonical_input)
    calendar = load_market_calendar(market_input)
    print(
        f"Computing lagged {window}-day dollar-volume ranks for "
        f"{panel['permno'].nunique():,} assets...",
        flush=True,
    )
    membership = compute_membership(panel, calendar, window=window, top_n=top_n)
    membership = membership.sort_values(["date", "rank"], kind="mergesort").reset_index(
        drop=True
    )
    validation = validate_membership(membership, top_n=top_n)
    del panel
    print("Attaching current-date CRSP fields...", flush=True)
    rows_written, missing_current_rows, missing_current_dollar_volume_rows = enrich_and_write_csv(
        membership, raw_parquet_dir=raw_parquet_dir, output_path=output_path
    )
    if rows_written != validation["rows"]:
        raise RuntimeError("Written TOP500 row count does not match membership")
    write_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        canonical_input=canonical_input,
        market_input=market_input,
        window=window,
        top_n=top_n,
        validation=validation,
        missing_current_rows=missing_current_rows,
        missing_current_dollar_volume_rows=missing_current_dollar_volume_rows,
    )
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank the daily top 500 stocks by lagged 21-day dollar volume."
    )
    parser.add_argument("--canonical-input", type=Path, default=DEFAULT_CANONICAL_INPUT)
    parser.add_argument("--market-input", type=Path, default=DEFAULT_MARKET_INPUT)
    parser.add_argument("--raw-parquet-dir", type=Path, default=DEFAULT_RAW_PARQUET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation = build_top500(
        canonical_input=args.canonical_input.expanduser().resolve(),
        market_input=args.market_input.expanduser().resolve(),
        raw_parquet_dir=args.raw_parquet_dir.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        manifest_path=args.manifest.expanduser().resolve(),
        window=args.window,
        top_n=args.top_n,
    )
    print(f"TOP500 rows: {validation['rows']:,}", flush=True)
    print(f"Dates: {validation['dates']:,}", flush=True)
    print(f"Date range: {validation['min_date']} to {validation['max_date']}", flush=True)
    print(f"Output: {args.output.expanduser().resolve()}", flush=True)
    print(f"Manifest: {args.manifest.expanduser().resolve()}", flush=True)
    return 0
