"""Create flat CRSP CSV deliverables in ``data/raw``.

Stock observations are streamed from the verified monthly Parquet download.
Market returns are downloaded from CRSP's CIZ daily index-return query.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_mining_neural_network.crsp_download import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    OUTPUT_COLUMNS,
    SqlConnection,
    connect_wrds,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_PARQUET_DIR = DEFAULT_RAW_DIR / "crsp_daily"
DEFAULT_SNAPSHOT_DATE = date(2025, 6, 30)

MARKET_COLUMNS = (
    "date",
    "vwretd",
    "vwretx",
    "vwtotval",
    "vwusdval",
    "vwtotcnt",
    "vwusdcnt",
    "ewretd",
    "ewretx",
    "ewtotval",
    "ewusdval",
    "ewtotcnt",
    "ewusdcnt",
    "sprtrn",
    "spindx",
)

MARKET_RETURN_SQL = """
SELECT
    dlycaldt AS date,
    vwretd,
    vwretx,
    vwtotval,
    vwusdval,
    vwtotcnt,
    vwusdcnt,
    ewretd,
    ewretx,
    ewtotval,
    ewusdval,
    ewtotcnt,
    ewusdcnt,
    sprtrn,
    spindx
FROM crsp.wrds_dailyindexret_query
WHERE dlycaldt BETWEEN %(start_date)s AND %(end_date)s
ORDER BY dlycaldt
"""


def compact_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def stock_csv_path(raw_dir: Path, start_date: date, end_date: date) -> Path:
    return raw_dir / (
        f"crsp_common_stocks_daily_{compact_date(start_date)}_"
        f"{compact_date(end_date)}.csv"
    )


def snapshot_csv_path(raw_dir: Path, snapshot_date: date) -> Path:
    return raw_dir / f"crsp_common_stocks_{compact_date(snapshot_date)}.csv"


def market_csv_path(raw_dir: Path, start_date: date, end_date: date) -> Path:
    return raw_dir / (
        f"crsp_market_returns_daily_{compact_date(start_date)}_"
        f"{compact_date(end_date)}.csv"
    )


def load_source_manifest(parquet_dir: Path) -> dict[str, object]:
    manifest_path = parquet_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Parquet source manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def source_partitions(
    parquet_dir: Path, start_date: date, end_date: date
) -> list[Path]:
    """Return ordered source paths after checking the source data contract."""
    manifest = load_source_manifest(parquet_dir)
    if manifest.get("requested_start_date") != start_date.isoformat():
        raise RuntimeError("Parquet source start date does not match requested CSV range")
    if manifest.get("requested_end_date") != end_date.isoformat():
        raise RuntimeError("Parquet source end date does not match requested CSV range")
    if tuple(manifest.get("columns", ())) != OUTPUT_COLUMNS:
        raise RuntimeError("Parquet source columns do not match the CRSP stock schema")

    paths = [
        parquet_dir / record["relative_path"]
        for record in manifest.get("partitions", [])
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Parquet source partition: {missing[0]}")
    if not paths:
        raise RuntimeError("Parquet source manifest contains no partitions")
    return paths


def export_stock_csv(
    *,
    parquet_dir: Path,
    output_path: Path,
    start_date: date,
    end_date: date,
) -> int:
    """Stream all monthly stock partitions into one atomic CSV file."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace stock CSV: {output_path}")
    partitions = source_partitions(parquet_dir, start_date, end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    total_rows = 0

    try:
        for index, path in enumerate(partitions, start=1):
            frame = pd.read_parquet(path, columns=list(OUTPUT_COLUMNS))
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame.to_csv(
                temporary,
                mode="w" if index == 1 else "a",
                header=index == 1,
                index=False,
                date_format="%Y-%m-%d",
            )
            total_rows += len(frame)
            if index == 1 or index % 12 == 0 or index == len(partitions):
                print(
                    f"Stock CSV: {index:03d}/{len(partitions):03d} partitions, "
                    f"{total_rows:,} rows",
                    flush=True,
                )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return total_rows


def export_stock_snapshot(
    *,
    parquet_dir: Path,
    output_path: Path,
    snapshot_date: date,
) -> int:
    """Extract one trading-day stock cross-section to an atomic CSV file."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace snapshot CSV: {output_path}")
    source = (
        parquet_dir
        / f"year={snapshot_date.year:04d}"
        / f"month={snapshot_date.month:02d}"
        / "crsp_daily.parquet"
    )
    if not source.exists():
        raise FileNotFoundError(f"Snapshot source partition not found: {source}")
    frame = pd.read_parquet(source, columns=list(OUTPUT_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[frame["date"].dt.date == snapshot_date].copy()
    if frame.empty:
        raise RuntimeError(f"No CRSP stock rows found for {snapshot_date}")
    if frame.duplicated(["date", "permno"]).any():
        raise RuntimeError("Snapshot contains duplicate (date, permno) keys")
    frame = frame.sort_values("permno", kind="mergesort")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(frame)


def download_market_returns(
    connection: SqlConnection,
    *,
    output_path: Path,
    start_date: date,
    end_date: date,
) -> int:
    """Download CRSP CIZ daily index returns to an atomic CSV file."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace market-return CSV: {output_path}")
    frame = connection.raw_sql(
        MARKET_RETURN_SQL,
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        date_cols=["date"],
    )
    missing = set(MARKET_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"Market-return response is missing columns: {sorted(missing)}")
    frame = frame.loc[:, MARKET_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty:
        raise RuntimeError("Market-return query returned no rows")
    if frame["date"].duplicated().any():
        raise RuntimeError("Market-return query returned duplicate dates")
    if frame["date"].dt.date.min() < start_date or frame["date"].dt.date.max() > end_date:
        raise RuntimeError("Market-return query returned dates outside the requested range")
    frame = frame.sort_values("date", kind="mergesort")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(frame)


def write_delivery_manifest(
    *,
    raw_dir: Path,
    start_date: date,
    end_date: date,
    snapshot_date: date,
    stock_path: Path,
    stock_rows: int,
    snapshot_path: Path,
    snapshot_rows: int,
    market_path: Path,
    market_rows: int,
) -> Path:
    manifest_path = raw_dir / "crsp_csv_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "snapshot_date": snapshot_date.isoformat(),
        "stock_source": "crsp.dsf_v2 joined to crsp.stksecurityinfohist",
        "market_source": "crsp.wrds_dailyindexret_query",
        "market_return_definition": {
            "primary": "vwretd",
            "description": "CRSP value-weighted market return including distributions",
        },
        "files": {
            "stocks": {
                "path": stock_path.name,
                "rows": stock_rows,
                "bytes": stock_path.stat().st_size,
            },
            "stock_snapshot": {
                "path": snapshot_path.name,
                "rows": snapshot_rows,
                "bytes": snapshot_path.stat().st_size,
            },
            "market_returns": {
                "path": market_path.name,
                "rows": market_rows,
                "bytes": market_path.stat().st_size,
            },
        },
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create CRSP stock and market-return CSV files in data/raw."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--snapshot-date", type=date.fromisoformat, default=DEFAULT_SNAPSHOT_DATE
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_date > args.end_date:
        raise ValueError("--start-date cannot be after --end-date")
    if not args.start_date <= args.snapshot_date <= args.end_date:
        raise ValueError("--snapshot-date must be inside the requested range")

    raw_dir = args.raw_dir.expanduser().resolve()
    parquet_dir = args.parquet_dir.expanduser().resolve()
    stocks = stock_csv_path(raw_dir, args.start_date, args.end_date)
    snapshot = snapshot_csv_path(raw_dir, args.snapshot_date)
    market = market_csv_path(raw_dir, args.start_date, args.end_date)
    conflicts = [path for path in (stocks, snapshot, market) if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to replace existing CSV files: {conflicts}")

    snapshot_rows = export_stock_snapshot(
        parquet_dir=parquet_dir,
        output_path=snapshot,
        snapshot_date=args.snapshot_date,
    )
    print(f"Snapshot CSV: {snapshot} ({snapshot_rows:,} rows)", flush=True)

    connection = connect_wrds()
    try:
        market_rows = download_market_returns(
            connection,
            output_path=market,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    finally:
        connection.close()
    print(f"Market CSV: {market} ({market_rows:,} rows)", flush=True)

    stock_rows = export_stock_csv(
        parquet_dir=parquet_dir,
        output_path=stocks,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Stock CSV: {stocks} ({stock_rows:,} rows)", flush=True)

    manifest = write_delivery_manifest(
        raw_dir=raw_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        snapshot_date=args.snapshot_date,
        stock_path=stocks,
        stock_rows=stock_rows,
        snapshot_path=snapshot,
        snapshot_rows=snapshot_rows,
        market_path=market,
        market_rows=market_rows,
    )
    print(f"Delivery manifest: {manifest}", flush=True)
    return 0

