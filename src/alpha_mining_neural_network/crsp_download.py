"""Download CRSP CIZ daily U.S. common-stock data from WRDS.

The downloader writes one Parquet file per calendar month.  Small partitions
keep memory use bounded and make an interrupted multi-decade download safely
resumable.
"""

from __future__ import annotations

import argparse
import json
import os
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import wrds


DEFAULT_START_DATE = date(2000, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "raw" / "crsp_daily"

OUTPUT_COLUMNS = (
    "permno",
    "permco",
    "ticker",
    "cusip",
    "primaryexch",
    "siccd",
    "date",
    "prc",
    "ret",
    "retx",
    "vol",
    "mktcap",
    "open",
    "high",
    "low",
    "close",
    "delist_flag",
)

COMMON_STOCK_SQL = """
SELECT
    dsf.permno,
    ssih.permco,
    ssih.ticker,
    ssih.cusip,
    ssih.primaryexch,
    ssih.siccd,
    dsf.dlycaldt AS date,
    dsf.dlyprc AS prc,
    dsf.dlyret AS ret,
    dsf.dlyretx AS retx,
    dsf.dlyvol AS vol,
    dsf.dlycap AS mktcap,
    dsf.dlyopen AS open,
    dsf.dlyhigh AS high,
    dsf.dlylow AS low,
    dsf.dlyclose AS close,
    dsf.dlydelflg AS delist_flag
FROM crsp.dsf_v2 AS dsf
INNER JOIN crsp.stksecurityinfohist AS ssih
    ON dsf.permno = ssih.permno
    AND ssih.secinfostartdt <= dsf.dlycaldt
    AND dsf.dlycaldt <= ssih.secinfoenddt
WHERE dsf.dlycaldt BETWEEN %(start_date)s AND %(end_date)s
    AND ssih.sharetype = 'NS'
    AND ssih.securitytype = 'EQTY'
    AND ssih.securitysubtype = 'COM'
    AND ssih.usincflg = 'Y'
    AND ssih.issuertype IN ('ACOR', 'CORP')
    AND ssih.primaryexch IN ('N', 'A', 'Q')
    AND ssih.conditionaltype IN ('RW', 'NW')
    AND ssih.tradingstatusflg = 'A'
ORDER BY dsf.dlycaldt, dsf.permno
"""


class SqlConnection(Protocol):
    """The subset of the WRDS connection API used by this module."""

    def raw_sql(self, sql: str, **kwargs: object) -> pd.DataFrame: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PartitionRecord:
    year: int
    month: int
    start_date: str
    end_date: str
    relative_path: str
    rows: int
    bytes: int


def parse_iso_date(value: str) -> date:
    """Parse a CLI date in ISO format."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid date; use YYYY-MM-DD"
        ) from exc


def month_intervals(start_date: date, end_date: date) -> list[tuple[date, date]]:
    """Return inclusive calendar-month intervals clipped to the requested range."""
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    intervals: list[tuple[date, date]] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        first = date(year, month, 1)
        last = date(year, month, monthrange(year, month)[1])
        intervals.append((max(first, start_date), min(last, end_date)))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return intervals


def connect_wrds() -> wrds.Connection:
    """Open WRDS using WRDS_USER and the standard ~/.pgpass credentials."""
    username = os.environ.get("WRDS_USER")
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def check_coverage(connection: SqlConnection, required_end_date: date) -> date:
    """Fail before downloading if CRSP does not cover the requested end date."""
    result = connection.raw_sql(
        "SELECT MAX(dlycaldt) AS max_date FROM crsp.dsf_v2"
    )
    if result.empty or pd.isna(result.loc[0, "max_date"]):
        raise RuntimeError("crsp.dsf_v2 is empty or inaccessible")
    maximum = pd.Timestamp(result.loc[0, "max_date"]).date()
    if maximum < required_end_date:
        raise RuntimeError(
            f"CRSP coverage ends at {maximum}, before requested {required_end_date}"
        )
    return maximum


def fetch_partition(
    connection: SqlConnection, start_date: date, end_date: date
) -> pd.DataFrame:
    """Query and validate one inclusive date interval."""
    frame = connection.raw_sql(
        COMMON_STOCK_SQL,
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        date_cols=["date"],
    )
    missing = set(OUTPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"WRDS response is missing columns: {sorted(missing)}")

    frame = frame.loc[:, OUTPUT_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    outside = ~frame["date"].dt.date.between(start_date, end_date)
    if outside.any():
        raise RuntimeError("WRDS returned rows outside the requested date interval")
    duplicates = frame.duplicated(["date", "permno"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["date", "permno"]].head().to_dict("records")
        raise RuntimeError(f"Duplicate (date, permno) rows returned: {examples}")
    return frame.sort_values(["date", "permno"], kind="mergesort").reset_index(drop=True)


def partition_path(output_dir: Path, interval_start: date) -> Path:
    """Return the Hive-style monthly Parquet path."""
    return (
        output_dir
        / f"year={interval_start.year:04d}"
        / f"month={interval_start.month:02d}"
        / "crsp_daily.parquet"
    )


def inspect_existing_partition(
    path: Path, start_date: date, end_date: date, output_dir: Path
) -> PartitionRecord:
    """Validate enough metadata to safely resume past an existing partition."""
    parquet = pd.read_parquet(path, columns=["date", "permno"])
    if list(parquet.columns) != ["date", "permno"]:
        raise RuntimeError(f"Existing partition has an invalid schema: {path}")
    if parquet.empty:
        raise RuntimeError(f"Existing partition is empty: {path}")
    parquet["date"] = pd.to_datetime(parquet["date"], errors="raise")
    if parquet["date"].dt.date.min() < start_date or parquet["date"].dt.date.max() > end_date:
        raise RuntimeError(f"Existing partition contains dates outside its interval: {path}")
    if parquet.duplicated(["date", "permno"]).any():
        raise RuntimeError(f"Existing partition contains duplicate keys: {path}")
    return make_record(path, start_date, end_date, len(parquet), output_dir)


def make_record(
    path: Path,
    start_date: date,
    end_date: date,
    rows: int,
    output_dir: Path,
) -> PartitionRecord:
    return PartitionRecord(
        year=start_date.year,
        month=start_date.month,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        relative_path=str(path.relative_to(output_dir)),
        rows=rows,
        bytes=path.stat().st_size,
    )


def write_partition(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish a compressed Parquet partition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_manifest(
    output_dir: Path,
    start_date: date,
    end_date: date,
    crsp_max_date: date,
    records: list[PartitionRecord],
) -> Path:
    """Write dataset provenance and partition counts atomically."""
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    manifest = {
        "dataset": "CRSP CIZ daily U.S. common stocks",
        "source_tables": ["crsp.dsf_v2", "crsp.stksecurityinfohist"],
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "crsp_max_date_at_download": crsp_max_date.isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "parquet",
        "partitioning": ["year", "month"],
        "columns": list(OUTPUT_COLUMNS),
        "filters": {
            "sharetype": ["NS"],
            "securitytype": ["EQTY"],
            "securitysubtype": ["COM"],
            "usincflg": ["Y"],
            "issuertype": ["ACOR", "CORP"],
            "primaryexch": ["N", "A", "Q"],
            "conditionaltype": ["RW", "NW"],
            "tradingstatusflg": ["A"],
        },
        "total_rows": sum(record.rows for record in records),
        "total_bytes": sum(record.bytes for record in records),
        "partitions": [asdict(record) for record in records],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest_path


def download_crsp_daily(
    connection: SqlConnection,
    *,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
) -> list[PartitionRecord]:
    """Download all monthly partitions and return their metadata records."""
    output_dir = output_dir.expanduser().resolve()
    crsp_max_date = check_coverage(connection, end_date)
    intervals = month_intervals(start_date, end_date)
    records: list[PartitionRecord] = []

    for index, (interval_start, interval_end) in enumerate(intervals, start=1):
        path = partition_path(output_dir, interval_start)
        label = f"[{index:03d}/{len(intervals):03d}] {interval_start:%Y-%m}"
        if path.exists():
            if not resume:
                raise FileExistsError(f"Partition already exists: {path}")
            record = inspect_existing_partition(
                path, interval_start, interval_end, output_dir
            )
            records.append(record)
            print(f"{label}: already present ({record.rows:,} rows)", flush=True)
            continue

        print(f"{label}: downloading...", flush=True)
        frame = fetch_partition(connection, interval_start, interval_end)
        if frame.empty:
            print(f"{label}: no trading rows", flush=True)
            continue
        write_partition(frame, path)
        record = make_record(path, interval_start, interval_end, len(frame), output_dir)
        records.append(record)
        print(
            f"{label}: wrote {record.rows:,} rows ({record.bytes / 1_048_576:.1f} MiB)",
            flush=True,
        )

    manifest_path = write_manifest(
        output_dir, start_date, end_date, crsp_max_date, records
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total rows: {sum(record.rows for record in records):,}", flush=True)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download CRSP CIZ daily U.S. common-stock data from WRDS."
    )
    parser.add_argument("--start-date", type=parse_iso_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_iso_date, default=DEFAULT_END_DATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail instead of validating and skipping existing monthly partitions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_date > args.end_date:
        raise ValueError("--start-date cannot be after --end-date")
    connection = connect_wrds()
    try:
        download_crsp_daily(
            connection,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )
    finally:
        connection.close()
    return 0

