"""Validate and canonicalize the raw CRSP stock panel.

The canonical contract is one row per ``(permno, date)`` key, globally sorted
by that key.  Publication is atomic: an invalid or interrupted build never
replaces a completed canonical dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_mining_neural_network.crsp_download import OUTPUT_COLUMNS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "crsp_common_stocks_daily_20000101_20251231.csv"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "canonical"
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "crsp_common_stocks_daily_20000101_20251231.csv"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "crsp_canonical_manifest.json"

INTEGER_COLUMNS = ("permno", "permco", "siccd")
FLOAT_COLUMNS = (
    "prc",
    "ret",
    "retx",
    "vol",
    "mktcap",
    "open",
    "high",
    "low",
    "close",
)
STRING_COLUMNS = (
    "ticker",
    "cusip",
    "primaryexch",
    "delist_flag",
)


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    assets: int
    min_date: str
    max_date: str
    duplicate_keys: int
    missing_key_rows: int
    order_violations: int


def validate_header(input_path: Path) -> None:
    """Require the exact raw-data column contract before sorting."""
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
    if tuple(header) != OUTPUT_COLUMNS:
        raise RuntimeError(
            f"Unexpected raw CSV columns. Expected {list(OUTPUT_COLUMNS)}, got {header}"
        )


def external_sort_by_key(input_path: Path, temporary_output: Path) -> None:
    """Use GNU sort for a bounded-memory global ``permno,date`` ordering."""
    validate_header(input_path)
    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(
        dir=temporary_output.parent, prefix=".crsp-sort-"
    ) as sort_directory:
        with input_path.open("rb") as source, temporary_output.open("wb") as target:
            header = source.readline()
            target.write(header)
            # subprocess inherits the underlying file descriptor, so publish the
            # buffered header and advance its shared offset before sort writes.
            target.flush()
            tail = subprocess.Popen(
                ["tail", "-n", "+2", os.fspath(input_path)],
                stdout=subprocess.PIPE,
            )
            assert tail.stdout is not None
            try:
                sort_result = subprocess.run(
                    [
                        "sort",
                        "--stable",
                        "--field-separator=,",
                        "--key=1,1n",
                        "--key=7,7",
                        "--parallel=4",
                        "--buffer-size=25%",
                        f"--temporary-directory={sort_directory}",
                    ],
                    stdin=tail.stdout,
                    stdout=target,
                    check=False,
                )
            finally:
                tail.stdout.close()
            tail_return_code = tail.wait()
            if tail_return_code != 0:
                raise RuntimeError(f"tail failed with exit code {tail_return_code}")
            if sort_result.returncode != 0:
                raise RuntimeError(
                    f"sort failed with exit code {sort_result.returncode}"
                )


def validation_dtypes() -> dict[str, str]:
    dtypes = {column: "string" for column in STRING_COLUMNS}
    dtypes.update({column: "float64" for column in FLOAT_COLUMNS})
    dtypes.update({"permno": "int64", "permco": "Int64", "siccd": "Int64"})
    dtypes["date"] = "string"
    return dtypes


def validate_canonical_csv(
    path: Path, *, chunk_size: int = 500_000
) -> ValidationSummary:
    """Validate schema, types, dates, key completeness, uniqueness, and order."""
    rows = 0
    assets = 0
    minimum_date: pd.Timestamp | None = None
    maximum_date: pd.Timestamp | None = None
    previous_permno: int | None = None
    previous_date: np.datetime64 | None = None
    duplicate_keys = 0
    missing_key_rows = 0
    order_violations = 0

    chunks = pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=validation_dtypes(),
        usecols=list(OUTPUT_COLUMNS),
    )
    for chunk_number, chunk in enumerate(chunks, start=1):
        if tuple(chunk.columns) != OUTPUT_COLUMNS:
            raise RuntimeError(f"Canonical schema mismatch in chunk {chunk_number}")
        dates = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="raise")
        missing_key_rows += int(chunk["permno"].isna().sum() + dates.isna().sum())
        permnos = chunk["permno"].to_numpy(dtype=np.int64, copy=False)
        date_values = dates.to_numpy(dtype="datetime64[ns]", copy=False)
        if len(chunk) == 0:
            continue

        if previous_permno is not None and previous_date is not None:
            if permnos[0] < previous_permno or (
                permnos[0] == previous_permno and date_values[0] <= previous_date
            ):
                order_violations += 1
                if permnos[0] == previous_permno and date_values[0] == previous_date:
                    duplicate_keys += 1

        same_permno = permnos[1:] == permnos[:-1]
        duplicate_keys += int(
            np.count_nonzero(same_permno & (date_values[1:] == date_values[:-1]))
        )
        order_violations += int(
            np.count_nonzero(
                (permnos[1:] < permnos[:-1])
                | (same_permno & (date_values[1:] <= date_values[:-1]))
            )
        )

        assets += int(np.count_nonzero(permnos[1:] != permnos[:-1]))
        if previous_permno is None or permnos[0] != previous_permno:
            assets += 1
        rows += len(chunk)
        chunk_min = dates.min()
        chunk_max = dates.max()
        minimum_date = chunk_min if minimum_date is None else min(minimum_date, chunk_min)
        maximum_date = chunk_max if maximum_date is None else max(maximum_date, chunk_max)
        previous_permno = int(permnos[-1])
        previous_date = date_values[-1]

        print(
            f"Validation: chunk {chunk_number:03d}, {rows:,} rows checked",
            flush=True,
        )

    if rows == 0 or minimum_date is None or maximum_date is None:
        raise RuntimeError("Canonical CSV contains no data rows")
    if missing_key_rows:
        raise RuntimeError(f"Found {missing_key_rows:,} rows with missing keys")
    if duplicate_keys:
        raise RuntimeError(f"Found {duplicate_keys:,} duplicate (permno, date) keys")
    if order_violations:
        raise RuntimeError(f"Found {order_violations:,} key-order violations")

    return ValidationSummary(
        rows=rows,
        assets=assets,
        min_date=minimum_date.date().isoformat(),
        max_date=maximum_date.date().isoformat(),
        duplicate_keys=duplicate_keys,
        missing_key_rows=missing_key_rows,
        order_violations=order_violations,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    *,
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    summary: ValidationSummary,
    sha256: str,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(input_path),
        "output": output_path.name,
        "primary_key": ["permno", "date"],
        "sort_order": ["permno", "date"],
        "columns": list(OUTPUT_COLUMNS),
        "validation": asdict(summary),
        "bytes": output_path.stat().st_size,
        "sha256": sha256,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def canonicalize_crsp(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> ValidationSummary:
    """Build and atomically publish the canonical CRSP stock CSV."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Raw CRSP CSV not found: {input_path}")
    conflicts = [path for path in (output_path, manifest_path) if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to replace canonical output: {conflicts}")
    if input_path == output_path:
        raise ValueError("Input and canonical output paths must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".part")
    try:
        print("Sorting raw CSV by (permno, date)...", flush=True)
        external_sort_by_key(input_path, temporary_output)
        print("Validating canonical schema and primary key...", flush=True)
        summary = validate_canonical_csv(temporary_output)
        print("Computing SHA-256...", flush=True)
        checksum = sha256_file(temporary_output)
        temporary_output.replace(output_path)
        write_manifest(
            manifest_path=manifest_path,
            input_path=input_path,
            output_path=output_path,
            summary=summary,
            sha256=checksum,
        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonicalize CRSP raw data to a unique (permno, date) panel."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = canonicalize_crsp(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(f"Canonical rows: {summary.rows:,}", flush=True)
    print(f"Unique assets: {summary.assets:,}", flush=True)
    print(f"Date range: {summary.min_date} to {summary.max_date}", flush=True)
    print(f"Canonical CSV: {args.output.expanduser().resolve()}", flush=True)
    print(f"Manifest: {args.manifest.expanduser().resolve()}", flush=True)
    return 0
