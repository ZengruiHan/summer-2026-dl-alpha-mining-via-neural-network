"""Create leakage-safe 8/1/1 rolling chronological folds."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANDARD_DIR = REPOSITORY_ROOT / "data" / "standard" / "standardized_features"
DEFAULT_STANDARD_MANIFEST = REPOSITORY_ROOT / "data" / "standard" / "standardization_manifest.json"
DEFAULT_LABELS = REPOSITORY_ROOT / "data" / "labels" / "daily_labels.csv"
DEFAULT_CELLULAR_DIR = REPOSITORY_ROOT / "data" / "cellular"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "chronological_folds"

TRAIN_YEARS = 8
VALIDATION_YEARS = 1
TEST_YEARS = 1
STEP_YEARS = 1


@dataclass(frozen=True)
class FoldSpec:
    index: int
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_year: int

    @property
    def name(self) -> str:
        return (
            f"fold_{self.index:02d}_train_{self.train_start_year}_{self.train_end_year}_"
            f"val_{self.validation_year}_test_{self.test_year}"
        )

    def years(self, split: str) -> list[int]:
        if split == "train":
            return list(range(self.train_start_year, self.train_end_year + 1))
        if split == "validation":
            return [self.validation_year]
        if split == "refit":
            return list(range(self.train_start_year, self.validation_year + 1))
        if split == "test":
            return [self.test_year]
        raise ValueError(f"Unknown split: {split}")


def generate_fold_specs(first_year: int, last_year: int) -> list[FoldSpec]:
    specs: list[FoldSpec] = []
    start = first_year
    while start + TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS - 1 <= last_year:
        validation_year = start + TRAIN_YEARS
        test_year = validation_year + VALIDATION_YEARS
        specs.append(
            FoldSpec(
                index=len(specs),
                train_start_year=start,
                train_end_year=validation_year - 1,
                validation_year=validation_year,
                test_year=test_year,
            )
        )
        start += STEP_YEARS
    if not specs:
        raise ValueError("No complete 8/1/1 fold fits the available year range")
    return specs


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def partition_labels(
    labels: pd.DataFrame,
    *,
    year: int,
    complete_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    feature_date_mask = labels["date"].ge(start) & labels["date"].lt(end)
    year_rows = labels.loc[feature_date_mask].copy()
    rows_before_purge = len(year_rows)
    boundary_mask = year_rows["target_date"].ge(start) & year_rows["target_date"].lt(end)
    missing_label_mask = year_rows["label"].isna() | year_rows["target_return"].isna()
    purged_boundary_rows = int((~boundary_mask).sum())
    missing_label_rows = int((boundary_mask & missing_label_mask).sum())
    year_rows = year_rows.loc[boundary_mask & ~missing_label_mask].copy()
    year_rows = year_rows.merge(
        complete_features,
        on=["date", "permno"],
        how="left",
        validate="one_to_one",
    )
    if year_rows["complete_numeric_features"].isna().any():
        raise RuntimeError(f"Feature completeness lookup failed for year {year}")
    year_rows["complete_numeric_features"] = year_rows[
        "complete_numeric_features"
    ].astype(bool)
    year_rows["label"] = year_rows["label"].astype("int8")
    year_rows = year_rows.sort_values(["date", "rank"], kind="mergesort").reset_index(
        drop=True
    )
    if year_rows.duplicated(["date", "permno"]).any():
        raise RuntimeError(f"Duplicate label keys after partitioning year {year}")
    stats = {
        "feature_rows": rows_before_purge,
        "boundary_purged_rows": purged_boundary_rows,
        "missing_label_rows": missing_label_rows,
        "labeled_rows": len(year_rows),
        "modeling_ready_rows": int(year_rows["complete_numeric_features"].sum()),
    }
    return year_rows, stats


def build_shared_label_partitions(
    *,
    labels_path: Path,
    standard_dir: Path,
    years: list[int],
    shared_dir: Path,
) -> dict[int, dict[str, dict[str, int | str]]]:
    labels = pd.read_csv(
        labels_path,
        dtype={
            "date": "string",
            "permno": "int32",
            "rank": "int16",
            "target_date": "string",
            "target_return": "float64",
            "label": "Int8",
        },
    )
    labels["date"] = pd.to_datetime(labels["date"], format="%Y-%m-%d", errors="raise")
    labels["target_date"] = pd.to_datetime(
        labels["target_date"], format="%Y-%m-%d", errors="coerce"
    )
    if labels.duplicated(["date", "permno"]).any():
        raise RuntimeError("Label source contains duplicate keys")

    numeric_columns: list[str] | None = None
    records: dict[int, dict[str, dict[str, int | str]]] = {}
    for year in years:
        feature_path = standard_dir / f"year={year:04d}" / "standardized_features.parquet"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        if numeric_columns is None:
            numeric_columns = [
                name
                for name in pq.ParquetFile(feature_path).schema_arrow.names
                if name.endswith("_z")
            ]
        features = pd.read_parquet(
            feature_path, columns=["date", "permno", *numeric_columns]
        )
        features["date"] = pd.to_datetime(features["date"], errors="raise")
        complete = features[["date", "permno"]].copy()
        complete["complete_numeric_features"] = features[numeric_columns].notna().all(
            axis=1
        )
        boundary_partition, boundary_stats = partition_labels(
            labels, year=year, complete_features=complete
        )
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        full_partition = labels.loc[
            labels["date"].ge(start)
            & labels["date"].lt(end)
            & labels["target_date"].notna()
            & labels["target_return"].notna()
            & labels["label"].notna()
        ].copy()
        full_partition = full_partition.merge(
            complete,
            on=["date", "permno"],
            how="left",
            validate="one_to_one",
        )
        if full_partition["complete_numeric_features"].isna().any():
            raise RuntimeError(f"Full feature completeness lookup failed for year {year}")
        full_partition["complete_numeric_features"] = full_partition[
            "complete_numeric_features"
        ].astype(bool)
        full_partition["label"] = full_partition["label"].astype("int8")
        full_partition = full_partition.sort_values(
            ["date", "rank"], kind="mergesort"
        ).reset_index(drop=True)

        full_output = (
            shared_dir / "labels_full" / f"year={year:04d}" / "labels.parquet"
        )
        boundary_output = (
            shared_dir / "labels_boundary" / f"year={year:04d}" / "labels.parquet"
        )
        full_output.parent.mkdir(parents=True, exist_ok=True)
        boundary_output.parent.mkdir(parents=True, exist_ok=True)
        full_partition.to_parquet(full_output, index=False, compression="zstd")
        boundary_partition.to_parquet(
            boundary_output, index=False, compression="zstd"
        )
        records[year] = {
            "full": {
                "labeled_rows": len(full_partition),
                "modeling_ready_rows": int(
                    full_partition["complete_numeric_features"].sum()
                ),
                "bytes": full_output.stat().st_size,
                "relative_path": str(full_output.relative_to(shared_dir.parent)),
            },
            "boundary": {
                **boundary_stats,
                "bytes": boundary_output.stat().st_size,
                "relative_path": str(boundary_output.relative_to(shared_dir.parent)),
            },
        }
        print(
            f"Shared labels {year}: {len(full_partition):,} full / "
            f"{len(boundary_partition):,} boundary-purged rows",
            flush=True,
        )
    return records


def build_folds(
    *,
    standard_dir: Path,
    standard_manifest_path: Path,
    labels_path: Path,
    cellular_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace fold directory: {output_dir}")
    standard_manifest = load_json(standard_manifest_path)
    cellular_manifest = load_json(cellular_dir / "cellular_manifest.json")
    standard_year_records = {
        int(record["year"]): record for record in standard_manifest["partitions"]
    }
    first_year = min(standard_year_records)
    last_year = max(standard_year_records)
    years = list(range(first_year, last_year + 1))
    specs = generate_fold_specs(first_year, last_year)

    temporary = output_dir.parent / f".{output_dir.name}.part"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    shared_dir = temporary / "_shared"
    try:
        label_records = build_shared_label_partitions(
            labels_path=labels_path,
            standard_dir=standard_dir,
            years=years,
            shared_dir=shared_dir,
        )
        fold_manifests: list[dict[str, object]] = []
        view_files = sorted((cellular_dir / "views").glob("*.view.json"))
        if len(view_files) != int(cellular_manifest["view_count"]):
            raise RuntimeError("Cellular view count does not match its manifest")

        cellular_year_records: dict[str, dict[int, dict[str, object]]] = {
            dataset: {int(record["year"]): record for record in records}
            for dataset, records in cellular_manifest["datasets"].items()
        }
        for spec in specs:
            fold_dir = temporary / spec.name
            views_dir = fold_dir / "cellular_views"
            for view in view_files:
                hardlink_or_copy(view, views_dir / view.name)

            split_payloads: dict[str, object] = {}
            for split in ("train", "validation", "refit", "test"):
                split_years = spec.years(split)
                split_dir = fold_dir / split
                label_rows = 0
                modeling_rows = 0
                feature_rows = 0
                end_year = max(split_years)
                label_variants: dict[str, str] = {}
                for year in split_years:
                    feature_source = (
                        standard_dir
                        / f"year={year:04d}"
                        / "standardized_features.parquet"
                    )
                    feature_destination = (
                        split_dir
                        / "standardized_features"
                        / f"year={year:04d}"
                        / "standardized_features.parquet"
                    )
                    hardlink_or_copy(feature_source, feature_destination)
                    feature_rows += int(standard_year_records[year]["rows"])

                    label_variant = "boundary" if year == end_year else "full"
                    label_variants[str(year)] = label_variant
                    label_source = shared_dir.parent / str(
                        label_records[year][label_variant]["relative_path"]
                    )
                    label_destination = (
                        split_dir / "labels" / f"year={year:04d}" / "labels.parquet"
                    )
                    hardlink_or_copy(label_source, label_destination)
                    label_rows += int(
                        label_records[year][label_variant]["labeled_rows"]
                    )
                    modeling_rows += int(
                        label_records[year][label_variant]["modeling_ready_rows"]
                    )

                    for dataset, by_year in cellular_year_records.items():
                        record = by_year[year]
                        cellular_source = cellular_dir / str(record["relative_path"])
                        cellular_destination = (
                            split_dir
                            / "cellular"
                            / dataset
                            / f"year={year:04d}"
                            / "part.parquet"
                        )
                        hardlink_or_copy(cellular_source, cellular_destination)

                start_year = min(split_years)
                cellular_rows = {
                    dataset: sum(int(by_year[year]["rows"]) for year in split_years)
                    for dataset, by_year in cellular_year_records.items()
                }
                split_manifest = {
                    "split": split,
                    "years": split_years,
                    "start_date_inclusive": f"{start_year:04d}-01-01",
                    "end_date_exclusive": f"{end_year + 1:04d}-01-01",
                    "feature_rows": feature_rows,
                    "labeled_rows_after_boundary_purge": label_rows,
                    "modeling_ready_rows": modeling_rows,
                    "label_partition_variants": label_variants,
                    "cellular_rows_by_master_dataset": cellular_rows,
                    "join_contract": (
                        "Inner join standardized_features and labels on (date, permno); "
                        "use complete_numeric_features as the modeling mask"
                    ),
                    "cellular_contract": (
                        "Select a view from ../../cellular_views and apply its filter "
                        "to the referenced master cellular dataset"
                    ),
                }
                (split_dir / "split_manifest.json").write_text(
                    json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
                )
                split_payloads[split] = split_manifest

            fold_manifest = {
                **asdict(spec),
                "name": spec.name,
                "train_years": TRAIN_YEARS,
                "validation_years": VALIDATION_YEARS,
                "refit_years": TRAIN_YEARS + VALIDATION_YEARS,
                "test_years": TEST_YEARS,
                "step_years": STEP_YEARS,
                "boundary_policy": (
                    "Rows are assigned by feature date; labels are retained only when "
                    "target_date is inside the same split, so one-step targets "
                    "never cross train/validation/test boundaries"
                ),
                "refit_policy": (
                    "After validation selection, refit spans train_start through test_start. "
                    "Its label boundary is test_start, restoring rows at the former "
                    "train/validation boundary because they are now internal to refit"
                ),
                "splits": split_payloads,
            }
            (fold_dir / "fold_manifest.json").write_text(
                json.dumps(fold_manifest, indent=2) + "\n", encoding="utf-8"
            )
            fold_manifests.append(fold_manifest)
            print(f"Created {spec.name}", flush=True)

        top_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "standard_input": str(standard_dir.resolve()),
            "label_input": str(labels_path.resolve()),
            "cellular_input": str(cellular_dir.resolve()),
            "layout": "8-year train / 1-year validation / 1-year test; roll 1 year",
            "refit_layout": (
                "After validation selection, use the 9-year train+validation union; "
                "its target boundary is the test start"
            ),
            "fold_count": len(specs),
            "first_fold": specs[0].name,
            "last_fold": specs[-1].name,
            "storage": (
                "Feature and cellular Parquet partitions are hard-linked when possible; "
                "this provides real per-fold files without duplicating disk blocks"
            ),
            "label_boundary_policy": (
                "Intermediate train years retain cross-year targets because both dates are "
                "inside train. The final year of every split uses a boundary-purged label "
                "partition that excludes missing/unlabeled targets and targets outside the split"
            ),
            "shared_label_partitions": {
                str(year): record for year, record in label_records.items()
            },
            "folds": [
                {
                    "name": fold["name"],
                    "train_start_year": fold["train_start_year"],
                    "train_end_year": fold["train_end_year"],
                    "validation_year": fold["validation_year"],
                    "test_year": fold["test_year"],
                }
                for fold in fold_manifests
            ],
        }
        (temporary / "chronological_folds_manifest.json").write_text(
            json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return top_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe 8/1/1 folds.")
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument(
        "--standard-manifest", type=Path, default=DEFAULT_STANDARD_MANIFEST
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--cellular-dir", type=Path, default=DEFAULT_CELLULAR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_folds(
        standard_dir=args.standard_dir.expanduser().resolve(),
        standard_manifest_path=args.standard_manifest.expanduser().resolve(),
        labels_path=args.labels.expanduser().resolve(),
        cellular_dir=args.cellular_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(f"Chronological folds: {manifest['fold_count']}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0
