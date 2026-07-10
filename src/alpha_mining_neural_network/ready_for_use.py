"""Prepare aligned node tensors, supervision, and sparse graph skeletons."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANDARD_DIR = REPOSITORY_ROOT / "data" / "standard" / "standardized_features"
DEFAULT_STANDARD_MANIFEST = REPOSITORY_ROOT / "data" / "standard" / "standardization_manifest.json"
DEFAULT_SECTOR_MAPPING = REPOSITORY_ROOT / "data" / "standard" / "sector_mapping.json"
DEFAULT_CELLULAR_DIR = REPOSITORY_ROOT / "data" / "cellular"
DEFAULT_FOLDS_DIR = REPOSITORY_ROOT / "data" / "chronological_folds"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"

LABEL_SENTINEL = np.int8(-100)
SIGNED_LABEL_SENTINEL = np.int8(-128)
EDGE_DATASETS = ("correlation_edges", "lead_lag_edges", "beta_topk_edges")
GROUP_DATASETS = ("beta_tau_groups",)


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            hardlink_or_copy(path, destination / path.relative_to(source))


def build_feature_arrays(
    feature_path: Path,
    numeric_columns: list[str],
    output_dir: Path,
) -> dict[str, object]:
    columns = ["date", "permno", "rank", *numeric_columns, "sector_index"]
    frame = pd.read_parquet(feature_path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True)
    counts = frame.groupby("date", sort=False).size()
    if (counts != 500).any():
        raise RuntimeError(f"Feature shard {feature_path} does not have 500 nodes per date")
    if frame.duplicated(["date", "permno"]).any():
        raise RuntimeError(f"Duplicate feature keys in {feature_path}")
    dates = frame["date"].drop_duplicates().to_numpy(dtype="datetime64[D]")
    date_count = len(dates)
    expected_ranks = np.tile(np.arange(1, 501, dtype=np.int16), date_count)
    if not np.array_equal(frame["rank"].to_numpy(dtype=np.int16), expected_ranks):
        raise RuntimeError(f"Ranks are not contiguous in {feature_path}")

    permno = frame["permno"].to_numpy(dtype=np.int32).reshape(date_count, 500)
    x_numeric = frame[numeric_columns].to_numpy(dtype=np.float32).reshape(
        date_count, 500, len(numeric_columns)
    )
    sector_index = frame["sector_index"].to_numpy(dtype=np.int16).reshape(
        date_count, 500
    )
    if np.isinf(x_numeric).any():
        raise RuntimeError(f"Infinite standardized feature in {feature_path}")
    complete = np.isfinite(x_numeric).all(axis=2)
    # Preserve all 500 graph nodes. Missing standardized values (principally
    # beta during warm-up) are neutral at zero after cross-sectional z-scoring;
    # complete_numeric_mask prevents them from entering supervised loss.
    x_numeric = np.nan_to_num(x_numeric, nan=0.0, posinf=0.0, neginf=0.0)

    save_npy(output_dir / "dates.npy", dates)
    save_npy(output_dir / "permno.npy", permno)
    save_npy(output_dir / "x_numeric.npy", x_numeric)
    save_npy(output_dir / "sector_index.npy", sector_index)
    save_npy(output_dir / "complete_numeric_mask.npy", complete)
    return {
        "dates": date_count,
        "nodes": int(date_count * 500),
        "numeric_features": len(numeric_columns),
        "complete_numeric_nodes": int(complete.sum()),
        "missing_numeric_nodes": int((~complete).sum()),
        "imputation": "missing standardized numeric values set to 0.0",
    }


def build_supervision_arrays(
    labels_path: Path,
    feature_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    dates = np.load(feature_dir / "dates.npy", mmap_mode="r")
    permno = np.load(feature_dir / "permno.npy", mmap_mode="r")
    complete = np.load(feature_dir / "complete_numeric_mask.npy", mmap_mode="r")
    date_count = len(dates)
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"], errors="raise")
    labels["target_date"] = pd.to_datetime(labels["target_date"], errors="raise")
    if labels.duplicated(["date", "permno"]).any():
        raise RuntimeError(f"Duplicate supervision keys in {labels_path}")
    date_index = pd.Index(dates.astype("datetime64[ns]")).get_indexer(labels["date"])
    rank_index = labels["rank"].to_numpy(dtype=np.int64) - 1
    if (date_index < 0).any() or (rank_index < 0).any() or (rank_index >= 500).any():
        raise RuntimeError(f"Supervision key outside feature tensor in {labels_path}")
    expected_permno = permno[date_index, rank_index]
    if not np.array_equal(expected_permno, labels["permno"].to_numpy(dtype=np.int32)):
        raise RuntimeError(f"Supervision PERMNO/rank misalignment in {labels_path}")

    y = np.full((date_count, 500), LABEL_SENTINEL, dtype=np.int8)
    y_signed = np.full((date_count, 500), SIGNED_LABEL_SENTINEL, dtype=np.int8)
    target_return = np.full((date_count, 500), np.nan, dtype=np.float32)
    target_date = np.full((date_count, 500), np.datetime64("NaT"), dtype="datetime64[D]")
    label_mask = np.zeros((date_count, 500), dtype=bool)
    signed_values = labels["label"].to_numpy(dtype=np.int8)
    y_signed[date_index, rank_index] = signed_values
    y[date_index, rank_index] = signed_values + 2
    target_return[date_index, rank_index] = labels["target_return"].to_numpy(
        dtype=np.float32
    )
    target_date[date_index, rank_index] = labels["target_date"].to_numpy(
        dtype="datetime64[D]"
    )
    label_mask[date_index, rank_index] = True
    model_mask = label_mask & complete
    if not set(np.unique(y[label_mask])).issubset({0, 1, 2, 3, 4}):
        raise RuntimeError(f"Invalid target class in {labels_path}")
    if not np.all(target_date[label_mask] > np.repeat(dates[:, None], 500, axis=1)[label_mask]):
        raise RuntimeError(f"Non-forward target date in {labels_path}")

    save_npy(output_dir / "y.npy", y)
    save_npy(output_dir / "y_signed.npy", y_signed)
    save_npy(output_dir / "target_return.npy", target_return)
    save_npy(output_dir / "target_date.npy", target_date)
    save_npy(output_dir / "label_mask.npy", label_mask)
    save_npy(output_dir / "model_mask.npy", model_mask)
    save_npy(output_dir / "loss_mask.npy", model_mask)
    save_npy(output_dir / "day_model_mask.npy", model_mask.any(axis=1))
    return {
        "labeled_nodes": int(label_mask.sum()),
        "model_ready_nodes": int(model_mask.sum()),
        "masked_labels": int((~label_mask).sum()),
        "class_counts": {
            str(value): int((y_signed[label_mask] == value).sum())
            for value in (-2, -1, 0, 1, 2)
        },
    }


def map_local_ranks(
    dates: np.ndarray,
    permno: np.ndarray,
    edge_dates: pd.Series,
    source_permno: np.ndarray,
    target_permno: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    date_index = pd.Index(dates.astype("datetime64[ns]")).get_indexer(edge_dates)
    if (date_index < 0).any():
        raise RuntimeError("Graph contains a date absent from node tensors")
    source_rank = np.empty(len(edge_dates), dtype=np.int16)
    target_rank = None if target_permno is None else np.empty(len(edge_dates), dtype=np.int16)
    unique_dates, starts, counts = np.unique(date_index, return_index=True, return_counts=True)
    for date_code, start, count in zip(unique_dates, starts, counts, strict=True):
        positions = slice(start, start + count)
        node_permnos = pd.Index(permno[date_code])
        source_rank[positions] = node_permnos.get_indexer(source_permno[positions]).astype(
            np.int16
        )
        if target_permno is not None and target_rank is not None:
            target_rank[positions] = node_permnos.get_indexer(
                target_permno[positions]
            ).astype(np.int16)
    if (source_rank < 0).any() or (target_rank is not None and (target_rank < 0).any()):
        raise RuntimeError("Graph endpoint is absent from its date's 500-node universe")
    return date_index.astype(np.int16), source_rank, target_rank


def date_pointer(date_index: np.ndarray, date_count: int) -> np.ndarray:
    counts = np.bincount(date_index.astype(np.int64), minlength=date_count)
    pointer = np.zeros(date_count + 1, dtype=np.int64)
    np.cumsum(counts, out=pointer[1:])
    return pointer


def build_edge_skeleton(
    source_path: Path,
    feature_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    dates = np.load(feature_dir / "dates.npy", mmap_mode="r")
    permno = np.load(feature_dir / "permno.npy", mmap_mode="r")
    frame = pd.read_parquet(source_path)
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["date", "source_permno", "target_permno"]).any():
        raise RuntimeError(f"Duplicate graph edges in {source_path}")
    if (frame["source_permno"] == frame["target_permno"]).any():
        raise RuntimeError(f"Self-loop found in source skeleton {source_path}")
    date_index, source_rank, target_rank = map_local_ranks(
        dates,
        permno,
        frame["date"],
        frame["source_permno"].to_numpy(dtype=np.int32),
        frame["target_permno"].to_numpy(dtype=np.int32),
    )
    assert target_rank is not None
    if (source_rank == target_rank).any():
        raise RuntimeError(f"Mapped self-loop found in {source_path}")
    weight = frame["weight"].to_numpy(dtype=np.float32)
    abs_weight = frame["abs_weight"].to_numpy(dtype=np.float32)
    topk_min_rank = frame["topk_min_rank"].to_numpy(dtype=np.uint8)
    if not np.isfinite(weight).all() or not np.allclose(np.abs(weight), abs_weight):
        raise RuntimeError(f"Invalid graph weights in {source_path}")

    pointer = date_pointer(date_index, len(dates))
    save_npy(output_dir / "date_ptr.npy", pointer)
    save_npy(output_dir / "graph_available.npy", np.diff(pointer) > 0)
    save_npy(output_dir / "src_rank.npy", source_rank)
    save_npy(output_dir / "dst_rank.npy", target_rank)
    save_npy(output_dir / "weight.npy", weight)
    save_npy(output_dir / "abs_weight.npy", abs_weight)
    save_npy(output_dir / "topk_min_rank.npy", topk_min_rank)
    return {
        "rows": len(frame),
        "dates_with_edges": int(np.unique(date_index).size),
        "topk_rank_min": int(topk_min_rank.min()) if len(frame) else None,
        "topk_rank_max": int(topk_min_rank.max()) if len(frame) else None,
    }


def build_group_skeleton(
    source_path: Path,
    feature_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    dates = np.load(feature_dir / "dates.npy", mmap_mode="r")
    permno = np.load(feature_dir / "permno.npy", mmap_mode="r")
    frame = pd.read_parquet(source_path)
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["date", "permno"]).any():
        raise RuntimeError(f"Duplicate beta group nodes in {source_path}")
    date_index, node_rank, _ = map_local_ranks(
        dates,
        permno,
        frame["date"],
        frame["permno"].to_numpy(dtype=np.int32),
    )
    beta_sign = frame["beta_sign"].to_numpy(dtype=np.int8)
    beta = frame["beta_60_z"].to_numpy(dtype=np.float32)
    if not set(np.unique(beta_sign)).issubset({-1, 1}):
        raise RuntimeError(f"Invalid beta signs in {source_path}")
    pointer = date_pointer(date_index, len(dates))
    save_npy(output_dir / "date_ptr.npy", pointer)
    save_npy(output_dir / "graph_available.npy", np.diff(pointer) > 0)
    save_npy(output_dir / "node_rank.npy", node_rank)
    save_npy(output_dir / "beta_sign.npy", beta_sign)
    save_npy(output_dir / "beta_60_z.npy", beta)
    return {"rows": len(frame), "dates_with_groups": int(np.unique(date_index).size)}


def prepare_shared_years(
    *,
    years: list[int],
    standard_dir: Path,
    numeric_columns: list[str],
    cellular_dir: Path,
    folds_dir: Path,
    shared_dir: Path,
) -> dict[str, object]:
    records: dict[str, object] = {}
    for year in years:
        feature_dir = shared_dir / "features" / f"year={year:04d}"
        feature_record = build_feature_arrays(
            standard_dir / f"year={year:04d}" / "standardized_features.parquet",
            numeric_columns,
            feature_dir,
        )
        supervision_records: dict[str, object] = {}
        for variant in ("full", "boundary"):
            labels_path = (
                folds_dir
                / "_shared"
                / f"labels_{variant}"
                / f"year={year:04d}"
                / "labels.parquet"
            )
            supervision_records[variant] = build_supervision_arrays(
                labels_path,
                feature_dir,
                shared_dir / f"supervision_{variant}" / f"year={year:04d}",
            )

        skeleton_records: dict[str, object] = {}
        for dataset in EDGE_DATASETS:
            skeleton_records[dataset] = build_edge_skeleton(
                cellular_dir / dataset / f"year={year:04d}" / "part.parquet",
                feature_dir,
                shared_dir / "skeletons" / dataset / f"year={year:04d}",
            )
        for dataset in GROUP_DATASETS:
            skeleton_records[dataset] = build_group_skeleton(
                cellular_dir / dataset / f"year={year:04d}" / "part.parquet",
                feature_dir,
                shared_dir / "skeletons" / dataset / f"year={year:04d}",
            )
        records[str(year)] = {
            "features": feature_record,
            "supervision": supervision_records,
            "skeletons": skeleton_records,
        }
        print(
            f"Ready shared year {year}: {feature_record['dates']} dates, "
            f"{sum(int(record['rows']) for record in skeleton_records.values()):,} skeleton rows",
            flush=True,
        )
    return records


def assemble_fold_links(
    *,
    folds_dir: Path,
    shared_dir: Path,
    output_root: Path,
) -> list[dict[str, object]]:
    fold_records: list[dict[str, object]] = []
    for source_fold in sorted(folds_dir.glob("fold_*")):
        if not source_fold.is_dir():
            continue
        destination_fold = output_root / source_fold.name
        fold_manifest = load_json(source_fold / "fold_manifest.json")
        split_records: dict[str, object] = {}
        for split in ("train", "validation", "refit", "test"):
            split_manifest = load_json(source_fold / split / "split_manifest.json")
            destination_split = destination_fold / split
            years = [int(year) for year in split_manifest["years"]]
            variants = split_manifest["label_partition_variants"]
            for year in years:
                hardlink_tree(
                    shared_dir / "features" / f"year={year:04d}",
                    destination_split / "features" / f"year={year:04d}",
                )
                variant = str(variants[str(year)])
                hardlink_tree(
                    shared_dir / f"supervision_{variant}" / f"year={year:04d}",
                    destination_split / "supervision" / f"year={year:04d}",
                )
                for dataset in (*EDGE_DATASETS, *GROUP_DATASETS):
                    hardlink_tree(
                        shared_dir / "skeletons" / dataset / f"year={year:04d}",
                        destination_split
                        / "skeletons"
                        / dataset
                        / f"year={year:04d}",
                    )
            ready_split_manifest = {
                "split": split,
                "years": years,
                "label_variants": variants,
                "feature_layout": {
                    "x_numeric": "[date, 500, 8] float32",
                    "sector_index": "[date, 500] int16",
                    "permno": "[date, 500] int32",
                    "imputation": (
                        "missing standardized values are 0.0; consult "
                        "complete_numeric_mask before supervised loss"
                    ),
                },
                "supervision_layout": {
                    "y": "[date, 500] int8 class IDs 0..4; -100 is masked",
                    "y_signed": "[date, 500] int8 proposal labels -2..2; -128 is masked",
                    "label_mask": "[date, 500] bool",
                    "loss_mask": "label_mask AND complete_numeric_mask",
                    "day_model_mask": "true when a date has at least one supervised node",
                },
                "skeleton_layout": (
                    "date_ptr indexes unfiltered master arrays. Slice one date first, then "
                    "apply the cellular view filter; if globally compacting filtered edges, "
                    "rebuild date_ptr"
                ),
            }
            (destination_split / "ready_split_manifest.json").write_text(
                json.dumps(ready_split_manifest, indent=2) + "\n", encoding="utf-8"
            )
            split_records[split] = ready_split_manifest

        hardlink_tree(source_fold / "cellular_views", destination_fold / "cellular_views")
        ready_fold_manifest = {
            "source_fold": str(source_fold.resolve()),
            "fold": fold_manifest,
            "splits": split_records,
        }
        (destination_fold / "ready_fold_manifest.json").write_text(
            json.dumps(ready_fold_manifest, indent=2) + "\n", encoding="utf-8"
        )
        fold_records.append({"name": source_fold.name, "splits": list(split_records)})
        print(f"Assembled ready fold {source_fold.name}", flush=True)
    return fold_records


def build_ready_for_use(
    *,
    standard_dir: Path,
    standard_manifest_path: Path,
    sector_mapping_path: Path,
    cellular_dir: Path,
    folds_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace ready-for-use output: {output_dir}")
    standard_manifest = load_json(standard_manifest_path)
    sector_mapping = load_json(sector_mapping_path)
    cellular_manifest = load_json(cellular_dir / "cellular_manifest.json")
    fold_manifest = load_json(folds_dir / "chronological_folds_manifest.json")
    numeric_columns = [str(column) for column in standard_manifest["standardized_columns"]]
    years = sorted(int(record["year"]) for record in standard_manifest["partitions"])
    temporary = output_dir.parent / f".{output_dir.name}.part"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    shared_dir = temporary / "_shared"
    try:
        shared_records = prepare_shared_years(
            years=years,
            standard_dir=standard_dir,
            numeric_columns=numeric_columns,
            cellular_dir=cellular_dir,
            folds_dir=folds_dir,
            shared_dir=shared_dir,
        )
        fold_records = assemble_fold_links(
            folds_dir=folds_dir,
            shared_dir=shared_dir,
            output_root=temporary,
        )
        feature_spec = {
            "numeric_columns": numeric_columns,
            "x_numeric_shape_per_year": "[trading_dates, 500, 8]",
            "sector_categories": len(sector_mapping["categories"]),
            "combined_fixed_feature_dimension": len(numeric_columns)
            + len(sector_mapping["categories"]),
            "sector_one_hot": (
                "Construct lazily as identity[sector_index]; this is exactly the proposal "
                "one-hot encoding without storing a dense 690-column tensor"
            ),
            "sector_embedding": "Use sector_index as the learned embedding lookup input",
        }
        (temporary / "feature_spec.json").write_text(
            json.dumps(feature_spec, indent=2) + "\n", encoding="utf-8"
        )
        graph_spec = {
            "node_index": "zero-based daily TOP500 rank",
            "edge_datasets": list(EDGE_DATASETS),
            "implicit_group_datasets": list(GROUP_DATASETS),
            "view_count": cellular_manifest["view_count"],
            "view_files": cellular_manifest["views"],
            "edge_filter_fields": ["abs_weight", "topk_min_rank"],
            "proposal_adjacency_weight": {
                "correlation": "sign(raw weight): +1 for rho>0, -1 for rho<0",
                "lead_lag": "1 for every selected directed edge",
                "beta": "1 for every selected undirected edge",
                "raw_weight_note": (
                    "raw correlation/lead-lag values are selection statistics or optional "
                    "edge attributes, not the proposal adjacency values"
                ),
            },
            "filter_order": (
                "date_ptr belongs to unfiltered master arrays: slice by date before applying "
                "k/tau filters, or rebuild date_ptr after global compaction"
            ),
            "beta_tau_expansion": (
                "Within each date, connect every pair sharing beta_sign; exclude self loops"
            ),
            "undirected_expansion": (
                "Correlation and beta edges are stored once per pair; emit both directions "
                "at load time with duplicated weight"
            ),
            "directed_relation": "lead_lag src_rank -> dst_rank; do not symmetrize",
            "self_loops": "not stored; add exactly once in the model normalization layer",
            "graph_available": (
                "Per-year boolean array derived from master date_ptr; false during relation "
                "burn-in. A filtered view can still be empty when master graph is available"
            ),
            "burn_in": {
                "correlation_first_date": "2000-07-24",
                "lead_lag_first_date": "2000-07-28",
                "beta_first_date": "2000-03-29",
                "common_relation_first_date": "2000-07-28",
            },
            "degenerate_views": {
                "lead_lag_tau_0p6": "zero edges globally",
                "lead_lag_tau_0p8": "zero edges globally",
                "lead_lag_tau_0p4": "highly sparse; empty on many available dates",
                "beta_tau_all": "all four tau values are identical for scalar cosine similarity",
            },
        }
        (temporary / "graph_spec.json").write_text(
            json.dumps(graph_spec, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "standard_input": str(standard_dir.resolve()),
            "cellular_input": str(cellular_dir.resolve()),
            "fold_input": str(folds_dir.resolve()),
            "fold_count": fold_manifest["fold_count"],
            "years": years,
            "node_count_per_date": 500,
            "numeric_feature_count": len(numeric_columns),
            "sector_category_count": len(sector_mapping["categories"]),
            "label_sentinel": int(LABEL_SENTINEL),
            "signed_label_sentinel": int(SIGNED_LABEL_SENTINEL),
            "checks": {
                "node_key_alignment": "passed",
                "feature_infinite_values": 0,
                "feature_nan_values_after_imputation": 0,
                "graph_missing_endpoints": 0,
                "graph_self_loops": 0,
                "model_label_domain": [0, 1, 2, 3, 4],
                "signed_label_domain": [-2, -1, 0, 1, 2],
                "target_dates_strictly_forward": True,
            },
            "shared_years": shared_records,
            "folds": fold_records,
        }
        (temporary / "ready_for_use_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build model-ready aligned tensors and graphs.")
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument(
        "--standard-manifest", type=Path, default=DEFAULT_STANDARD_MANIFEST
    )
    parser.add_argument("--sector-mapping", type=Path, default=DEFAULT_SECTOR_MAPPING)
    parser.add_argument("--cellular-dir", type=Path, default=DEFAULT_CELLULAR_DIR)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_ready_for_use(
        standard_dir=args.standard_dir.expanduser().resolve(),
        standard_manifest_path=args.standard_manifest.expanduser().resolve(),
        sector_mapping_path=args.sector_mapping.expanduser().resolve(),
        cellular_dir=args.cellular_dir.expanduser().resolve(),
        folds_dir=args.folds_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(f"Ready folds: {manifest['fold_count']}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0
