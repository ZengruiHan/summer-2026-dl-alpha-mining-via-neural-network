"""Evaluate M0 OOS Rank IC, Pearson IC, and classification accuracy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from alpha_mining_neural_network.probability_export import (
    hardlink_or_copy,
    load_json,
    sha256_file,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBABILITY_DIR = REPOSITORY_ROOT / "results" / "probabilities"
DEFAULT_SCORE_DIR = REPOSITORY_ROOT / "results" / "ranking_scores"
DEFAULT_READY_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "prediction_metrics"


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks with deterministic stable sorting."""

    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Rank input must be one-dimensional")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Correlation inputs must be aligned one-dimensional arrays")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def evaluate_one_day(
    scores: np.ndarray,
    target_return: np.ndarray,
    predicted_class: np.ndarray,
    target_class: np.ndarray,
) -> tuple[float, float, float]:
    if not (
        scores.shape
        == target_return.shape
        == predicted_class.shape
        == target_class.shape
    ):
        raise ValueError("Daily metric arrays must align")
    if scores.ndim != 1 or len(scores) < 2:
        return float("nan"), float("nan"), float("nan")
    rank_ic = pearson_correlation(
        average_ranks(scores), average_ranks(target_return)
    )
    ic = pearson_correlation(scores, target_return)
    accuracy = float(np.mean(predicted_class == target_class))
    return rank_ic, ic, accuracy


def _save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else None


def _summary_metrics(
    rank_ic: np.ndarray,
    ic: np.ndarray,
    accuracy: np.ndarray,
    evaluation_count: np.ndarray,
) -> dict[str, Any]:
    return {
        "rank_ic": _mean_or_none(rank_ic),
        "ic": _mean_or_none(ic),
        "accuracy": _mean_or_none(accuracy),
        "calendar_dates": len(rank_ic),
        "evaluated_dates": int(np.isfinite(accuracy).sum()),
        "rank_ic_valid_dates": int(np.isfinite(rank_ic).sum()),
        "ic_valid_dates": int(np.isfinite(ic).sum()),
        "evaluation_nodes": int(evaluation_count.sum()),
    }


def _source_fingerprint(
    probability_manifest_path: Path,
    score_manifest_path: Path,
    source_records: dict[str, Any],
) -> str:
    payload = {
        "probability_manifest_sha256": sha256_file(probability_manifest_path),
        "score_manifest_sha256": sha256_file(score_manifest_path),
        "full_supervision_sources": source_records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate_prediction_metrics(
    *,
    probability_dir: Path,
    score_dir: Path,
    ready_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    probability_manifest_path = probability_dir / "probability_manifest.json"
    score_manifest_path = score_dir / "score_manifest.json"
    probability_manifest = load_json(probability_manifest_path)
    score_manifest = load_json(score_manifest_path)
    if probability_manifest.get("status") != "complete":
        raise RuntimeError("Probability input is incomplete")
    if score_manifest.get("status") != "complete":
        raise RuntimeError("Ranking score input is incomplete")
    if probability_manifest.get("model") != "M0" or score_manifest.get("model") != "M0":
        raise RuntimeError("Prediction inputs are not M0 artifacts")

    dates = np.load(probability_dir / "dates.npy", mmap_mode="r", allow_pickle=False)
    permno = np.load(
        probability_dir / "permno.npy", mmap_mode="r", allow_pickle=False
    )
    probabilities = np.load(
        probability_dir / "probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    inference_mask = np.load(
        probability_dir / "inference_mask.npy", mmap_mode="r", allow_pickle=False
    )
    boundary_evaluation_mask = np.load(
        probability_dir / "evaluation_mask.npy", mmap_mode="r", allow_pickle=False
    )
    fold_index = np.load(
        probability_dir / "fold_index.npy", mmap_mode="r", allow_pickle=False
    )
    scores = np.load(
        score_dir / "ranking_scores.npy", mmap_mode="r", allow_pickle=False
    )
    expected_node_shape = (len(dates), 500)
    if probabilities.shape != (*expected_node_shape, 5):
        raise RuntimeError("Unexpected probability shape")
    for name, values in (
        ("permno", permno),
        ("inference_mask", inference_mask),
        ("boundary_evaluation_mask", boundary_evaluation_mask),
        ("scores", scores),
    ):
        if values.shape != expected_node_shape:
            raise RuntimeError(f"{name} does not align with probabilities")
    if fold_index.shape != (len(dates),):
        raise RuntimeError("fold_index does not align with dates")
    expected_scores = probabilities @ np.arange(-2, 3, dtype=np.float32)
    if not np.array_equal(scores, expected_scores, equal_nan=True):
        raise RuntimeError("Ranking scores do not equal expected class values")

    full_evaluation_mask = np.zeros(expected_node_shape, dtype=bool)
    target_class = np.full(expected_node_shape, -100, dtype=np.int8)
    target_return = np.full(expected_node_shape, np.nan, dtype=np.float32)
    target_date = np.full(
        expected_node_shape, np.datetime64("NaT"), dtype="datetime64[D]"
    )
    source_records: dict[str, Any] = {}
    offset = 0
    for year in range(2009, 2026):
        feature_dir = ready_dir / "_shared" / "features" / f"year={year}"
        supervision_dir = (
            ready_dir / "_shared" / "supervision_full" / f"year={year}"
        )
        year_dates = np.load(
            feature_dir / "dates.npy", mmap_mode="r", allow_pickle=False
        )
        year_permno = np.load(
            feature_dir / "permno.npy", mmap_mode="r", allow_pickle=False
        )
        complete = np.load(
            feature_dir / "complete_numeric_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        y = np.load(supervision_dir / "y.npy", mmap_mode="r", allow_pickle=False)
        label_mask = np.load(
            supervision_dir / "label_mask.npy", mmap_mode="r", allow_pickle=False
        )
        loss_mask = np.load(
            supervision_dir / "loss_mask.npy", mmap_mode="r", allow_pickle=False
        )
        returns = np.load(
            supervision_dir / "target_return.npy", mmap_mode="r", allow_pickle=False
        )
        return_dates = np.load(
            supervision_dir / "target_date.npy", mmap_mode="r", allow_pickle=False
        )
        stop = offset + len(year_dates)
        if stop > len(dates):
            raise RuntimeError("Full supervision extends beyond predictions")
        if not np.array_equal(dates[offset:stop], year_dates):
            raise RuntimeError(f"Date alignment failed for full supervision {year}")
        if not np.array_equal(permno[offset:stop], year_permno):
            raise RuntimeError(f"PERMNO alignment failed for full supervision {year}")
        if not np.array_equal(inference_mask[offset:stop], complete):
            raise RuntimeError(f"Inference/feature mask alignment failed for {year}")
        expected_loss_mask = complete & label_mask & np.isfinite(returns)
        if not np.array_equal(loss_mask, expected_loss_mask):
            raise RuntimeError(f"Full loss mask contract failed for {year}")
        if not set(np.unique(y[loss_mask]).tolist()).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"Invalid full target class in {year}")
        date_grid = np.broadcast_to(year_dates[:, None], loss_mask.shape)
        if not np.all(return_dates[loss_mask] > date_grid[loss_mask]):
            raise RuntimeError(f"Non-forward full target date in {year}")

        full_evaluation_mask[offset:stop] = loss_mask
        target_class[offset:stop] = y
        target_return[offset:stop] = returns
        target_date[offset:stop] = return_dates
        source_records[str(year)] = {
            "dates_sha256": sha256_file(feature_dir / "dates.npy"),
            "permno_sha256": sha256_file(feature_dir / "permno.npy"),
            "complete_numeric_mask_sha256": sha256_file(
                feature_dir / "complete_numeric_mask.npy"
            ),
            "y_sha256": sha256_file(supervision_dir / "y.npy"),
            "label_mask_sha256": sha256_file(
                supervision_dir / "label_mask.npy"
            ),
            "loss_mask_sha256": sha256_file(supervision_dir / "loss_mask.npy"),
            "target_return_sha256": sha256_file(
                supervision_dir / "target_return.npy"
            ),
            "target_date_sha256": sha256_file(
                supervision_dir / "target_date.npy"
            ),
        }
        offset = stop
    if offset != len(dates):
        raise RuntimeError("Predictions contain dates outside 2009-2025 supervision")

    source_fingerprint = _source_fingerprint(
        probability_manifest_path, score_manifest_path, source_records
    )
    if output_dir.exists() and not overwrite:
        manifest_path = output_dir / "prediction_metrics.json"
        if not manifest_path.exists():
            raise RuntimeError("Existing prediction metric output lacks manifest")
        existing = load_json(manifest_path)
        if existing.get("source_fingerprint") != source_fingerprint:
            raise RuntimeError("Existing metrics belong to different inputs")
        for filename, record in existing["files"].items():
            path = output_dir / filename
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Existing metric artifact failed integrity: {path}")
        return existing

    predicted_class = probabilities.argmax(axis=2).astype(np.int8)
    predicted_class[~inference_mask] = -100
    daily_rank_ic = np.full(len(dates), np.nan, dtype=np.float64)
    daily_ic = np.full(len(dates), np.nan, dtype=np.float64)
    daily_accuracy = np.full(len(dates), np.nan, dtype=np.float64)
    daily_evaluation_count = full_evaluation_mask.sum(axis=1).astype(np.int16)
    for date_index in range(len(dates)):
        mask = full_evaluation_mask[date_index]
        if mask.sum() < 2:
            continue
        rank_ic, ic, accuracy = evaluate_one_day(
            scores[date_index, mask],
            target_return[date_index, mask],
            predicted_class[date_index, mask],
            target_class[date_index, mask],
        )
        daily_rank_ic[date_index] = rank_ic
        daily_ic[date_index] = ic
        daily_accuracy[date_index] = accuracy

    overall = _summary_metrics(
        daily_rank_ic, daily_ic, daily_accuracy, daily_evaluation_count
    )
    daily_rows: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        daily_rows.append(
            {
                "date": str(date),
                "fold_index": int(fold_index[index]),
                "evaluation_count": int(daily_evaluation_count[index]),
                "rank_ic": repr(float(daily_rank_ic[index])),
                "ic": repr(float(daily_ic[index])),
                "accuracy": repr(float(daily_accuracy[index])),
            }
        )

    fold_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(17):
        mask = fold_index == fold
        summary = _summary_metrics(
            daily_rank_ic[mask],
            daily_ic[mask],
            daily_accuracy[mask],
            daily_evaluation_count[mask],
        )
        summary.update({"fold_index": fold, "test_year": 2009 + fold})
        fold_summaries.append(summary)
        fold_rows.append(
            {
                "fold_index": fold,
                "test_year": 2009 + fold,
                "calendar_dates": summary["calendar_dates"],
                "evaluated_dates": summary["evaluated_dates"],
                "evaluation_nodes": summary["evaluation_nodes"],
                "rank_ic": repr(summary["rank_ic"]),
                "ic": repr(summary["ic"]),
                "accuracy": repr(summary["accuracy"]),
            }
        )

    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    temporary = output_dir.with_name(output_dir.name + ".part")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        files: dict[str, dict[str, Any]] = {}
        for filename in ("dates.npy", "permno.npy", "fold_index.npy"):
            source = probability_dir / filename
            destination = temporary / filename
            method = hardlink_or_copy(source, destination)
            files[filename] = {
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "publication_method": method,
            }
        derived_arrays = {
            "full_evaluation_mask.npy": full_evaluation_mask,
            "predicted_class.npy": predicted_class,
            "daily_rank_ic.npy": daily_rank_ic,
            "daily_ic.npy": daily_ic,
            "daily_accuracy.npy": daily_accuracy,
            "daily_evaluation_count.npy": daily_evaluation_count,
        }
        for filename, values in derived_arrays.items():
            path = temporary / filename
            _save_array(path, values)
            files[filename] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }
        daily_csv = temporary / "daily_metrics.csv"
        fold_csv = temporary / "fold_metrics.csv"
        _write_csv(
            daily_csv,
            ["date", "fold_index", "evaluation_count", "rank_ic", "ic", "accuracy"],
            daily_rows,
        )
        _write_csv(
            fold_csv,
            [
                "fold_index",
                "test_year",
                "calendar_dates",
                "evaluated_dates",
                "evaluation_nodes",
                "rank_ic",
                "ic",
                "accuracy",
            ],
            fold_rows,
        )
        for path in (daily_csv, fold_csv):
            files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }

        restored_mask = full_evaluation_mask & ~boundary_evaluation_mask
        dropped_mask = boundary_evaluation_mask & ~full_evaluation_mask
        legacy_metrics = probability_manifest.get("source_oos_metrics", {})
        manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "M0",
            "artifact": "OOS prediction metrics",
            "source_fingerprint": source_fingerprint,
            "probability_input": str(probability_dir.resolve()),
            "score_input": str(score_dir.resolve()),
            "full_supervision_input": str(
                (ready_dir / "_shared" / "supervision_full").resolve()
            ),
            "evaluation_policy": (
                "Use inference_mask AND full next-trading-day label availability; "
                "full supervision restores observable cross-year targets removed only "
                "for fold-boundary leakage control during training"
            ),
            "aggregation": "compute each metric per date, then take an equal-weight mean over valid dates",
            "rank_ic_definition": "Spearman correlation between ranking score and target_return; average ranks for ties",
            "ic_definition": "Pearson correlation between ranking score and target_return",
            "accuracy_definition": "mean(argmax probability class ID == five-class target) per date",
            "argmax_tie_rule": "lowest class ID, equivalent to numpy argmax first occurrence",
            "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "overall": overall,
            "folds": fold_summaries,
            "boundary_purge_reconciliation": {
                "legacy_boundary_evaluation_nodes": int(boundary_evaluation_mask.sum()),
                "full_evaluation_nodes": int(full_evaluation_mask.sum()),
                "restored_cross_year_nodes": int(restored_mask.sum()),
                "boundary_nodes_not_in_full": int(dropped_mask.sum()),
                "legacy_boundary_metrics": legacy_metrics,
            },
            "full_supervision_sources": source_records,
            "files": files,
            "checks": {
                "date_alignment": "passed",
                "permno_alignment": "passed",
                "inference_feature_mask_alignment": "passed",
                "score_probability_identity": "passed",
                "strictly_forward_target_dates": "passed",
                "target_class_domain": [0, 1, 2, 3, 4],
                "daily_equal_weight_aggregation": "passed",
            },
        }
        write_json(temporary / "prediction_metrics.json", manifest)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M0 OOS Rank IC, Pearson IC, and accuracy."
    )
    parser.add_argument("--probability-dir", type=Path, default=DEFAULT_PROBABILITY_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = evaluate_prediction_metrics(
        probability_dir=args.probability_dir.expanduser().resolve(),
        score_dir=args.score_dir.expanduser().resolve(),
        ready_dir=args.ready_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    overall = manifest["overall"]
    print(f"Rank IC: {overall['rank_ic']:.10f}", flush=True)
    print(f"IC: {overall['ic']:.10f}", flush=True)
    print(f"Accuracy: {overall['accuracy']:.10f}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
