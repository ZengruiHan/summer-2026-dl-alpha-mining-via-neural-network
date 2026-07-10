"""Publish already-generated M0 test probabilities to a stable results path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "results" / "models" / "M0"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "probabilities"
EXPORTED_ARRAYS = {
    "dates.npy": "dates.npy",
    "permno.npy": "permno.npy",
    "probabilities.npy": "probabilities.npy",
    "inference_mask.npy": "inference_mask.npy",
    "evaluation_mask.npy": "evaluation_mask.npy",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_probability_arrays(
    *,
    dates: np.ndarray,
    permno: np.ndarray,
    probabilities: np.ndarray,
    inference_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, Any]:
    if dates.ndim != 1:
        raise RuntimeError("dates must be one-dimensional")
    expected_node_shape = (len(dates), 500)
    if permno.shape != expected_node_shape:
        raise RuntimeError("PERMNO shape does not match [date, 500]")
    if inference_mask.shape != expected_node_shape:
        raise RuntimeError("inference_mask shape does not match [date, 500]")
    if evaluation_mask.shape != expected_node_shape:
        raise RuntimeError("evaluation_mask shape does not match [date, 500]")
    if probabilities.shape != (*expected_node_shape, 5):
        raise RuntimeError("Probability shape does not match [date, 500, 5]")
    date_numbers = dates.astype("datetime64[D]").astype(np.int64)
    if len(dates) > 1 and not np.all(np.diff(date_numbers) > 0):
        raise RuntimeError("Probability dates are not strictly increasing")
    if any(len(np.unique(row)) != 500 for row in permno):
        raise RuntimeError("Duplicate PERMNO within a probability date")
    if not np.all(evaluation_mask <= inference_mask):
        raise RuntimeError("evaluation_mask is not a subset of inference_mask")
    if not np.isfinite(probabilities[inference_mask]).all():
        raise RuntimeError("Inference-ready probability contains NaN/Inf")
    if not np.isnan(probabilities[~inference_mask]).all():
        raise RuntimeError("Non-inference probability must be NaN")
    if not np.allclose(
        probabilities[inference_mask].sum(axis=1), 1.0, atol=2e-6
    ):
        raise RuntimeError("Probability vector does not sum to one")
    return {
        "date_count": len(dates),
        "node_count": int(np.prod(expected_node_shape)),
        "inference_nodes": int(inference_mask.sum()),
        "evaluation_nodes": int(evaluation_mask.sum()),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "probability_shape": list(probabilities.shape),
    }


def validate_source_bundle(
    source_dir: Path,
    *,
    expected_hashes: dict[str, str] | None,
) -> dict[str, Any]:
    for source_name in EXPORTED_ARRAYS:
        source_path = source_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if expected_hashes is not None:
            expected_hash = expected_hashes.get(source_name)
            if expected_hash is None or sha256_file(source_path) != expected_hash:
                raise RuntimeError(f"Source probability hash failed: {source_path}")
    return validate_probability_arrays(
        dates=np.load(source_dir / "dates.npy", mmap_mode="r", allow_pickle=False),
        permno=np.load(source_dir / "permno.npy", mmap_mode="r", allow_pickle=False),
        probabilities=np.load(
            source_dir / "probabilities.npy", mmap_mode="r", allow_pickle=False
        ),
        inference_mask=np.load(
            source_dir / "inference_mask.npy", mmap_mode="r", allow_pickle=False
        ),
        evaluation_mask=np.load(
            source_dir / "evaluation_mask.npy", mmap_mode="r", allow_pickle=False
        ),
    )


def publish_bundle(
    source_dir: Path,
    output_dir: Path,
    *,
    expected_hashes: dict[str, str] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checks = validate_source_bundle(source_dir, expected_hashes=expected_hashes)
    files: dict[str, dict[str, Any]] = {}
    for source_name, output_name in EXPORTED_ARRAYS.items():
        source_path = source_dir / source_name
        output_path = output_dir / output_name
        method = hardlink_or_copy(source_path, output_path)
        source_hash = sha256_file(source_path)
        if sha256_file(output_path) != source_hash:
            raise RuntimeError(f"Published probability hash failed: {output_path}")
        files[output_name] = {
            "source_name": source_name,
            "sha256": source_hash,
            "publication_method": method,
            "bytes": output_path.stat().st_size,
        }
    manifest = {
        **metadata,
        **checks,
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "masked_probability_policy": "NaN where inference_mask is false",
        "files": files,
    }
    write_json(output_dir / "probability_manifest.json", manifest)
    return manifest


def export_test_probabilities(
    *,
    source_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_manifest_path = source_dir / "M0_manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("status") != "complete":
        raise RuntimeError("M0 source run is not complete")
    if source_manifest.get("completed_fold_count") != 17:
        raise RuntimeError("M0 source does not contain all 17 folds")
    if source_manifest.get("completed_fold_indices") != list(range(17)):
        raise RuntimeError("M0 source fold indices are incomplete or unordered")
    source_oos_dir = source_dir / "oos"
    source_oos_manifest_path = source_oos_dir / "oos_manifest.json"
    source_oos_manifest = load_json(source_oos_manifest_path)
    oos_checks = validate_source_bundle(source_oos_dir, expected_hashes=None)
    oos_arrays = {
        name: np.load(source_oos_dir / name, mmap_mode="r", allow_pickle=False)
        for name in EXPORTED_ARRAYS
    }
    oos_fold_index = np.load(
        source_oos_dir / "fold_index.npy", mmap_mode="r", allow_pickle=False
    )
    if oos_fold_index.shape != (oos_checks["date_count"],):
        raise RuntimeError("OOS fold_index shape does not match dates")

    fold_summaries: list[dict[str, Any]] = []
    offset = 0
    for fold_record in source_manifest["folds"]:
        fold_index = int(fold_record["fold_index"])
        test_year = int(fold_record["test_year"])
        if test_year != 2009 + fold_index:
            raise RuntimeError("Unexpected M0 fold/test-year mapping")
        source_prediction_dir = (
            source_dir / fold_record["fold_name"] / "predictions" / "test"
        )
        prediction_manifest_path = source_prediction_dir / "prediction_manifest.json"
        prediction_manifest_hash = sha256_file(prediction_manifest_path)
        if prediction_manifest_hash != fold_record["test"]["prediction_manifest_sha256"]:
            raise RuntimeError(f"Source test manifest hash failed for fold {fold_index}")
        prediction_manifest = load_json(prediction_manifest_path)
        if prediction_manifest.get("stage") != "refit_train_plus_validation_checkpoint":
            raise RuntimeError("Source probabilities are not from refit checkpoint")
        if not fold_record["test"].get("probabilities_generated_once"):
            raise RuntimeError("Source fold lacks one-time probability guarantee")
        fold_checks = validate_source_bundle(
            source_prediction_dir,
            expected_hashes=prediction_manifest["file_sha256"],
        )
        stop = offset + fold_checks["date_count"]
        for source_name in EXPORTED_ARRAYS:
            fold_values = np.load(
                source_prediction_dir / source_name,
                mmap_mode="r",
                allow_pickle=False,
            )
            oos_values = oos_arrays[source_name][offset:stop]
            if not np.array_equal(fold_values, oos_values, equal_nan=True):
                raise RuntimeError(
                    f"OOS array is not the exact fold concat: {source_name}, fold {fold_index}"
                )
        if not np.all(oos_fold_index[offset:stop] == fold_index):
            raise RuntimeError(f"OOS fold_index mismatch for fold {fold_index}")
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "test_year": test_year,
                "date_start": fold_checks["date_start"],
                "date_end": fold_checks["date_end"],
                "date_count": fold_checks["date_count"],
                "source_prediction_manifest_sha256": prediction_manifest_hash,
                "source_probability_sha256": prediction_manifest["file_sha256"][
                    "probabilities.npy"
                ],
            }
        )
        offset = stop
    if offset != oos_checks["date_count"]:
        raise RuntimeError("Fold test dates do not exhaust the OOS bundle")

    if output_dir.exists() and not overwrite:
        existing_manifest_path = output_dir / "probability_manifest.json"
        if not existing_manifest_path.exists():
            raise FileExistsError(
                f"Output exists without a manifest: {output_dir}; use --overwrite"
            )
        existing = load_json(existing_manifest_path)
        if existing.get("source_run_fingerprint") != source_manifest["run_fingerprint"]:
            raise FileExistsError(
                f"Output belongs to another M0 run: {output_dir}; use --overwrite"
            )
        for filename, record in existing["files"].items():
            output_path = output_dir / filename
            source_path = source_oos_dir / record["source_name"]
            if (
                not output_path.exists()
                or sha256_file(output_path) != record["sha256"]
                or sha256_file(source_path) != record["sha256"]
            ):
                raise RuntimeError(
                    f"Existing probability publication failed integrity: {output_path}"
                )
        return existing

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(output_dir.name + ".part")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        publication = publish_bundle(
            source_oos_dir,
            temporary,
            expected_hashes=None,
            metadata={
                "model": "M0",
                "scope": "canonical_concatenated_non_overlapping_oos_test",
                "source": str(source_oos_dir.resolve()),
                "source_stage": "concatenated_fold_test_probabilities",
                "test_inference_recomputed": False,
            },
        )
        fold_index_source = source_oos_dir / "fold_index.npy"
        fold_index_output = temporary / "fold_index.npy"
        method = hardlink_or_copy(fold_index_source, fold_index_output)
        fold_index_hash = sha256_file(fold_index_source)
        if sha256_file(fold_index_output) != fold_index_hash:
            raise RuntimeError("Published OOS fold_index hash failed")
        publication["files"]["fold_index.npy"] = {
            "source_name": "fold_index.npy",
            "sha256": fold_index_hash,
            "publication_method": method,
            "bytes": fold_index_output.stat().st_size,
        }
        root_manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "M0",
            "artifact": "canonical concatenated test-period predicted probabilities",
            "test_inference_recomputed": False,
            "publication_policy": (
                "Verified publication of the one-time probabilities generated "
                "after each fold's train+validation refit"
            ),
            "source": str(source_dir.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_oos_manifest_sha256": sha256_file(source_oos_manifest_path),
            "source_run_fingerprint": source_manifest["run_fingerprint"],
            "fold_count": len(fold_summaries),
            "fold_indices": list(range(17)),
            "test_years": list(range(2009, 2026)),
            "date_start": oos_checks["date_start"],
            "date_end": oos_checks["date_end"],
            "date_count": oos_checks["date_count"],
            "node_count": oos_checks["node_count"],
            "inference_nodes": oos_checks["inference_nodes"],
            "evaluation_nodes": oos_checks["evaluation_nodes"],
            "probability_shape": oos_checks["probability_shape"],
            "class_axis_signed_labels": [-2, -1, 0, 1, 2],
            "masked_probability_policy": "NaN where inference_mask is false",
            "files": publication["files"],
            "fold_sources": fold_summaries,
            "source_oos_metrics": source_oos_manifest["metrics"],
            "checks": {
                "source_refit_stage": "passed",
                "source_file_hashes": "passed",
                "oos_exact_fold_concatenation": "passed",
                "published_file_hashes": "passed",
                "strict_date_order": "passed",
                "daily_permno_uniqueness": "passed",
                "probability_normalization": "passed",
                "mask_contract": "passed",
            },
        }
        write_json(temporary / "probability_manifest.json", root_manifest)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish M0 test probabilities without rerunning test inference."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_test_probabilities(
        source_dir=args.source_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"Published folds: {manifest['fold_count']}", flush=True)
    print(f"OOS shape: {manifest['probability_shape']}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
