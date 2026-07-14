"""Convert test probabilities into ranking scores and daily long-short weights."""

from __future__ import annotations

import argparse
import math
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
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "results" / "probabilities"
DEFAULT_SCORE_DIR = REPOSITORY_ROOT / "results" / "ranking_scores"
DEFAULT_PORTFOLIO_DIR = REPOSITORY_ROOT / "results" / "portfolios"
CLASS_VALUES = np.arange(-2, 3, dtype=np.float32)
ALIGNMENT_FILES = (
    "dates.npy",
    "permno.npy",
    "fold_index.npy",
    "inference_mask.npy",
    "evaluation_mask.npy",
)


def probability_to_score(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 3 or probabilities.shape[-1] != 5:
        raise ValueError("probabilities must have shape [date, asset, 5]")
    return (probabilities @ CLASS_VALUES).astype(np.float32)


def construct_daily_portfolios(
    scores: np.ndarray,
    permno: np.ndarray,
    inference_mask: np.ndarray,
    *,
    tail_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Rank by (score, PERMNO) and form exact, disjoint equal-size tails.

    Ordinal rank one is the lowest score.  PERMNO ascending is the secondary
    key for exact score ties.  The leg size is floor(tail_fraction * N_t),
    which never allocates more than the requested tail fraction.
    """

    if not 0.0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be strictly between 0 and 0.5")
    if scores.shape != permno.shape or scores.shape != inference_mask.shape:
        raise ValueError("scores, permno, and inference_mask must align")
    if scores.ndim != 2:
        raise ValueError("daily score arrays must be two-dimensional")

    weights = np.zeros(scores.shape, dtype=np.float32)
    positions = np.zeros(scores.shape, dtype=np.int8)
    ordinal_rank = np.full(scores.shape, -1, dtype=np.int16)
    long_leg_size = np.zeros(scores.shape[0], dtype=np.int16)
    short_leg_size = np.zeros(scores.shape[0], dtype=np.int16)
    eligible_count = inference_mask.sum(axis=1).astype(np.int16)

    for date_index in range(scores.shape[0]):
        eligible = np.flatnonzero(inference_mask[date_index])
        if not np.isfinite(scores[date_index, eligible]).all():
            raise RuntimeError(f"Non-finite eligible score on date index {date_index}")
        if len(np.unique(permno[date_index])) != permno.shape[1]:
            raise RuntimeError(f"Duplicate daily PERMNO on date index {date_index}")
        leg_size = math.floor(tail_fraction * len(eligible))
        if leg_size < 1 or 2 * leg_size > len(eligible):
            raise RuntimeError(
                f"Insufficient eligible assets on date index {date_index}"
            )

        order_within_eligible = np.lexsort(
            (permno[date_index, eligible], scores[date_index, eligible])
        )
        ordered_nodes = eligible[order_within_eligible]
        ordinal_rank[date_index, ordered_nodes] = np.arange(
            1, len(eligible) + 1, dtype=np.int16
        )
        short_nodes = ordered_nodes[:leg_size]
        long_nodes = ordered_nodes[-leg_size:]
        positions[date_index, short_nodes] = -1
        positions[date_index, long_nodes] = 1
        weights[date_index, short_nodes] = np.float32(-0.5 / leg_size)
        weights[date_index, long_nodes] = np.float32(0.5 / leg_size)
        short_leg_size[date_index] = leg_size
        long_leg_size[date_index] = leg_size

    gross_exposure = np.abs(weights).sum(axis=1).astype(np.float32)
    net_exposure = weights.sum(axis=1).astype(np.float32)
    return {
        "weights": weights,
        "positions": positions,
        "ordinal_rank": ordinal_rank,
        "eligible_count": eligible_count,
        "long_leg_size": long_leg_size,
        "short_leg_size": short_leg_size,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
    }


def _save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _publish_alignment(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for filename in ALIGNMENT_FILES:
        source = input_dir / filename
        destination = output_dir / filename
        method = hardlink_or_copy(source, destination)
        digest = sha256_file(source)
        if sha256_file(destination) != digest:
            raise RuntimeError(f"Alignment publication hash failed: {destination}")
        records[filename] = {
            "source": str(source.resolve()),
            "sha256": digest,
            "publication_method": method,
            "bytes": destination.stat().st_size,
        }
    return records


def _verify_existing(
    output_dir: Path,
    manifest_name: str,
    *,
    input_manifest_hash: str,
) -> dict[str, Any] | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / manifest_name
    if not manifest_path.exists():
        raise RuntimeError(f"Existing output lacks manifest: {output_dir}")
    manifest = load_json(manifest_path)
    if manifest.get("input_probability_manifest_sha256") != input_manifest_hash:
        raise RuntimeError(
            f"Existing output belongs to another probability input: {output_dir}"
        )
    for filename, record in manifest["files"].items():
        path = output_dir / filename
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Existing derived artifact failed integrity: {path}")
    return manifest


def build_scores_and_portfolios(
    *,
    input_dir: Path,
    score_dir: Path,
    portfolio_dir: Path,
    tail_fraction: float = 0.2,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_manifest_path = input_dir / "probability_manifest.json"
    input_manifest = load_json(input_manifest_path)
    if input_manifest.get("status") != "complete" or not input_manifest.get("model"):
        raise RuntimeError(
            "Input probability publication is not a complete model artifact"
        )
    model_name = str(input_manifest["model"])
    if input_manifest.get("class_axis_signed_labels") != [-2, -1, 0, 1, 2]:
        raise RuntimeError("Probability class axis does not match the proposal")
    input_manifest_hash = sha256_file(input_manifest_path)

    if not overwrite:
        existing_scores = _verify_existing(
            score_dir,
            "score_manifest.json",
            input_manifest_hash=input_manifest_hash,
        )
        existing_portfolios = _verify_existing(
            portfolio_dir,
            "portfolio_manifest.json",
            input_manifest_hash=input_manifest_hash,
        )
        if existing_scores is not None or existing_portfolios is not None:
            if existing_scores is None or existing_portfolios is None:
                raise RuntimeError(
                    "Only one of the paired score/portfolio outputs exists"
                )
            return {"scores": existing_scores, "portfolios": existing_portfolios}

    probabilities = np.load(
        input_dir / "probabilities.npy", mmap_mode="r", allow_pickle=False
    )
    dates = np.load(input_dir / "dates.npy", mmap_mode="r", allow_pickle=False)
    permno = np.load(input_dir / "permno.npy", mmap_mode="r", allow_pickle=False)
    inference_mask = np.load(
        input_dir / "inference_mask.npy", mmap_mode="r", allow_pickle=False
    )
    evaluation_mask = np.load(
        input_dir / "evaluation_mask.npy", mmap_mode="r", allow_pickle=False
    )
    fold_index = np.load(
        input_dir / "fold_index.npy", mmap_mode="r", allow_pickle=False
    )
    expected_node_shape = (len(dates), 500)
    if probabilities.shape != (*expected_node_shape, 5):
        raise RuntimeError("Unexpected probability tensor shape")
    if any(
        array.shape != expected_node_shape
        for array in (permno, inference_mask, evaluation_mask)
    ):
        raise RuntimeError("Probability alignment array shape mismatch")
    if fold_index.shape != (len(dates),):
        raise RuntimeError("fold_index shape mismatch")
    if not np.all(evaluation_mask <= inference_mask):
        raise RuntimeError("evaluation_mask is not a subset of inference_mask")

    scores = probability_to_score(probabilities)
    if not np.isfinite(scores[inference_mask]).all():
        raise RuntimeError("Ranking score is not finite under inference_mask")
    if not np.isnan(scores[~inference_mask]).all():
        raise RuntimeError("Masked ranking scores must remain NaN")
    portfolio = construct_daily_portfolios(
        scores,
        permno,
        inference_mask,
        tail_fraction=tail_fraction,
    )

    weights = portfolio["weights"]
    positions = portfolio["positions"]
    if not np.allclose(portfolio["gross_exposure"], 1.0, atol=2e-6):
        raise RuntimeError("Daily portfolio gross exposure is not one")
    if not np.allclose(portfolio["net_exposure"], 0.0, atol=2e-7):
        raise RuntimeError("Daily portfolio is not dollar neutral")
    if np.any(weights[~inference_mask] != 0.0):
        raise RuntimeError("Ineligible asset received portfolio weight")
    if not set(np.unique(positions)).issubset({-1, 0, 1}):
        raise RuntimeError("Invalid portfolio position code")

    if overwrite:
        shutil.rmtree(score_dir, ignore_errors=True)
        shutil.rmtree(portfolio_dir, ignore_errors=True)
    score_temporary = score_dir.with_name(score_dir.name + ".part")
    portfolio_temporary = portfolio_dir.with_name(portfolio_dir.name + ".part")
    shutil.rmtree(score_temporary, ignore_errors=True)
    shutil.rmtree(portfolio_temporary, ignore_errors=True)
    score_temporary.mkdir(parents=True)
    portfolio_temporary.mkdir(parents=True)
    try:
        score_files = _publish_alignment(input_dir, score_temporary)
        _save_array(score_temporary / "ranking_scores.npy", scores)
        _save_array(score_temporary / "ordinal_rank.npy", portfolio["ordinal_rank"])
        for filename in ("ranking_scores.npy", "ordinal_rank.npy"):
            path = score_temporary / filename
            score_files[filename] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }
        score_manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "artifact": "cross-sectional ranking scores",
            "input": str(input_dir.resolve()),
            "input_probability_manifest_sha256": input_manifest_hash,
            "formula": "score[i,t] = sum_{c=-2}^{2} c * probability[i,t,c]",
            "class_axis_signed_labels": [-2, -1, 0, 1, 2],
            "ranking_rule": "ascending (score, permno); ordinal rank 1 is lowest; -1 is masked",
            "shape": list(scores.shape),
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "inference_nodes": int(inference_mask.sum()),
            "files": score_files,
            "checks": {
                "score_formula": "passed",
                "masked_scores_nan": "passed",
                "daily_permno_tie_break": "passed",
            },
        }
        write_json(score_temporary / "score_manifest.json", score_manifest)

        portfolio_files = _publish_alignment(input_dir, portfolio_temporary)
        derived_arrays = {
            "weights.npy": weights,
            "positions.npy": positions,
            "eligible_count.npy": portfolio["eligible_count"],
            "long_leg_size.npy": portfolio["long_leg_size"],
            "short_leg_size.npy": portfolio["short_leg_size"],
            "gross_exposure.npy": portfolio["gross_exposure"],
            "net_exposure.npy": portfolio["net_exposure"],
        }
        for filename, values in derived_arrays.items():
            path = portfolio_temporary / filename
            _save_array(path, values)
            portfolio_files[filename] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }
        portfolio_manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "artifact": "daily equal-weight dollar-neutral long-short portfolios",
            "input": str(input_dir.resolve()),
            "ranking_scores": str(score_dir.resolve()),
            "input_probability_manifest_sha256": input_manifest_hash,
            "tail_fraction": tail_fraction,
            "leg_size_rule": "floor(tail_fraction * inference-ready asset count)",
            "tie_break_rule": "ascending (score, permno); lowest tail is short and highest tail is long",
            "weight_rule": "long=+1/(2*|L_t|), short=-1/(2*|S_t|), otherwise=0",
            "universe_rule": "use inference_mask only; evaluation_mask is not used to construct holdings",
            "rebalance_frequency": "daily",
            "transaction_cost_policy": "not applied; kappa is not specified",
            "weight_shape": list(weights.shape),
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "eligible_count_min": int(portfolio["eligible_count"].min()),
            "eligible_count_max": int(portfolio["eligible_count"].max()),
            "long_leg_size_min": int(portfolio["long_leg_size"].min()),
            "long_leg_size_max": int(portfolio["long_leg_size"].max()),
            "short_leg_size_min": int(portfolio["short_leg_size"].min()),
            "short_leg_size_max": int(portfolio["short_leg_size"].max()),
            "files": portfolio_files,
            "checks": {
                "long_short_disjoint": "passed",
                "equal_leg_sizes": "passed",
                "long_weight_sum": "+0.5 daily",
                "short_weight_sum": "-0.5 daily",
                "gross_exposure": "1.0 daily",
                "net_exposure": "0.0 daily",
                "masked_asset_weight": "0.0",
                "date_and_permno_alignment": "passed",
            },
        }
        write_json(portfolio_temporary / "portfolio_manifest.json", portfolio_manifest)
        score_temporary.replace(score_dir)
        portfolio_temporary.replace(portfolio_dir)
    except BaseException:
        shutil.rmtree(score_temporary, ignore_errors=True)
        shutil.rmtree(portfolio_temporary, ignore_errors=True)
        if score_dir.exists() and not portfolio_dir.exists():
            shutil.rmtree(score_dir, ignore_errors=True)
        raise
    return {"scores": score_manifest, "portfolios": portfolio_manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build model ranking scores and daily long-short portfolios."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--portfolio-dir", type=Path, default=DEFAULT_PORTFOLIO_DIR)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_scores_and_portfolios(
        input_dir=args.input_dir.expanduser().resolve(),
        score_dir=args.score_dir.expanduser().resolve(),
        portfolio_dir=args.portfolio_dir.expanduser().resolve(),
        tail_fraction=args.tail_fraction,
        overwrite=args.overwrite,
    )
    print(f"Score shape: {result['scores']['shape']}", flush=True)
    print(f"Weight shape: {result['portfolios']['weight_shape']}", flush=True)
    print(f"Scores: {args.score_dir.expanduser().resolve()}", flush=True)
    print(f"Portfolios: {args.portfolio_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
