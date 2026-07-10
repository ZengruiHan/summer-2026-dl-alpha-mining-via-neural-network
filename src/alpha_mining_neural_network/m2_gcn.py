"""Train M2-C, the proposal's same-sector graph convolutional network.

M2-C uses the same walk-forward chronology and five-class output contract as
M0.  Each trading date is a 500-node graph, and nodes sharing ``sector_index``
are connected.  The implementation delegates GCN math to the NumPy-only
``gcn`` module and keeps test data unopened until validation selection and
train-plus-validation refitting are complete.
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from alpha_mining_neural_network.gcn import (
    GCNParameters,
    PreparedGraphSplit,
    fit_candidate,
    parameter_count,
    predict_probabilities_with_metrics,
    refit_model,
    save_model,
    strict_loss_accuracy,
    with_sector_graphs,
)
from alpha_mining_neural_network.m0 import (
    CLASS_VALUES,
    LABEL_SENTINEL,
    build_oos_bundle,
    config_fingerprint,
    load_json,
    load_prepared_split,
    prediction_metrics,
    sha256_file,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "M2-C"
MODEL_FAMILY = "gcn_multiclass"
GRAPH_RELATION = "sector"
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "models" / MODEL_NAME
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "training" / f"{MODEL_NAME}.json"


def validate_config(config: dict[str, Any]) -> None:
    """Validate the deterministic M2-C training contract."""

    if config.get("model") != MODEL_NAME:
        raise ValueError(f"Config must set model='{MODEL_NAME}'")
    if config.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"Config must set model_family='{MODEL_FAMILY}'")
    if config.get("graph_relation") != GRAPH_RELATION:
        raise ValueError("M2-C graph_relation must be 'sector'")
    if config.get("optimizer") != "adam":
        raise ValueError("Only Adam is supported")
    if config.get("selection_metric") != "validation_daily_mean_cross_entropy":
        raise ValueError("Unsupported selection metric")
    for key in (
        "numeric_feature_count",
        "sector_category_count",
        "class_count",
        "max_epochs",
        "patience",
        "batch_dates",
    ):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["class_count"]) != 5:
        raise ValueError("M2-C requires five output classes")
    if float(config.get("min_delta", -1.0)) < 0:
        raise ValueError("min_delta must be non-negative")
    for key in ("adam_beta1", "adam_beta2"):
        value = float(config.get(key, 0.0))
        if not 0.0 < value < 1.0:
            raise ValueError(f"{key} must be in (0, 1)")
    if float(config.get("adam_epsilon", 0.0)) <= 0:
        raise ValueError("adam_epsilon must be positive")
    if (
        config.get("gradient_clip_norm") is not None
        and float(config["gradient_clip_norm"]) <= 0
    ):
        raise ValueError("gradient_clip_norm must be positive")

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("At least one hyperparameter candidate is required")
    identifiers: set[str] = set()
    for candidate in candidates:
        identifier = str(candidate.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("Candidate IDs must be non-empty and unique")
        identifiers.add(identifier)
        if int(candidate.get("hidden_dim", 0)) <= 0:
            raise ValueError(f"Candidate {identifier} has invalid hidden_dim")
        if float(candidate.get("learning_rate", 0.0)) <= 0:
            raise ValueError(f"Candidate {identifier} has invalid learning_rate")
        if float(candidate.get("l2", -1.0)) < 0:
            raise ValueError(f"Candidate {identifier} has invalid l2")


def load_sector_split(
    split_dir: Path,
    *,
    numeric_feature_count: int,
    sector_category_count: int,
    include_evaluation_metadata: bool,
) -> PreparedGraphSplit:
    """Load tensors and derive the fixed same-sector graph for every date."""

    data = load_prepared_split(
        split_dir,
        numeric_feature_count=numeric_feature_count,
        sector_category_count=sector_category_count,
        include_evaluation_metadata=include_evaluation_metadata,
    )
    return with_sector_graphs(data)


def validate_split_contract(
    split: PreparedGraphSplit,
    *,
    expected_name: str,
    expected_years: list[int],
) -> None:
    """Prevent a mislabeled directory from crossing chronology boundaries."""

    if split.data.name != expected_name:
        raise RuntimeError(
            f"Expected split '{expected_name}', found '{split.data.name}'"
        )
    if split.data.years != expected_years:
        raise RuntimeError(
            f"{expected_name} years {split.data.years} do not match fold years "
            f"{expected_years}"
        )


def write_prediction_bundle(
    output_dir: Path,
    *,
    split: PreparedGraphSplit,
    probabilities: np.ndarray,
    parameters: GCNParameters,
    exact_metrics: tuple[float, float, int] | None,
    stage: str,
    fold_index: int,
) -> dict[str, Any]:
    """Write M0-shaped probabilities, masks, labels, and ranking scores."""

    data = split.data
    if (
        data.permno is None
        or data.y_signed is None
        or data.target_return is None
        or data.target_date is None
        or data.inference_mask is None
    ):
        raise RuntimeError("Evaluation metadata was not loaded")
    if probabilities.shape != (*data.y.shape, 5):
        raise ValueError("Probability tensor does not align with the split")
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = prediction_metrics(
        y=data.y,
        target_return=data.target_return,
        evaluation_mask=data.loss_mask,
        probabilities=probabilities,
    )
    exact_loss, exact_accuracy, exact_nodes = (
        strict_loss_accuracy(parameters, split)
        if exact_metrics is None
        else exact_metrics
    )
    if exact_nodes != metrics["model_ready_nodes"]:
        raise RuntimeError("Exact and probability metric node counts disagree")
    metrics["daily_mean_cross_entropy"] = exact_loss
    metrics["daily_mean_accuracy"] = exact_accuracy

    stored_probabilities = probabilities.astype(np.float32, copy=True)
    stored_probabilities[~data.inference_mask] = np.nan
    scores = (stored_probabilities @ CLASS_VALUES).astype(np.float32)
    predicted_class = probabilities.argmax(axis=2).astype(np.int8)
    predicted_signed = (predicted_class - 2).astype(np.int8)
    predicted_class[~data.inference_mask] = np.int8(LABEL_SENTINEL)
    predicted_signed[~data.inference_mask] = np.int8(-128)
    arrays = {
        "dates": data.dates,
        "permno": data.permno,
        "probabilities": stored_probabilities,
        "scores": scores,
        "predicted_class": predicted_class,
        "predicted_signed": predicted_signed,
        "y": data.y,
        "y_signed": data.y_signed,
        "target_return": data.target_return,
        "target_date": data.target_date,
        "inference_mask": data.inference_mask,
        "evaluation_mask": data.loss_mask,
    }
    for name, values in arrays.items():
        with (output_dir / f"{name}.npy").open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    file_sha256 = {
        f"{name}.npy": sha256_file(output_dir / f"{name}.npy") for name in arrays
    }
    manifest = {
        "stage": stage,
        "fold_index": fold_index,
        "split": data.name,
        "years": data.years,
        "date_start": str(data.dates[0]),
        "date_end": str(data.dates[-1]),
        "graph_relation": split.relation,
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "probability_shape": list(probabilities.shape),
        "masked_prediction_policy": (
            "probabilities/scores are NaN and classes use sentinels where "
            "inference_mask is false"
        ),
        "score_definition": "sum_{c=-2}^{2} c * p(c)",
        "evaluation_mask_definition": (
            "loss_mask = label_mask AND complete_numeric_mask"
        ),
        "inference_mask_definition": "complete_numeric_mask",
        "file_sha256": file_sha256,
        "metrics": metrics,
    }
    write_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def fit_fold(
    fold_dir: Path,
    output_dir: Path,
    *,
    config: dict[str, Any],
    config_hash: str,
    run_fingerprint: str,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Select, refit, and evaluate M2-C on one chronological fold."""

    fold_manifest = load_json(fold_dir / "ready_fold_manifest.json")
    fold = fold_manifest["fold"]
    if str(fold.get("name")) != fold_dir.name:
        raise RuntimeError(
            f"Fold manifest name {fold.get('name')!r} does not match directory "
            f"{fold_dir.name!r}"
        )
    fold_index = int(fold["index"])
    fold_seed = int(config["seed"]) + fold_index * 10_000
    numeric_count = int(config["numeric_feature_count"])
    sector_count = int(config["sector_category_count"])
    started = time.perf_counter()

    train = load_sector_split(
        fold_dir / "train",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=False,
    )
    validate_split_contract(
        train,
        expected_name="train",
        expected_years=list(
            range(
                int(fold["train_start_year"]),
                int(fold["train_end_year"]) + 1,
            )
        ),
    )
    validation = load_sector_split(
        fold_dir / "validation",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=True,
    )
    validate_split_contract(
        validation,
        expected_name="validation",
        expected_years=[int(fold["validation_year"])],
    )

    candidate_records: list[dict[str, Any]] = []
    candidate_models: dict[str, GCNParameters] = {}
    for candidate in config["candidates"]:
        model, record = fit_candidate(
            train,
            validation,
            config=config,
            candidate=candidate,
            seed=fold_seed,
        )
        candidate_models[str(candidate["id"])] = model
        candidate_records.append(record)
    selected_record = min(
        candidate_records,
        key=lambda record: (
            record["best_validation_daily_mean_cross_entropy"],
            record["candidate_id"],
        ),
    )
    selected_candidate = next(
        candidate
        for candidate in config["candidates"]
        if str(candidate["id"]) == selected_record["candidate_id"]
    )
    selection_model = candidate_models[selected_record["candidate_id"]]
    selection_checkpoint = output_dir / "selection_model.npz"
    save_model(selection_checkpoint, selection_model, metadata=checkpoint_metadata)
    validation_started = time.perf_counter()
    validation_probabilities, validation_exact_metrics = (
        predict_probabilities_with_metrics(selection_model, validation)
    )
    validation_inference_seconds = time.perf_counter() - validation_started
    validation_prediction_manifest = write_prediction_bundle(
        output_dir / "predictions" / "validation",
        split=validation,
        probabilities=validation_probabilities,
        parameters=selection_model,
        exact_metrics=validation_exact_metrics,
        stage="train_only_selected_checkpoint",
        fold_index=fold_index,
    )

    del train, candidate_models, selection_model, validation_probabilities

    refit = load_sector_split(
        fold_dir / "refit",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=False,
    )
    validate_split_contract(
        refit,
        expected_name="refit",
        expected_years=list(
            range(
                int(fold["train_start_year"]),
                int(fold["validation_year"]) + 1,
            )
        ),
    )
    final_model, refit_record = refit_model(
        refit,
        config=config,
        candidate=selected_candidate,
        epochs=int(selected_record["best_epoch"]),
        seed=fold_seed + 9_000,
    )
    final_checkpoint = output_dir / "model.npz"
    save_model(final_checkpoint, final_model, metadata=checkpoint_metadata)
    del refit

    # The test split is intentionally opened only after selection and refit.
    test = load_sector_split(
        fold_dir / "test",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=True,
    )
    validate_split_contract(
        test,
        expected_name="test",
        expected_years=[int(fold["test_year"])],
    )
    test_started = time.perf_counter()
    test_probabilities, test_exact_metrics = predict_probabilities_with_metrics(
        final_model, test
    )
    test_inference_seconds = time.perf_counter() - test_started
    test_prediction_manifest = write_prediction_bundle(
        output_dir / "predictions" / "test",
        split=test,
        probabilities=test_probabilities,
        parameters=final_model,
        exact_metrics=test_exact_metrics,
        stage="refit_train_plus_validation_checkpoint",
        fold_index=fold_index,
    )

    record = {
        "status": "complete",
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "fold_index": fold_index,
        "fold_name": fold["name"],
        "config_fingerprint": config_hash,
        "run_fingerprint": run_fingerprint,
        "input_fold": str(fold_dir.resolve()),
        "train_years": list(
            range(
                int(fold["train_start_year"]),
                int(fold["train_end_year"]) + 1,
            )
        ),
        "validation_year": int(fold["validation_year"]),
        "refit_years": list(
            range(
                int(fold["train_start_year"]),
                int(fold["validation_year"]) + 1,
            )
        ),
        "test_year": int(fold["test_year"]),
        "feature_contract": {
            "numeric_features": numeric_count,
            "numeric_feature_names": checkpoint_metadata["numeric_feature_names"],
            "sector_categories": sector_count,
            "sector_implementation": (
                "first-layer lookup exactly equivalent to sector one-hot"
            ),
            "graph_used": True,
            "graph_relation": GRAPH_RELATION,
            "graph_definition": (
                "same sector_index; one model-owned self-loop per node"
            ),
            "normalization": (
                "absolute in/out degree; reduces to symmetric normalization"
            ),
            "known_structural_limitation": (
                "a complete same-sector clique averages member features exactly, so "
                "members of the same sector receive identical logits"
            ),
        },
        "objective": (
            "daily-equal mean categorical cross-entropy; node-equal within date"
        ),
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "selection": {
            "metric": config["selection_metric"],
            "selected_candidate": selected_record["candidate_id"],
            "selected_hidden_dim": selected_record["hidden_dim"],
            "selected_learning_rate": selected_record["learning_rate"],
            "selected_l2": selected_record["l2"],
            "selected_epoch": selected_record["best_epoch"],
            "candidate_records": candidate_records,
            "validation_metrics": validation_prediction_manifest["metrics"],
            "validation_inference_seconds": validation_inference_seconds,
            "selection_checkpoint": "selection_model.npz",
            "selection_checkpoint_sha256": sha256_file(selection_checkpoint),
            "validation_prediction_manifest_sha256": sha256_file(
                output_dir / "predictions" / "validation" / "prediction_manifest.json"
            ),
        },
        "refit": refit_record,
        "test": {
            "metrics": test_prediction_manifest["metrics"],
            "inference_seconds": test_inference_seconds,
            "probabilities_generated_once": True,
            "prediction_manifest_sha256": sha256_file(
                output_dir / "predictions" / "test" / "prediction_manifest.json"
            ),
        },
        "checkpoint": {
            "path": "model.npz",
            "sha256": sha256_file(final_checkpoint),
            "parameter_count": parameter_count(final_model),
            "device": "cpu",
            "gpu_memory_bytes": 0,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "fold_manifest.json", record)
    return record


def _verify_prediction_bundle(
    prediction_dir: Path, expected_manifest_sha256: str
) -> None:
    manifest_path = prediction_dir / "prediction_manifest.json"
    if (
        not manifest_path.exists()
        or sha256_file(manifest_path) != expected_manifest_sha256
    ):
        raise RuntimeError(
            f"Prediction manifest failed integrity check: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    for filename, expected_hash in manifest["file_sha256"].items():
        path = prediction_dir / filename
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Prediction array failed integrity check: {path}")


def _complete_fold_matches(path: Path, run_fingerprint: str) -> bool:
    manifest_path = path / "fold_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        return False
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(
            "Existing fold uses a different config, input, or implementation: "
            f"{path}. Use --overwrite."
        )
    checkpoint = path / str(manifest["checkpoint"]["path"])
    if (
        not checkpoint.exists()
        or sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]
    ):
        raise RuntimeError(f"Existing checkpoint failed integrity check: {checkpoint}")
    selection_checkpoint = path / manifest["selection"]["selection_checkpoint"]
    if (
        not selection_checkpoint.exists()
        or sha256_file(selection_checkpoint)
        != manifest["selection"]["selection_checkpoint_sha256"]
    ):
        raise RuntimeError(
            f"Selection checkpoint failed integrity check: {selection_checkpoint}"
        )
    _verify_prediction_bundle(
        path / "predictions" / "validation",
        manifest["selection"]["validation_prediction_manifest_sha256"],
    )
    _verify_prediction_bundle(
        path / "predictions" / "test",
        manifest["test"]["prediction_manifest_sha256"],
    )
    return True


def train_all_folds(
    *,
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    fold_indices: set[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run M2-C through all requested walk-forward folds."""

    config = load_json(config_path)
    validate_config(config)
    config_hash = config_fingerprint(config)
    input_manifest_path = input_dir / "ready_for_use_manifest.json"
    feature_spec_path = input_dir / "feature_spec.json"
    input_manifest = load_json(input_manifest_path)
    feature_spec = load_json(feature_spec_path)
    if int(input_manifest["numeric_feature_count"]) != int(
        config["numeric_feature_count"]
    ):
        raise RuntimeError("Config numeric feature count does not match ready data")
    if int(input_manifest["sector_category_count"]) != int(
        config["sector_category_count"]
    ):
        raise RuntimeError("Config sector category count does not match ready data")
    numeric_names = [str(name) for name in feature_spec["numeric_columns"]]
    if len(numeric_names) != int(config["numeric_feature_count"]):
        raise RuntimeError("feature_spec numeric columns do not match the config")
    expected_fixed_dimension = int(config["numeric_feature_count"]) + int(
        config["sector_category_count"]
    )
    if (
        int(feature_spec["combined_fixed_feature_dimension"])
        != expected_fixed_dimension
    ):
        raise RuntimeError("feature_spec fixed feature dimension does not match M2-C")

    input_manifest_hash = sha256_file(input_manifest_path)
    feature_spec_hash = sha256_file(feature_spec_path)
    workflow_hash = sha256_file(Path(__file__))
    core_path = Path(__file__).with_name("gcn.py")
    core_hash = sha256_file(core_path)
    shared_m0_path = Path(__file__).with_name("m0.py")
    shared_m0_hash = sha256_file(shared_m0_path)
    proposal_path = REPOSITORY_ROOT / "documents" / "proposal" / "proposal.pdf"
    proposal_hash = sha256_file(proposal_path)
    run_inputs = {
        "config_sha256": config_hash,
        "ready_for_use_manifest_sha256": input_manifest_hash,
        "feature_spec_sha256": feature_spec_hash,
        "workflow_implementation_sha256": workflow_hash,
        "gcn_core_implementation_sha256": core_hash,
        "shared_m0_implementation_sha256": shared_m0_hash,
        "numpy_version": np.__version__,
        "proposal_sha256": proposal_hash,
    }
    run_fingerprint = config_fingerprint(run_inputs)
    checkpoint_metadata: dict[str, Any] = {
        "model_family": MODEL_FAMILY,
        "graph_relation": GRAPH_RELATION,
        "numeric_feature_names": numeric_names,
        "feature_spec_sha256": feature_spec_hash,
        "ready_for_use_manifest_sha256": input_manifest_hash,
        "workflow_implementation_sha256": workflow_hash,
        "gcn_core_implementation_sha256": core_hash,
        "shared_m0_implementation_sha256": shared_m0_hash,
        "run_fingerprint": run_fingerprint,
        "parameter_dtype": "float32",
        "numpy_version": np.__version__,
    }

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = output_dir / "config.json"
    if existing_config_path.exists() and not overwrite:
        if config_fingerprint(load_json(existing_config_path)) != config_hash:
            raise RuntimeError(
                f"Existing {MODEL_NAME} output uses a different config: "
                f"{output_dir}. Use --overwrite."
            )
    existing_run_manifest = output_dir / f"{MODEL_NAME}_manifest.json"
    if existing_run_manifest.exists() and not overwrite:
        previous = load_json(existing_run_manifest)
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Existing {MODEL_NAME} output binds to different data or code: "
                f"{output_dir}. Use --overwrite."
            )
    write_json(output_dir / "config.json", config)

    fold_dirs = sorted(path for path in input_dir.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != int(input_manifest["fold_count"]):
        raise RuntimeError("Ready fold directory count does not match manifest")
    selected_fold_dirs: list[Path] = []
    for fold_dir in fold_dirs:
        index = int(load_json(fold_dir / "ready_fold_manifest.json")["fold"]["index"])
        if fold_indices is None or index in fold_indices:
            selected_fold_dirs.append(fold_dir)
    if not selected_fold_dirs:
        raise ValueError("No folds selected")
    if fold_indices is not None:
        found = {
            int(load_json(path / "ready_fold_manifest.json")["fold"]["index"])
            for path in selected_fold_dirs
        }
        if found != fold_indices:
            raise ValueError(
                f"Requested folds not found: {sorted(fold_indices - found)}"
            )

    run_started = time.perf_counter()
    for fold_dir in selected_fold_dirs:
        final_fold = output_dir / fold_dir.name
        if _complete_fold_matches(final_fold, run_fingerprint):
            print(f"[{fold_dir.name}] already complete; reusing", flush=True)
            continue
        if final_fold.exists():
            raise RuntimeError(
                f"Incomplete output exists: {final_fold}. Use --overwrite."
            )
        temporary_fold = output_dir / f"{fold_dir.name}.part"
        shutil.rmtree(temporary_fold, ignore_errors=True)
        temporary_fold.mkdir(parents=True)
        print(f"[{fold_dir.name}] fitting {MODEL_NAME}", flush=True)
        try:
            record = fit_fold(
                fold_dir,
                temporary_fold,
                config=config,
                config_hash=config_hash,
                run_fingerprint=run_fingerprint,
                checkpoint_metadata=checkpoint_metadata,
            )
            temporary_fold.replace(final_fold)
        except BaseException:
            shutil.rmtree(temporary_fold, ignore_errors=True)
            raise
        print(
            f"[{fold_dir.name}] selected={record['selection']['selected_candidate']} "
            f"epoch={record['selection']['selected_epoch']} "
            f"test_ce={record['test']['metrics']['daily_mean_cross_entropy']:.6f}",
            flush=True,
        )

    completed_records: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        result_dir = output_dir / fold_dir.name
        if result_dir.exists() and _complete_fold_matches(result_dir, run_fingerprint):
            completed_records.append(load_json(result_dir / "fold_manifest.json"))
    completed_records.sort(key=lambda item: item["fold_index"])
    oos_manifest = build_oos_bundle(output_dir, completed_records)
    expected_fold_count = int(input_manifest["fold_count"])
    manifest = {
        "status": (
            "complete" if len(completed_records) == expected_fold_count else "partial"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "proposal_contract": {
            "input": "standardized 0-cochain",
            "graph_used": True,
            "graph_relation": GRAPH_RELATION,
            "objective": "five-class daily-equal categorical cross-entropy",
            "chronology": (
                "8 train years / 1 validation year / refit on 9 years / " "1 test year"
            ),
            "selection": (
                "validation-only hyperparameter and early-stopping epoch selection"
            ),
            "test_probability_policy": ("generated once after refit for each fold"),
        },
        "implementation_decisions_not_fixed_by_proposal": {
            "optimizer": config["optimizer"],
            "activation": "ReLU",
            "normalization": "absolute in/out degree",
            "sector_representation": (
                "sparse first-layer lookup equivalent to one-hot"
            ),
            "hidden_layers": 1,
            "known_sector_clique_limitation": (
                "nodes sharing a sector are exactly averaged and therefore tie; "
                "this is retained for a faithful M2-C baseline"
            ),
        },
        "config": str(config_path.resolve()),
        "config_fingerprint": config_hash,
        "run_fingerprint": run_fingerprint,
        "run_inputs": run_inputs,
        "input": str(input_dir.resolve()),
        "input_manifest_sha256": input_manifest_hash,
        "feature_spec_sha256": feature_spec_hash,
        "workflow_implementation_sha256": workflow_hash,
        "gcn_core_implementation_sha256": core_hash,
        "shared_m0_implementation_sha256": shared_m0_hash,
        "proposal_sha256": proposal_hash,
        "expected_fold_count": expected_fold_count,
        "completed_fold_count": len(completed_records),
        "completed_fold_indices": [
            int(record["fold_index"]) for record in completed_records
        ],
        "parameter_count_by_fold": {
            str(record["fold_index"]): int(record["checkpoint"]["parameter_count"])
            for record in completed_records
        },
        "folds": completed_records,
        "oos": oos_manifest,
        "run_wall_seconds": time.perf_counter() - run_started,
    }
    write_json(output_dir / f"{MODEL_NAME}_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit M2-C Sector-GCN through chronological folds and save predictions."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fold",
        dest="folds",
        type=int,
        action="append",
        help="Fit only this zero-based fold index; repeat to select several.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Delete the {MODEL_NAME} output directory before fitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = train_all_folds(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        fold_indices=set(args.folds) if args.folds else None,
        overwrite=args.overwrite,
    )
    print(
        f"{MODEL_NAME} folds: "
        f"{manifest['completed_fold_count']}/{manifest['expected_fold_count']}",
        flush=True,
    )
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
