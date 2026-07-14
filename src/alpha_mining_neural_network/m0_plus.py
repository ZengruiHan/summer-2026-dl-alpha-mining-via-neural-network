"""Train M0-Plus, an XGBoost five-class return-ranking model.

M0-Plus keeps the M0 feature and chronology contracts while replacing the
linear softmax model with gradient-boosted decision trees. Numeric zeros are
stored explicitly and the 690-category sector block is represented sparsely,
so a training fold does not require a dense [rows, 698] matrix.
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from alpha_mining_neural_network.m0 import (
    CLASS_VALUES,
    LABEL_SENTINEL,
    PreparedSplit,
    build_oos_bundle,
    config_fingerprint,
    load_json,
    load_prepared_split,
    prediction_metrics,
    sha256_file,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "M0-Plus"
MODEL_FAMILY = "xgboost_multiclass"
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "models" / MODEL_NAME
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "training" / f"{MODEL_NAME}.json"


def require_xgboost() -> Any:
    """Import XGBoost lazily so data-contract tests do not require it."""

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "M0-Plus requires XGBoost. Install it with "
            "`python -m pip install xgboost`."
        ) from exc
    return xgb


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model") != MODEL_NAME:
        raise ValueError(f"Config must set model='{MODEL_NAME}'")
    if config.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"Config must set model_family='{MODEL_FAMILY}'")
    if config.get("selection_metric") != "validation_daily_mean_cross_entropy":
        raise ValueError("Unsupported selection metric")
    for key in (
        "numeric_feature_count",
        "sector_category_count",
        "class_count",
        "num_boost_round",
        "early_stopping_rounds",
        "max_bin",
    ):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["class_count"]) != 5:
        raise ValueError("M0-Plus requires five output classes")
    if config.get("tree_method") not in {"hist", "approx"}:
        raise ValueError("tree_method must be 'hist' or 'approx'")

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("At least one hyperparameter candidate is required")
    identifiers: set[str] = set()
    for candidate in candidates:
        identifier = str(candidate.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("Candidate IDs must be non-empty and unique")
        identifiers.add(identifier)
        if float(candidate.get("eta", 0.0)) <= 0:
            raise ValueError(f"Candidate {identifier} has invalid eta")
        if int(candidate.get("max_depth", 0)) <= 0:
            raise ValueError(f"Candidate {identifier} has invalid max_depth")
        if float(candidate.get("min_child_weight", -1.0)) < 0:
            raise ValueError(f"Candidate {identifier} has invalid min_child_weight")
        for key in ("subsample", "colsample_bytree"):
            value = float(candidate.get(key, 0.0))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"Candidate {identifier} has invalid {key}")
        for key in ("reg_lambda", "reg_alpha", "gamma"):
            if float(candidate.get(key, -1.0)) < 0:
                raise ValueError(f"Candidate {identifier} has invalid {key}")


def build_sparse_features(
    split: PreparedSplit,
    mask: np.ndarray,
    *,
    sector_category_count: int,
) -> Any:
    """Build CSR numeric plus one-hot sector features for selected nodes.

    Every numeric value is explicitly stored, including exact zeros. Inactive
    sector indicators remain structurally absent, which is the standard sparse
    one-hot representation used by tree boosters.
    """

    try:
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "M0-Plus requires SciPy. Install it with `python -m pip install scipy`."
        ) from exc

    if mask.shape != split.y.shape or mask.dtype != np.bool_:
        raise ValueError("mask must be boolean and align with the split node axes")
    numeric = split.x_numeric[mask].astype(np.float32, copy=False)
    sector = split.sector_index[mask].astype(np.int32, copy=False)
    if numeric.ndim != 2:
        raise RuntimeError("Selected numeric features must be two-dimensional")
    if not np.isfinite(numeric).all():
        raise RuntimeError("Selected numeric features contain NaN or Inf")
    if len(sector) and (sector.min() < 0 or sector.max() >= sector_category_count):
        raise RuntimeError("Selected sector index is outside the configured vocabulary")

    row_count, numeric_count = numeric.shape
    if row_count == 0:
        raise ValueError("mask contains no selected feature rows")
    entries_per_row = numeric_count + 1
    data = np.empty((row_count, entries_per_row), dtype=np.float32)
    data[:, :numeric_count] = numeric
    data[:, numeric_count] = 1.0
    indices = np.empty((row_count, entries_per_row), dtype=np.int32)
    indices[:, :numeric_count] = np.arange(numeric_count, dtype=np.int32)
    indices[:, numeric_count] = numeric_count + sector
    indptr = np.arange(
        0,
        row_count * entries_per_row + 1,
        entries_per_row,
        dtype=np.int64,
    )
    return csr_matrix(
        (data.reshape(-1), indices.reshape(-1), indptr),
        shape=(row_count, numeric_count + sector_category_count),
        dtype=np.float32,
    )


def daily_equal_weights(mask: np.ndarray) -> np.ndarray:
    """Give every active date equal total weight and keep mean weight at one."""

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    counts = mask.sum(axis=1).astype(np.int64)
    active = counts > 0
    if not active.any():
        raise ValueError("mask contains no selected nodes")
    total_nodes = int(counts.sum())
    total_weight_per_date = total_nodes / int(active.sum())
    per_node = total_weight_per_date / counts[active]
    weights = np.repeat(per_node, counts[active]).astype(np.float32)
    if len(weights) != total_nodes or not np.isfinite(weights).all():
        raise RuntimeError("Failed to construct finite date-equal weights")
    return weights


def build_dmatrix(
    xgb: Any,
    split: PreparedSplit,
    mask: np.ndarray,
    *,
    sector_category_count: int,
    supervised: bool,
) -> Any:
    matrix = build_sparse_features(
        split, mask, sector_category_count=sector_category_count
    )
    kwargs: dict[str, Any] = {"missing": np.nan}
    if supervised:
        labels = split.y[mask].astype(np.int32, copy=False)
        if not set(np.unique(labels).tolist()).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError("Supervised XGBoost labels must be in [0, 4]")
        kwargs["label"] = labels
        kwargs["weight"] = daily_equal_weights(mask)
    return xgb.DMatrix(matrix, **kwargs)


def candidate_params(
    config: dict[str, Any], candidate: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": int(config["class_count"]),
        "eval_metric": "mlogloss",
        "tree_method": str(config["tree_method"]),
        "max_bin": int(config["max_bin"]),
        "nthread": int(config.get("nthread", 0)),
        "seed": seed,
        "verbosity": 0,
    }
    if config.get("device"):
        params["device"] = str(config["device"])
    for key in (
        "eta",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    ):
        params[key] = candidate[key]
    return params


def fit_candidates(
    xgb: Any,
    dtrain: Any,
    dvalidation: Any,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    best_booster: Any | None = None
    best_record: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        started = time.perf_counter()
        evaluations: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params=candidate_params(config, candidate, seed=seed),
            dtrain=dtrain,
            num_boost_round=int(config["num_boost_round"]),
            evals=[(dvalidation, "validation")],
            evals_result=evaluations,
            early_stopping_rounds=int(config["early_stopping_rounds"]),
            verbose_eval=False,
        )
        history = [float(value) for value in evaluations["validation"]["mlogloss"]]
        best_iteration = int(getattr(booster, "best_iteration", len(history) - 1))
        best_score = float(getattr(booster, "best_score", history[best_iteration]))
        record = {
            "candidate_id": str(candidate["id"]),
            "params": candidate_params(config, candidate, seed=seed),
            "best_iteration": best_iteration,
            "best_num_boost_round": best_iteration + 1,
            "best_validation_daily_mean_cross_entropy": best_score,
            "rounds_run": len(history),
            "stopped_early": len(history) < int(config["num_boost_round"]),
            "training_seconds": time.perf_counter() - started,
            "validation_mlogloss_history": history,
        }
        records.append(record)
        key = (best_score, str(candidate["id"]))
        current_key = (
            (
                float(best_record["best_validation_daily_mean_cross_entropy"]),
                str(best_record["candidate_id"]),
            )
            if best_record is not None
            else None
        )
        if current_key is None or key < current_key:
            best_booster = booster
            best_record = record
    if best_booster is None or best_record is None:
        raise RuntimeError("No XGBoost candidate completed")
    return best_booster, best_record, records


def booster_predict(
    booster: Any, dmatrix: Any, *, num_boost_round: int | None = None
) -> np.ndarray:
    if num_boost_round is None:
        values = booster.predict(dmatrix)
    else:
        try:
            values = booster.predict(
                dmatrix, iteration_range=(0, int(num_boost_round))
            )
        except TypeError:  # XGBoost versions before iteration_range support.
            values = booster.predict(dmatrix, ntree_limit=int(num_boost_round))
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        if values.size % 5:
            raise RuntimeError("XGBoost returned an invalid probability vector")
        values = values.reshape(-1, 5)
    if values.ndim != 2 or values.shape[1] != 5:
        raise RuntimeError("XGBoost probabilities must have shape [rows, 5]")
    if not np.isfinite(values).all():
        raise RuntimeError("XGBoost returned NaN or Inf probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, atol=2e-5):
        raise RuntimeError("XGBoost probability rows do not sum to one")
    return values


def place_probabilities(
    mask: np.ndarray, selected_probabilities: np.ndarray
) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    if selected_probabilities.shape != (int(mask.sum()), 5):
        raise ValueError("Selected probability rows do not match the mask")
    output = np.full((*mask.shape, 5), np.nan, dtype=np.float32)
    output[mask] = selected_probabilities
    return output


def predict_split(
    xgb: Any,
    booster: Any,
    split: PreparedSplit,
    *,
    sector_category_count: int,
    num_boost_round: int | None = None,
) -> np.ndarray:
    if split.inference_mask is None:
        raise RuntimeError("Inference mask was not loaded")
    dmatrix = build_dmatrix(
        xgb,
        split,
        split.inference_mask,
        sector_category_count=sector_category_count,
        supervised=False,
    )
    selected = booster_predict(
        booster, dmatrix, num_boost_round=num_boost_round
    )
    return place_probabilities(split.inference_mask, selected)


def write_prediction_bundle(
    output_dir: Path,
    *,
    split: PreparedSplit,
    probabilities: np.ndarray,
    stage: str,
    fold_index: int,
) -> dict[str, Any]:
    if (
        split.permno is None
        or split.y_signed is None
        or split.target_return is None
        or split.target_date is None
        or split.inference_mask is None
    ):
        raise RuntimeError("Evaluation metadata was not loaded")
    if probabilities.shape != (*split.y.shape, 5):
        raise RuntimeError("Prediction shape does not align with the split")
    if not np.isfinite(probabilities[split.inference_mask]).all():
        raise RuntimeError("Inference-ready probabilities must be finite")
    if not np.isnan(probabilities[~split.inference_mask]).all():
        raise RuntimeError("Masked probabilities must be NaN")

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = prediction_metrics(
        y=split.y,
        target_return=split.target_return,
        evaluation_mask=split.loss_mask,
        probabilities=probabilities,
    )
    scores = (probabilities @ CLASS_VALUES).astype(np.float32)
    predicted_class = np.full(split.y.shape, LABEL_SENTINEL, dtype=np.int8)
    predicted_class[split.inference_mask] = probabilities[
        split.inference_mask
    ].argmax(axis=1).astype(np.int8)
    predicted_signed = np.full(split.y.shape, -128, dtype=np.int8)
    predicted_signed[split.inference_mask] = (
        predicted_class[split.inference_mask] - 2
    ).astype(np.int8)
    arrays = {
        "dates": split.dates,
        "permno": split.permno,
        "probabilities": probabilities,
        "scores": scores,
        "predicted_class": predicted_class,
        "predicted_signed": predicted_signed,
        "y": split.y,
        "y_signed": split.y_signed,
        "target_return": split.target_return,
        "target_date": split.target_date,
        "inference_mask": split.inference_mask,
        "evaluation_mask": split.loss_mask,
    }
    for name, values in arrays.items():
        with (output_dir / f"{name}.npy").open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    file_sha256 = {
        f"{name}.npy": sha256_file(output_dir / f"{name}.npy") for name in arrays
    }
    manifest = {
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "stage": stage,
        "fold_index": fold_index,
        "split": split.name,
        "years": split.years,
        "date_start": str(split.dates[0]),
        "date_end": str(split.dates[-1]),
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "probability_shape": list(probabilities.shape),
        "masked_prediction_policy": "probabilities/scores are NaN and classes use sentinels where inference_mask is false",
        "score_definition": "sum_{c=-2}^{2} c * p(c)",
        "evaluation_mask_definition": "loss_mask = label_mask AND complete_numeric_mask",
        "inference_mask_definition": "complete_numeric_mask",
        "file_sha256": file_sha256,
        "metrics": metrics,
    }
    write_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def fit_fold(
    xgb: Any,
    fold_dir: Path,
    output_dir: Path,
    *,
    config: dict[str, Any],
    config_hash: str,
    run_fingerprint: str,
    numeric_feature_names: list[str],
) -> dict[str, Any]:
    fold_manifest = load_json(fold_dir / "ready_fold_manifest.json")
    fold = fold_manifest["fold"]
    fold_index = int(fold["index"])
    fold_seed = int(config["seed"]) + fold_index * 10_000
    numeric_count = int(config["numeric_feature_count"])
    sector_count = int(config["sector_category_count"])
    started = time.perf_counter()

    train = load_prepared_split(
        fold_dir / "train",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=False,
    )
    validation = load_prepared_split(
        fold_dir / "validation",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=True,
    )
    dtrain = build_dmatrix(
        xgb,
        train,
        train.loss_mask,
        sector_category_count=sector_count,
        supervised=True,
    )
    dvalidation = build_dmatrix(
        xgb,
        validation,
        validation.loss_mask,
        sector_category_count=sector_count,
        supervised=True,
    )
    selection_model, selected_record, candidate_records = fit_candidates(
        xgb,
        dtrain,
        dvalidation,
        config=config,
        seed=fold_seed,
    )
    selected_candidate = next(
        candidate
        for candidate in config["candidates"]
        if str(candidate["id"]) == selected_record["candidate_id"]
    )
    selected_rounds = int(selected_record["best_num_boost_round"])
    selection_model.set_attr(
        model_name=MODEL_NAME,
        selected_num_boost_round=str(selected_rounds),
        run_fingerprint=run_fingerprint,
    )
    selection_checkpoint = output_dir / "selection_model.json"
    selection_model.save_model(selection_checkpoint)
    validation_started = time.perf_counter()
    validation_probabilities = predict_split(
        xgb,
        selection_model,
        validation,
        sector_category_count=sector_count,
        num_boost_round=selected_rounds,
    )
    validation_inference_seconds = time.perf_counter() - validation_started
    validation_manifest = write_prediction_bundle(
        output_dir / "predictions" / "validation",
        split=validation,
        probabilities=validation_probabilities,
        stage="train_only_selected_checkpoint",
        fold_index=fold_index,
    )
    del train, validation, dtrain, dvalidation, selection_model

    refit = load_prepared_split(
        fold_dir / "refit",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=False,
    )
    drefit = build_dmatrix(
        xgb,
        refit,
        refit.loss_mask,
        sector_category_count=sector_count,
        supervised=True,
    )
    refit_started = time.perf_counter()
    final_model = xgb.train(
        params=candidate_params(
            config, selected_candidate, seed=fold_seed + 9_000
        ),
        dtrain=drefit,
        num_boost_round=selected_rounds,
        verbose_eval=False,
    )
    refit_seconds = time.perf_counter() - refit_started
    final_model.set_attr(
        model_name=MODEL_NAME,
        selected_num_boost_round=str(selected_rounds),
        run_fingerprint=run_fingerprint,
    )
    checkpoint = output_dir / "model.json"
    final_model.save_model(checkpoint)
    del refit, drefit

    # Test data is opened only after candidate selection and refit are final.
    test = load_prepared_split(
        fold_dir / "test",
        numeric_feature_count=numeric_count,
        sector_category_count=sector_count,
        include_evaluation_metadata=True,
    )
    test_started = time.perf_counter()
    test_probabilities = predict_split(
        xgb,
        final_model,
        test,
        sector_category_count=sector_count,
    )
    test_inference_seconds = time.perf_counter() - test_started
    test_manifest = write_prediction_bundle(
        output_dir / "predictions" / "test",
        split=test,
        probabilities=test_probabilities,
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
            range(int(fold["train_start_year"]), int(fold["train_end_year"]) + 1)
        ),
        "validation_year": int(fold["validation_year"]),
        "refit_years": list(
            range(int(fold["train_start_year"]), int(fold["validation_year"]) + 1)
        ),
        "test_year": int(fold["test_year"]),
        "feature_contract": {
            "numeric_features": numeric_count,
            "numeric_feature_names": numeric_feature_names,
            "sector_categories": sector_count,
            "fixed_input_dimension": numeric_count + sector_count,
            "sector_implementation": "sparse one-hot CSR",
            "graph_used": False,
        },
        "objective": "date-equal weighted five-class multiclass log loss",
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "selection": {
            "metric": config["selection_metric"],
            "selected_candidate": selected_record["candidate_id"],
            "selected_num_boost_round": selected_rounds,
            "candidate_records": candidate_records,
            "validation_metrics": validation_manifest["metrics"],
            "validation_inference_seconds": validation_inference_seconds,
            "selection_checkpoint": selection_checkpoint.name,
            "selection_checkpoint_sha256": sha256_file(selection_checkpoint),
            "validation_prediction_manifest_sha256": sha256_file(
                output_dir
                / "predictions"
                / "validation"
                / "prediction_manifest.json"
            ),
        },
        "refit": {
            "seed": fold_seed + 9_000,
            "num_boost_round": selected_rounds,
            "training_seconds": refit_seconds,
        },
        "test": {
            "metrics": test_manifest["metrics"],
            "inference_seconds": test_inference_seconds,
            "probabilities_generated_once": True,
            "prediction_manifest_sha256": sha256_file(
                output_dir / "predictions" / "test" / "prediction_manifest.json"
            ),
        },
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "boosting_rounds": selected_rounds,
            "tree_count": selected_rounds * int(config["class_count"]),
            "device": str(config.get("device", "cpu")),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "fold_manifest.json", record)
    return record


def verify_prediction_bundle(path: Path, expected_manifest_hash: str) -> None:
    manifest_path = path / "prediction_manifest.json"
    if not manifest_path.exists() or sha256_file(manifest_path) != expected_manifest_hash:
        raise RuntimeError(f"Prediction manifest failed integrity check: {manifest_path}")
    manifest = load_json(manifest_path)
    for filename, expected_hash in manifest["file_sha256"].items():
        array_path = path / filename
        if not array_path.exists() or sha256_file(array_path) != expected_hash:
            raise RuntimeError(f"Prediction array failed integrity check: {array_path}")


def complete_fold_matches(path: Path, run_fingerprint: str) -> bool:
    manifest_path = path / "fold_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        return False
    if manifest.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(
            f"Existing fold belongs to another run: {path}. Use --overwrite."
        )
    checkpoint = path / str(manifest["checkpoint"]["path"])
    if (
        not checkpoint.exists()
        or sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]
    ):
        raise RuntimeError(f"Checkpoint failed integrity check: {checkpoint}")
    selection_checkpoint = path / str(
        manifest["selection"]["selection_checkpoint"]
    )
    if (
        not selection_checkpoint.exists()
        or sha256_file(selection_checkpoint)
        != manifest["selection"]["selection_checkpoint_sha256"]
    ):
        raise RuntimeError(
            f"Selection checkpoint failed integrity check: {selection_checkpoint}"
        )
    verify_prediction_bundle(
        path / "predictions" / "validation",
        manifest["selection"]["validation_prediction_manifest_sha256"],
    )
    verify_prediction_bundle(
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
    config = load_json(config_path)
    validate_config(config)
    xgb = require_xgboost()
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
    if int(feature_spec["combined_fixed_feature_dimension"]) != (
        int(config["numeric_feature_count"])
        + int(config["sector_category_count"])
    ):
        raise RuntimeError("M0-Plus feature dimension does not match feature_spec")

    config_hash = config_fingerprint(config)
    input_manifest_hash = sha256_file(input_manifest_path)
    feature_spec_hash = sha256_file(feature_spec_path)
    implementation_hash = sha256_file(Path(__file__))
    run_inputs = {
        "config_sha256": config_hash,
        "ready_for_use_manifest_sha256": input_manifest_hash,
        "feature_spec_sha256": feature_spec_hash,
        "implementation_sha256": implementation_hash,
        "xgboost_version": str(xgb.__version__),
    }
    run_fingerprint = config_fingerprint(run_inputs)

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_config = output_dir / "config.json"
    if saved_config.exists() and not overwrite:
        if config_fingerprint(load_json(saved_config)) != config_hash:
            raise RuntimeError(
                f"Existing {MODEL_NAME} output uses another config. Use --overwrite."
            )
    existing_manifest = output_dir / f"{MODEL_NAME}_manifest.json"
    if existing_manifest.exists() and not overwrite:
        previous = load_json(existing_manifest)
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Existing {MODEL_NAME} output belongs to another run. Use --overwrite."
            )
    write_json(saved_config, config)

    fold_dirs = sorted(path for path in input_dir.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != int(input_manifest["fold_count"]):
        raise RuntimeError("Ready fold directory count does not match manifest")
    selected: list[Path] = []
    found: set[int] = set()
    for fold_dir in fold_dirs:
        index = int(
            load_json(fold_dir / "ready_fold_manifest.json")["fold"]["index"]
        )
        if fold_indices is None or index in fold_indices:
            selected.append(fold_dir)
            found.add(index)
    if not selected:
        raise ValueError("No folds selected")
    if fold_indices is not None and found != fold_indices:
        raise ValueError(f"Requested folds not found: {sorted(fold_indices - found)}")

    run_started = time.perf_counter()
    for fold_dir in selected:
        final_fold = output_dir / fold_dir.name
        if complete_fold_matches(final_fold, run_fingerprint):
            print(f"[{fold_dir.name}] already complete; reusing", flush=True)
            continue
        if final_fold.exists():
            raise RuntimeError(f"Incomplete output exists: {final_fold}. Use --overwrite.")
        temporary = output_dir / f"{fold_dir.name}.part"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        print(f"[{fold_dir.name}] fitting {MODEL_NAME}", flush=True)
        try:
            record = fit_fold(
                xgb,
                fold_dir,
                temporary,
                config=config,
                config_hash=config_hash,
                run_fingerprint=run_fingerprint,
                numeric_feature_names=[
                    str(name) for name in feature_spec["numeric_columns"]
                ],
            )
            temporary.replace(final_fold)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(
            f"[{fold_dir.name}] selected={record['selection']['selected_candidate']} "
            f"rounds={record['selection']['selected_num_boost_round']} "
            f"test_ce={record['test']['metrics']['daily_mean_cross_entropy']:.6f}",
            flush=True,
        )

    completed: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        result_dir = output_dir / fold_dir.name
        if result_dir.exists() and complete_fold_matches(result_dir, run_fingerprint):
            completed.append(load_json(result_dir / "fold_manifest.json"))
    completed.sort(key=lambda record: int(record["fold_index"]))
    oos_manifest = build_oos_bundle(output_dir, completed)
    expected_count = int(input_manifest["fold_count"])
    manifest = {
        "status": "complete" if len(completed) == expected_count else "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "method": "XGBoost gradient-boosted decision trees with multi:softprob",
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
        "config": str(config_path.resolve()),
        "config_fingerprint": config_hash,
        "run_fingerprint": run_fingerprint,
        "run_inputs": run_inputs,
        "xgboost_version": str(xgb.__version__),
        "feature_contract": {
            "input": "standardized 0-cochain",
            "sector_representation": "sparse one-hot CSR",
            "graph_used": False,
        },
        "chronology": "8 train years / 1 validation year / refit on 9 years / 1 test year",
        "selection": "validation-only candidate and early-stopping round selection",
        "expected_fold_count": expected_count,
        "completed_fold_count": len(completed),
        "completed_fold_indices": [int(record["fold_index"]) for record in completed],
        "folds": completed,
        "oos": oos_manifest,
        "run_wall_seconds": time.perf_counter() - run_started,
    }
    write_json(output_dir / f"{MODEL_NAME}_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit M0-Plus XGBoost models through chronological folds."
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
