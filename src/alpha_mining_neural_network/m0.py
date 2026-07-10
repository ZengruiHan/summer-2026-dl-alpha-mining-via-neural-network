"""Train the proposal's M0 multinomial-logistic baseline.

M0 consumes the standardized 0-cochain only.  The 690-dimensional sector
one-hot block is represented exactly, but sparsely, by a sector-specific row
of coefficients instead of materializing a dense one-hot tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "models" / "M0"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "training" / "M0.json"
CLASS_VALUES = np.arange(-2, 3, dtype=np.float32)
LABEL_SENTINEL = -100


@dataclass
class PreparedSplit:
    """In-memory tensors for one chronological split."""

    name: str
    years: list[int]
    dates: np.ndarray
    x_numeric: np.ndarray
    sector_index: np.ndarray
    y: np.ndarray
    loss_mask: np.ndarray
    inference_mask: np.ndarray | None = None
    permno: np.ndarray | None = None
    y_signed: np.ndarray | None = None
    target_return: np.ndarray | None = None
    target_date: np.ndarray | None = None


@dataclass
class ModelParameters:
    """Coefficients for numeric features, sparse sector one-hot, and bias."""

    numeric_weight: np.ndarray
    sector_weight: np.ndarray
    bias: np.ndarray

    def copy(self) -> ModelParameters:
        return ModelParameters(
            self.numeric_weight.copy(), self.sector_weight.copy(), self.bias.copy()
        )


@dataclass
class AdamState:
    first: ModelParameters
    second: ModelParameters
    step: int = 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model") != "M0":
        raise ValueError("M0 config must set model='M0'")
    if config.get("model_family") != "multinomial_logistic":
        raise ValueError("M0 model_family must be multinomial_logistic")
    if config.get("selection_metric") != "validation_daily_mean_cross_entropy":
        raise ValueError("Unsupported selection metric")
    if config.get("optimizer") != "adam":
        raise ValueError("Only the deterministic NumPy Adam implementation is supported")
    for key in ("numeric_feature_count", "sector_category_count", "class_count"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config["class_count"]) != 5:
        raise ValueError("M0 requires five output classes")
    for key in ("max_epochs", "patience", "batch_dates"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("At least one hyperparameter candidate is required")
    identifiers: set[str] = set()
    for candidate in candidates:
        identifier = str(candidate.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("Candidate IDs must be non-empty and unique")
        identifiers.add(identifier)
        if float(candidate.get("learning_rate", 0.0)) <= 0:
            raise ValueError(f"Candidate {identifier} has invalid learning_rate")
        if float(candidate.get("l2", -1.0)) < 0:
            raise ValueError(f"Candidate {identifier} has invalid l2")


def _concatenate(parts: list[np.ndarray], name: str) -> np.ndarray:
    if not parts:
        raise RuntimeError(f"No arrays found for {name}")
    return np.concatenate(parts, axis=0)


def load_prepared_split(
    split_dir: Path,
    *,
    numeric_feature_count: int,
    sector_category_count: int,
    include_evaluation_metadata: bool,
) -> PreparedSplit:
    """Load one split without ever expanding sector one-hot features."""

    manifest = load_json(split_dir / "ready_split_manifest.json")
    years = [int(year) for year in manifest["years"]]
    arrays: dict[str, list[np.ndarray]] = {
        "dates": [],
        "x_numeric": [],
        "sector_index": [],
        "y": [],
        "loss_mask": [],
        "inference_mask": [],
    }
    if include_evaluation_metadata:
        arrays.update(
            {"permno": [], "y_signed": [], "target_return": [], "target_date": []}
        )

    previous_date: np.datetime64 | None = None
    for year in years:
        feature_dir = split_dir / "features" / f"year={year}"
        supervision_dir = split_dir / "supervision" / f"year={year}"
        dates = np.load(feature_dir / "dates.npy", allow_pickle=False)
        x_numeric = np.load(feature_dir / "x_numeric.npy", allow_pickle=False)
        sector = np.load(feature_dir / "sector_index.npy", allow_pickle=False)
        complete = np.load(
            feature_dir / "complete_numeric_mask.npy", allow_pickle=False
        )
        y = np.load(supervision_dir / "y.npy", allow_pickle=False)
        loss_mask = np.load(supervision_dir / "loss_mask.npy", allow_pickle=False)
        day_model_mask = np.load(
            supervision_dir / "day_model_mask.npy", allow_pickle=False
        )

        if x_numeric.ndim != 3 or x_numeric.shape[1:] != (
            500,
            numeric_feature_count,
        ):
            raise RuntimeError(f"Unexpected x_numeric shape in {feature_dir}")
        expected_node_shape = x_numeric.shape[:2]
        if sector.shape != expected_node_shape or y.shape != expected_node_shape:
            raise RuntimeError(f"Node array shape mismatch in year {year}")
        if loss_mask.shape != expected_node_shape or complete.shape != expected_node_shape:
            raise RuntimeError(f"Mask shape mismatch in year {year}")
        if dates.shape != (x_numeric.shape[0],):
            raise RuntimeError(f"Date shape mismatch in year {year}")
        if not np.all(np.diff(dates.astype("datetime64[D]").astype(np.int64)) > 0):
            raise RuntimeError(f"Dates are not strictly increasing in year {year}")
        if previous_date is not None and not dates[0] > previous_date:
            raise RuntimeError(f"Dates overlap or reverse at year {year}")
        previous_date = dates[-1]
        if not np.isfinite(x_numeric).all():
            raise RuntimeError(f"Non-finite model input in year {year}")
        if sector.min() < 0 or sector.max() >= sector_category_count:
            raise RuntimeError(f"Sector index outside configured vocabulary in year {year}")
        expected_loss_mask = complete & (y != LABEL_SENTINEL)
        if not np.array_equal(loss_mask, expected_loss_mask):
            raise RuntimeError(f"loss_mask contract failed in year {year}")
        if not np.array_equal(day_model_mask, loss_mask.any(axis=1)):
            raise RuntimeError(f"day_model_mask contract failed in year {year}")
        if not set(np.unique(y[loss_mask]).tolist()).issubset({0, 1, 2, 3, 4}):
            raise RuntimeError(f"Invalid M0 class ID in year {year}")

        arrays["dates"].append(dates.astype("datetime64[D]", copy=False))
        arrays["x_numeric"].append(x_numeric.astype(np.float32, copy=False))
        arrays["sector_index"].append(sector.astype(np.int16, copy=False))
        arrays["y"].append(y.astype(np.int8, copy=False))
        arrays["loss_mask"].append(loss_mask.astype(bool, copy=False))
        arrays["inference_mask"].append(complete.astype(bool, copy=False))

        if include_evaluation_metadata:
            permno = np.load(feature_dir / "permno.npy", allow_pickle=False)
            y_signed = np.load(
                supervision_dir / "y_signed.npy", allow_pickle=False
            )
            target_return = np.load(
                supervision_dir / "target_return.npy", allow_pickle=False
            )
            target_date = np.load(
                supervision_dir / "target_date.npy", allow_pickle=False
            )
            for name, value in (
                ("permno", permno),
                ("y_signed", y_signed),
                ("target_return", target_return),
                ("target_date", target_date),
            ):
                if value.shape != expected_node_shape:
                    raise RuntimeError(f"{name} shape mismatch in year {year}")
            if not np.array_equal(y_signed[loss_mask] + 2, y[loss_mask]):
                raise RuntimeError(f"Signed/model label mapping failed in year {year}")
            if not np.isfinite(target_return[loss_mask]).all():
                raise RuntimeError(f"Missing target return under loss_mask in year {year}")
            arrays["permno"].append(permno.astype(np.int32, copy=False))
            arrays["y_signed"].append(y_signed.astype(np.int8, copy=False))
            arrays["target_return"].append(
                target_return.astype(np.float32, copy=False)
            )
            arrays["target_date"].append(
                target_date.astype("datetime64[D]", copy=False)
            )

    return PreparedSplit(
        name=str(manifest["split"]),
        years=years,
        dates=_concatenate(arrays["dates"], "dates"),
        x_numeric=_concatenate(arrays["x_numeric"], "x_numeric"),
        sector_index=_concatenate(arrays["sector_index"], "sector_index"),
        y=_concatenate(arrays["y"], "y"),
        loss_mask=_concatenate(arrays["loss_mask"], "loss_mask"),
        inference_mask=_concatenate(arrays["inference_mask"], "inference_mask"),
        permno=(
            _concatenate(arrays["permno"], "permno")
            if include_evaluation_metadata
            else None
        ),
        y_signed=(
            _concatenate(arrays["y_signed"], "y_signed")
            if include_evaluation_metadata
            else None
        ),
        target_return=(
            _concatenate(arrays["target_return"], "target_return")
            if include_evaluation_metadata
            else None
        ),
        target_date=(
            _concatenate(arrays["target_date"], "target_date")
            if include_evaluation_metadata
            else None
        ),
    )


def initialize_model(
    numeric_feature_count: int, sector_category_count: int, class_count: int = 5
) -> ModelParameters:
    """Use the unique deterministic zero initialization for this convex model."""

    return ModelParameters(
        numeric_weight=np.zeros(
            (numeric_feature_count, class_count), dtype=np.float32
        ),
        sector_weight=np.zeros(
            (sector_category_count, class_count), dtype=np.float32
        ),
        bias=np.zeros(class_count, dtype=np.float32),
    )


def model_logits(
    parameters: ModelParameters, x_numeric: np.ndarray, sector_index: np.ndarray
) -> np.ndarray:
    return (
        x_numeric @ parameters.numeric_weight
        + parameters.sector_weight[sector_index]
        + parameters.bias
    )


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def softmax_and_true_class_nll(
    logits: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return softmax and uncapped NLL using a max-shift log-sum-exp."""

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalizer = exponentiated.sum(axis=-1, keepdims=True)
    probabilities = exponentiated / normalizer
    true_shifted_logit = shifted[np.arange(len(target)), target]
    negative_log_likelihood = np.log(normalizer[:, 0]) - true_shifted_logit
    return probabilities, negative_log_likelihood


def predict_probabilities(
    parameters: ModelParameters, split: PreparedSplit, *, chunk_dates: int = 128
) -> np.ndarray:
    node_count = split.y.shape[1]
    output = np.empty((*split.y.shape, 5), dtype=np.float32)
    for start in range(0, len(split.dates), chunk_dates):
        stop = min(start + chunk_dates, len(split.dates))
        x = split.x_numeric[start:stop].reshape(-1, split.x_numeric.shape[-1])
        sector = split.sector_index[start:stop].reshape(-1)
        probabilities = stable_softmax(model_logits(parameters, x, sector))
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Non-finite probability encountered during training")
        output[start:stop] = probabilities.reshape(stop - start, node_count, 5)
    if not np.isfinite(output).all():
        raise RuntimeError("M0 produced non-finite probabilities")
    if not np.allclose(output.sum(axis=2), 1.0, atol=2e-6):
        raise RuntimeError("M0 probability rows do not sum to one")
    return output


def daily_loss_accuracy(
    y: np.ndarray, loss_mask: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float, int]:
    losses: list[float] = []
    accuracies: list[float] = []
    node_count = 0
    for date_index in range(y.shape[0]):
        mask = loss_mask[date_index]
        if not mask.any():
            continue
        target = y[date_index, mask].astype(np.int64)
        selected = probabilities[date_index, mask]
        correct_probability = selected[np.arange(len(target)), target]
        losses.append(float(-np.log(np.clip(correct_probability, 1e-12, 1.0)).mean()))
        accuracies.append(float((selected.argmax(axis=1) == target).mean()))
        node_count += len(target)
    if not losses:
        raise RuntimeError("Split contains no model-ready labels")
    return float(np.mean(losses)), float(np.mean(accuracies)), node_count


def strict_model_loss_accuracy(
    parameters: ModelParameters,
    split: PreparedSplit,
    *,
    chunk_dates: int = 128,
) -> tuple[float, float, int]:
    """Evaluate exact daily-equal CE from logits rather than clipped probabilities."""

    daily_losses: list[float] = []
    daily_accuracies: list[float] = []
    node_count = 0
    node_axis = split.y.shape[1]
    for start in range(0, len(split.dates), chunk_dates):
        stop = min(start + chunk_dates, len(split.dates))
        x = split.x_numeric[start:stop].reshape(-1, split.x_numeric.shape[-1])
        sector = split.sector_index[start:stop].reshape(-1)
        logits = model_logits(parameters, x, sector).reshape(
            stop - start, node_axis, 5
        )
        for local_date in range(stop - start):
            absolute_date = start + local_date
            mask = split.loss_mask[absolute_date]
            if not mask.any():
                continue
            target = split.y[absolute_date, mask].astype(np.int64)
            selected_logits = logits[local_date, mask]
            probabilities, nll = softmax_and_true_class_nll(
                selected_logits, target
            )
            if not np.isfinite(nll).all():
                raise RuntimeError("Non-finite exact cross-entropy during evaluation")
            daily_losses.append(float(nll.mean()))
            daily_accuracies.append(
                float((probabilities.argmax(axis=1) == target).mean())
            )
            node_count += len(target)
    if not daily_losses:
        raise RuntimeError("Split contains no model-ready labels")
    return (
        float(np.mean(daily_losses)),
        float(np.mean(daily_accuracies)),
        node_count,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
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


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def prediction_metrics(
    *,
    y: np.ndarray,
    target_return: np.ndarray,
    evaluation_mask: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    cross_entropy, accuracy, node_count = daily_loss_accuracy(
        y, evaluation_mask, probabilities
    )
    scores = probabilities @ CLASS_VALUES
    rank_ic: list[float] = []
    pearson_ic: list[float] = []
    evaluated_dates = 0
    for date_index in range(y.shape[0]):
        mask = evaluation_mask[date_index] & np.isfinite(target_return[date_index])
        if mask.sum() < 2:
            continue
        evaluated_dates += 1
        daily_score = scores[date_index, mask]
        daily_return = target_return[date_index, mask]
        pearson = _correlation(daily_score, daily_return)
        spearman = _correlation(
            _average_ranks(daily_score), _average_ranks(daily_return)
        )
        if pearson is not None:
            pearson_ic.append(pearson)
        if spearman is not None:
            rank_ic.append(spearman)
    return {
        "daily_mean_cross_entropy": cross_entropy,
        "daily_mean_accuracy": accuracy,
        "daily_mean_rank_ic": float(np.mean(rank_ic)) if rank_ic else None,
        "daily_mean_pearson_ic": float(np.mean(pearson_ic)) if pearson_ic else None,
        "model_ready_nodes": node_count,
        "calendar_dates": int(y.shape[0]),
        "evaluated_dates": evaluated_dates,
        "rank_ic_valid_dates": len(rank_ic),
        "pearson_ic_valid_dates": len(pearson_ic),
    }


def _zeros_like(parameters: ModelParameters) -> ModelParameters:
    return ModelParameters(
        np.zeros_like(parameters.numeric_weight),
        np.zeros_like(parameters.sector_weight),
        np.zeros_like(parameters.bias),
    )


def initialize_adam(parameters: ModelParameters) -> AdamState:
    return AdamState(_zeros_like(parameters), _zeros_like(parameters))


def _adam_array_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    first *= beta1
    first += (1.0 - beta1) * gradient
    second *= beta2
    second += (1.0 - beta2) * np.square(gradient)
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)


def adam_update(
    parameters: ModelParameters,
    gradients: ModelParameters,
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    state.step += 1
    for parameter, gradient, first, second in (
        (
            parameters.numeric_weight,
            gradients.numeric_weight,
            state.first.numeric_weight,
            state.second.numeric_weight,
        ),
        (
            parameters.sector_weight,
            gradients.sector_weight,
            state.first.sector_weight,
            state.second.sector_weight,
        ),
        (parameters.bias, gradients.bias, state.first.bias, state.second.bias),
    ):
        _adam_array_update(
            parameter,
            gradient,
            first,
            second,
            step=state.step,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )


def train_one_epoch(
    parameters: ModelParameters,
    state: AdamState,
    split: PreparedSplit,
    *,
    rng: np.random.Generator,
    learning_rate: float,
    l2: float,
    batch_dates: int,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> float:
    """Take one shuffled pass; every batch averages dates, then nodes within date."""

    active_dates = np.flatnonzero(split.loss_mask.any(axis=1))
    date_order = rng.permutation(active_dates)
    batch_count = math.ceil(len(date_order) / batch_dates)
    objective_sum = 0.0
    objective_dates = 0
    sector_count = parameters.sector_weight.shape[0]
    class_count = parameters.bias.shape[0]

    # array_split makes batch sizes differ by at most one, so a short final
    # batch does not receive the same full Adam step as a normal-size batch.
    for batch_indices in np.array_split(date_order, batch_count):
        batch_mask = split.loss_mask[batch_indices]
        counts = batch_mask.sum(axis=1)
        active = counts > 0
        if not active.any():
            continue
        batch_indices = batch_indices[active]
        batch_mask = batch_mask[active]
        counts = counts[active]
        active_date_count = len(batch_indices)

        x_all = split.x_numeric[batch_indices]
        sector_all = split.sector_index[batch_indices]
        y_all = split.y[batch_indices]
        x = x_all[batch_mask]
        sector = sector_all[batch_mask].astype(np.int64, copy=False)
        target = y_all[batch_mask].astype(np.int64, copy=False)
        per_date_node_weight = 1.0 / counts.astype(np.float32)
        weight_grid = np.broadcast_to(
            per_date_node_weight[:, None], batch_mask.shape
        )
        sample_weight = weight_grid[batch_mask] / active_date_count

        logits = model_logits(parameters, x, sector)
        probabilities, negative_log_likelihood = softmax_and_true_class_nll(
            logits, target
        )
        data_loss = float(np.sum(negative_log_likelihood * sample_weight))
        penalty = 0.5 * l2 * float(
            np.square(parameters.numeric_weight).sum()
            + np.square(parameters.sector_weight).sum()
        )

        error = probabilities
        error[np.arange(len(target)), target] -= 1.0
        error *= sample_weight[:, None]
        numeric_gradient = x.T @ error
        numeric_gradient += l2 * parameters.numeric_weight
        sector_gradient = np.empty_like(parameters.sector_weight)
        for class_index in range(class_count):
            sector_gradient[:, class_index] = np.bincount(
                sector,
                weights=error[:, class_index],
                minlength=sector_count,
            ).astype(np.float32, copy=False)
        sector_gradient += l2 * parameters.sector_weight
        gradients = ModelParameters(
            numeric_gradient.astype(np.float32, copy=False),
            sector_gradient,
            error.sum(axis=0).astype(np.float32, copy=False),
        )
        adam_update(
            parameters,
            gradients,
            state,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        if not (
            np.isfinite(parameters.numeric_weight).all()
            and np.isfinite(parameters.sector_weight).all()
            and np.isfinite(parameters.bias).all()
        ):
            raise RuntimeError("Non-finite M0 parameter encountered during training")
        objective_sum += (data_loss + penalty) * active_date_count
        objective_dates += active_date_count

    if objective_dates == 0:
        raise RuntimeError("Training split contains no active dates")
    return objective_sum / objective_dates


def fit_candidate(
    train: PreparedSplit,
    validation: PreparedSplit,
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
) -> tuple[ModelParameters, dict[str, Any]]:
    parameters = initialize_model(
        int(config["numeric_feature_count"]),
        int(config["sector_category_count"]),
        int(config["class_count"]),
    )
    state = initialize_adam(parameters)
    rng = np.random.default_rng(seed)
    best_parameters: ModelParameters | None = None
    best_loss = float("inf")
    best_epoch = 0
    patience_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_objective = train_one_epoch(
            parameters,
            state,
            train,
            rng=rng,
            learning_rate=float(candidate["learning_rate"]),
            l2=float(candidate["l2"]),
            batch_dates=int(config["batch_dates"]),
            beta1=float(config["adam_beta1"]),
            beta2=float(config["adam_beta2"]),
            epsilon=float(config["adam_epsilon"]),
        )
        validation_loss, validation_accuracy, validation_nodes = strict_model_loss_accuracy(
            parameters, validation
        )
        history.append(
            {
                "epoch": epoch,
                "mean_batch_training_objective": train_objective,
                "validation_daily_mean_cross_entropy": validation_loss,
                "validation_daily_mean_accuracy": validation_accuracy,
                "validation_model_ready_nodes": validation_nodes,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_parameters = parameters.copy()
        if validation_loss < patience_loss - float(config["min_delta"]):
            patience_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["patience"]):
            break

    if best_parameters is None:
        raise RuntimeError("No validation checkpoint was selected")
    record = {
        "candidate_id": str(candidate["id"]),
        "learning_rate": float(candidate["learning_rate"]),
        "l2": float(candidate["l2"]),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_daily_mean_cross_entropy": best_loss,
        "epochs_run": len(history),
        "stopped_early": len(history) < int(config["max_epochs"]),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }
    return best_parameters, record


def refit_model(
    refit: PreparedSplit,
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    epochs: int,
    seed: int,
) -> tuple[ModelParameters, dict[str, Any]]:
    parameters = initialize_model(
        int(config["numeric_feature_count"]),
        int(config["sector_category_count"]),
        int(config["class_count"]),
    )
    state = initialize_adam(parameters)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        objective = train_one_epoch(
            parameters,
            state,
            refit,
            rng=rng,
            learning_rate=float(candidate["learning_rate"]),
            l2=float(candidate["l2"]),
            batch_dates=int(config["batch_dates"]),
            beta1=float(config["adam_beta1"]),
            beta2=float(config["adam_beta2"]),
            epsilon=float(config["adam_epsilon"]),
        )
        history.append(
            {"epoch": epoch, "mean_batch_training_objective": objective}
        )
    return parameters, {
        "seed": seed,
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


def save_model(
    path: Path,
    parameters: ModelParameters,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "numeric_weight": parameters.numeric_weight,
        "sector_weight": parameters.sector_weight,
        "bias": parameters.bias,
        "class_values": CLASS_VALUES.astype(np.int8),
    }
    for key, value in (metadata or {}).items():
        payload[key] = np.asarray(value)
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def load_model(path: Path) -> ModelParameters:
    with np.load(path, allow_pickle=False) as checkpoint:
        if not np.array_equal(checkpoint["class_values"], CLASS_VALUES.astype(np.int8)):
            raise RuntimeError(f"Unexpected class mapping in {path}")
        return ModelParameters(
            checkpoint["numeric_weight"].astype(np.float32, copy=True),
            checkpoint["sector_weight"].astype(np.float32, copy=True),
            checkpoint["bias"].astype(np.float32, copy=True),
        )


def write_prediction_bundle(
    output_dir: Path,
    *,
    split: PreparedSplit,
    probabilities: np.ndarray,
    parameters: ModelParameters,
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
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = prediction_metrics(
        y=split.y,
        target_return=split.target_return,
        evaluation_mask=split.loss_mask,
        probabilities=probabilities,
    )
    exact_loss, exact_accuracy, exact_nodes = strict_model_loss_accuracy(
        parameters, split
    )
    if exact_nodes != metrics["model_ready_nodes"]:
        raise RuntimeError("Exact and prediction metric node counts disagree")
    metrics["daily_mean_cross_entropy"] = exact_loss
    metrics["daily_mean_accuracy"] = exact_accuracy

    stored_probabilities = probabilities.astype(np.float32, copy=True)
    stored_probabilities[~split.inference_mask] = np.nan
    scores = (stored_probabilities @ CLASS_VALUES).astype(np.float32)
    predicted_class = probabilities.argmax(axis=2).astype(np.int8)
    predicted_signed = (predicted_class - 2).astype(np.int8)
    predicted_class[~split.inference_mask] = np.int8(LABEL_SENTINEL)
    predicted_signed[~split.inference_mask] = np.int8(-128)
    arrays = {
        "dates": split.dates,
        "permno": split.permno,
        "probabilities": stored_probabilities,
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


def _parameter_count(config: dict[str, Any]) -> int:
    return (
        int(config["numeric_feature_count"])
        * int(config["class_count"])
        + int(config["sector_category_count"])
        * int(config["class_count"])
        + int(config["class_count"])
    )


def fit_fold(
    fold_dir: Path,
    output_dir: Path,
    *,
    config: dict[str, Any],
    config_hash: str,
    run_fingerprint: str,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    fold_manifest = load_json(fold_dir / "ready_fold_manifest.json")
    fold = fold_manifest["fold"]
    fold_index = int(fold["index"])
    fold_seed = int(config["seed"]) + fold_index * 10_000
    started = time.perf_counter()

    train = load_prepared_split(
        fold_dir / "train",
        numeric_feature_count=int(config["numeric_feature_count"]),
        sector_category_count=int(config["sector_category_count"]),
        include_evaluation_metadata=False,
    )
    validation = load_prepared_split(
        fold_dir / "validation",
        numeric_feature_count=int(config["numeric_feature_count"]),
        sector_category_count=int(config["sector_category_count"]),
        include_evaluation_metadata=True,
    )

    candidate_records: list[dict[str, Any]] = []
    candidate_models: dict[str, ModelParameters] = {}
    for candidate in config["candidates"]:
        # Every candidate sees the same deterministic date-batch order so the
        # validation comparison is not confounded by a different shuffle.
        candidate_seed = fold_seed
        model, record = fit_candidate(
            train,
            validation,
            config=config,
            candidate=candidate,
            seed=candidate_seed,
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
    save_model(
        selection_checkpoint, selection_model, metadata=checkpoint_metadata
    )
    validation_started = time.perf_counter()
    validation_probabilities = predict_probabilities(selection_model, validation)
    validation_inference_seconds = time.perf_counter() - validation_started
    validation_prediction_manifest = write_prediction_bundle(
        output_dir / "predictions" / "validation",
        split=validation,
        probabilities=validation_probabilities,
        parameters=selection_model,
        stage="train_only_selected_checkpoint",
        fold_index=fold_index,
    )

    del train, candidate_models, selection_model, validation_probabilities

    refit = load_prepared_split(
        fold_dir / "refit",
        numeric_feature_count=int(config["numeric_feature_count"]),
        sector_category_count=int(config["sector_category_count"]),
        include_evaluation_metadata=False,
    )
    refit_seed = fold_seed + 9_000
    final_model, refit_record = refit_model(
        refit,
        config=config,
        candidate=selected_candidate,
        epochs=int(selected_record["best_epoch"]),
        seed=refit_seed,
    )
    final_checkpoint = output_dir / "model.npz"
    save_model(final_checkpoint, final_model, metadata=checkpoint_metadata)
    del refit

    # Test data is opened only after selection and refit are final.  This is
    # the sole call that generates test probabilities for this fold.
    test = load_prepared_split(
        fold_dir / "test",
        numeric_feature_count=int(config["numeric_feature_count"]),
        sector_category_count=int(config["sector_category_count"]),
        include_evaluation_metadata=True,
    )
    test_started = time.perf_counter()
    test_probabilities = predict_probabilities(final_model, test)
    test_inference_seconds = time.perf_counter() - test_started
    test_prediction_manifest = write_prediction_bundle(
        output_dir / "predictions" / "test",
        split=test,
        probabilities=test_probabilities,
        parameters=final_model,
        stage="refit_train_plus_validation_checkpoint",
        fold_index=fold_index,
    )

    record = {
        "status": "complete",
        "model": "M0",
        "model_family": "multinomial_logistic",
        "fold_index": fold_index,
        "fold_name": fold["name"],
        "config_fingerprint": config_hash,
        "run_fingerprint": run_fingerprint,
        "input_fold": str(fold_dir.resolve()),
        "train_years": list(range(int(fold["train_start_year"]), int(fold["train_end_year"]) + 1)),
        "validation_year": int(fold["validation_year"]),
        "refit_years": list(range(int(fold["train_start_year"]), int(fold["validation_year"]) + 1)),
        "test_year": int(fold["test_year"]),
        "feature_contract": {
            "numeric_features": int(config["numeric_feature_count"]),
            "numeric_feature_names": checkpoint_metadata["numeric_feature_names"],
            "sector_categories": int(config["sector_category_count"]),
            "fixed_input_dimension": int(config["numeric_feature_count"])
            + int(config["sector_category_count"]),
            "sector_implementation": "sparse coefficient lookup exactly equivalent to one-hot",
            "graph_used": False,
        },
        "objective": "daily-equal mean categorical cross-entropy; node-equal within date",
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "selection": {
            "metric": config["selection_metric"],
            "selected_candidate": selected_record["candidate_id"],
            "selected_learning_rate": selected_record["learning_rate"],
            "selected_l2": selected_record["l2"],
            "selected_epoch": selected_record["best_epoch"],
            "candidate_records": candidate_records,
            "validation_metrics": validation_prediction_manifest["metrics"],
            "validation_inference_seconds": validation_inference_seconds,
            "selection_checkpoint": "selection_model.npz",
            "selection_checkpoint_sha256": sha256_file(selection_checkpoint),
            "validation_prediction_manifest_sha256": sha256_file(
                output_dir
                / "predictions"
                / "validation"
                / "prediction_manifest.json"
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
            "parameter_count": _parameter_count(config),
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
        raise RuntimeError(f"Prediction manifest failed integrity check: {manifest_path}")
    prediction_manifest = load_json(manifest_path)
    for filename, expected_hash in prediction_manifest["file_sha256"].items():
        array_path = prediction_dir / filename
        if not array_path.exists() or sha256_file(array_path) != expected_hash:
            raise RuntimeError(
                f"Prediction array failed integrity check: {array_path}"
            )


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
    if not checkpoint.exists() or sha256_file(checkpoint) != manifest["checkpoint"]["sha256"]:
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


def build_oos_bundle(output_dir: Path, fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    temporary = output_dir / "oos.part"
    final = output_dir / "oos"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    names = (
        "dates",
        "permno",
        "probabilities",
        "scores",
        "predicted_class",
        "predicted_signed",
        "y",
        "y_signed",
        "target_return",
        "target_date",
        "inference_mask",
        "evaluation_mask",
    )
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in names}
    fold_index_parts: list[np.ndarray] = []
    for record in sorted(fold_records, key=lambda item: item["fold_index"]):
        prediction_dir = (
            output_dir / record["fold_name"] / "predictions" / "test"
        )
        date_count = 0
        for name in names:
            value = np.load(prediction_dir / f"{name}.npy", allow_pickle=False)
            pieces[name].append(value)
            if name == "dates":
                date_count = len(value)
        fold_index_parts.append(
            np.full(date_count, int(record["fold_index"]), dtype=np.int8)
        )
    combined = {name: _concatenate(values, name) for name, values in pieces.items()}
    fold_index = _concatenate(fold_index_parts, "fold_index")
    date_numbers = combined["dates"].astype("datetime64[D]").astype(np.int64)
    if not np.all(np.diff(date_numbers) > 0):
        raise RuntimeError("Concatenated OOS dates are not strictly increasing")
    if any(len(np.unique(row)) != row.size for row in combined["permno"]):
        raise RuntimeError("Concatenated OOS output has a duplicate daily PERMNO")
    inference_mask = combined["inference_mask"]
    probabilities = combined["probabilities"]
    if not np.isfinite(probabilities[inference_mask]).all():
        raise RuntimeError("Finite OOS inference rows contain invalid probabilities")
    if not np.isnan(probabilities[~inference_mask]).all():
        raise RuntimeError("Masked OOS inference rows do not use NaN probabilities")
    if not np.allclose(
        probabilities[inference_mask].sum(axis=1), 1.0, atol=2e-6
    ):
        raise RuntimeError("OOS probability rows do not sum to one")
    for name, values in {**combined, "fold_index": fold_index}.items():
        with (temporary / f"{name}.npy").open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    metrics = prediction_metrics(
        y=combined["y"],
        target_return=combined["target_return"],
        evaluation_mask=combined["evaluation_mask"],
        probabilities=combined["probabilities"],
    )
    metric_date_count = sum(
        int(record["test"]["metrics"]["evaluated_dates"])
        for record in fold_records
    )
    if metric_date_count:
        for metric_name in ("daily_mean_cross_entropy", "daily_mean_accuracy"):
            metrics[metric_name] = sum(
                float(record["test"]["metrics"][metric_name])
                * int(record["test"]["metrics"]["evaluated_dates"])
                for record in fold_records
            ) / metric_date_count
    manifest = {
        "fold_count": len(fold_records),
        "fold_indices": [int(record["fold_index"]) for record in fold_records],
        "date_start": str(combined["dates"][0]),
        "date_end": str(combined["dates"][-1]),
        "probability_shape": list(combined["probabilities"].shape),
        "non_overlapping_strictly_increasing_dates": True,
        "class_id_to_signed_label": {str(index): index - 2 for index in range(5)},
        "metrics": metrics,
    }
    write_json(temporary / "oos_manifest.json", manifest)
    shutil.rmtree(final, ignore_errors=True)
    temporary.replace(final)
    return manifest


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
    if int(feature_spec["combined_fixed_feature_dimension"]) != (
        int(config["numeric_feature_count"])
        + int(config["sector_category_count"])
    ):
        raise RuntimeError("M0 fixed feature dimension does not match feature_spec")

    input_manifest_hash = sha256_file(input_manifest_path)
    feature_spec_hash = sha256_file(feature_spec_path)
    implementation_hash = sha256_file(Path(__file__))
    proposal_path = REPOSITORY_ROOT / "docs" / "proposal" / "proposal.pdf"
    proposal_hash = sha256_file(proposal_path)
    run_inputs = {
        "config_sha256": config_hash,
        "ready_for_use_manifest_sha256": input_manifest_hash,
        "feature_spec_sha256": feature_spec_hash,
        "implementation_sha256": implementation_hash,
        "proposal_sha256": proposal_hash,
    }
    run_fingerprint = config_fingerprint(run_inputs)
    checkpoint_metadata: dict[str, Any] = {
        "model_family": "multinomial_logistic",
        "numeric_feature_names": feature_spec["numeric_columns"],
        "feature_spec_sha256": feature_spec_hash,
        "ready_for_use_manifest_sha256": input_manifest_hash,
        "implementation_sha256": implementation_hash,
        "run_fingerprint": run_fingerprint,
        "parameter_dtype": "float32",
        "numpy_version": np.__version__,
    }

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = output_dir / "config.json"
    if existing_config_path.exists() and not overwrite:
        existing_config = load_json(existing_config_path)
        if config_fingerprint(existing_config) != config_hash:
            raise RuntimeError(
                f"Existing M0 output uses a different config: {output_dir}. "
                "Use --overwrite."
            )
    existing_run_manifest = output_dir / "M0_manifest.json"
    if existing_run_manifest.exists() and not overwrite:
        previous = load_json(existing_run_manifest)
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"Existing M0 output binds to different data or code: {output_dir}. "
                "Use --overwrite."
            )
    write_json(output_dir / "config.json", config)

    fold_dirs = sorted(path for path in input_dir.glob("fold_*") if path.is_dir())
    if len(fold_dirs) != int(input_manifest["fold_count"]):
        raise RuntimeError("Ready fold directory count does not match manifest")
    selected_fold_dirs: list[Path] = []
    for fold_dir in fold_dirs:
        fold_index = int(load_json(fold_dir / "ready_fold_manifest.json")["fold"]["index"])
        if fold_indices is None or fold_index in fold_indices:
            selected_fold_dirs.append(fold_dir)
    if not selected_fold_dirs:
        raise ValueError("No folds selected")
    if fold_indices is not None:
        found = {
            int(load_json(path / "ready_fold_manifest.json")["fold"]["index"])
            for path in selected_fold_dirs
        }
        if found != fold_indices:
            raise ValueError(f"Requested folds not found: {sorted(fold_indices - found)}")

    run_started = time.perf_counter()
    for fold_dir in selected_fold_dirs:
        fold_name = fold_dir.name
        final_fold = output_dir / fold_name
        if _complete_fold_matches(final_fold, run_fingerprint):
            print(f"[{fold_name}] already complete; reusing", flush=True)
            continue
        if final_fold.exists():
            raise RuntimeError(f"Incomplete output exists: {final_fold}. Use --overwrite.")
        temporary_fold = output_dir / f"{fold_name}.part"
        shutil.rmtree(temporary_fold, ignore_errors=True)
        temporary_fold.mkdir(parents=True)
        print(f"[{fold_name}] fitting M0", flush=True)
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
            f"[{fold_name}] selected={record['selection']['selected_candidate']} "
            f"epoch={record['selection']['selected_epoch']} "
            f"test_ce={record['test']['metrics']['daily_mean_cross_entropy']:.6f}",
            flush=True,
        )

    completed_records: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        result_dir = output_dir / fold_dir.name
        if result_dir.exists() and _complete_fold_matches(
            result_dir, run_fingerprint
        ):
            completed_records.append(load_json(result_dir / "fold_manifest.json"))
    completed_records.sort(key=lambda item: item["fold_index"])
    oos_manifest = build_oos_bundle(output_dir, completed_records)
    expected_fold_count = int(input_manifest["fold_count"])
    manifest = {
        "status": "complete" if len(completed_records) == expected_fold_count else "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "M0",
        "model_family": "multinomial_logistic",
        "proposal_contract": {
            "input": "standardized 0-cochain",
            "graph_used": False,
            "objective": "five-class daily-equal categorical cross-entropy",
            "chronology": "8 train years / 1 validation year / refit on 9 years / 1 test year",
            "selection": "validation-only hyperparameter and early-stopping epoch selection",
            "test_probability_policy": "generated once after refit for each fold",
        },
        "implementation_decisions_not_fixed_by_proposal": {
            "optimizer": config["optimizer"],
            "selection_metric": config["selection_metric"],
            "sector_representation": "sparse lookup equivalent to 690-dimensional one-hot; no redundant learned embedding",
            "linear_logits": "Wh+b is passed directly to softmax",
        },
        "config": str(config_path.resolve()),
        "config_fingerprint": config_hash,
        "run_fingerprint": run_fingerprint,
        "run_inputs": run_inputs,
        "input": str(input_dir.resolve()),
        "input_manifest_sha256": input_manifest_hash,
        "feature_spec_sha256": feature_spec_hash,
        "implementation_sha256": implementation_hash,
        "proposal_sha256": proposal_hash,
        "expected_fold_count": expected_fold_count,
        "completed_fold_count": len(completed_records),
        "completed_fold_indices": [
            int(record["fold_index"]) for record in completed_records
        ],
        "parameter_count_per_fold": _parameter_count(config),
        "folds": completed_records,
        "oos": oos_manifest,
        "run_wall_seconds": time.perf_counter() - run_started,
    }
    write_json(output_dir / "M0_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit M0 through chronological folds and save checkpoints/predictions."
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
        help="Delete the M0 output directory before fitting.",
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
        f"M0 folds: {manifest['completed_fold_count']}/{manifest['expected_fold_count']}",
        flush=True,
    )
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
