"""Dependency-free graph convolutional network primitives and training.

The project models every trading date as an independent graph.  Edges use the
message-passing convention ``sender -> receiver`` (equivalently,
``A[receiver, sender]``).  The implementation is deliberately NumPy-only so it
can run in the same minimal environment as the M0 baseline.

The first layer keeps the 690-dimensional sector one-hot block sparse:

    Q0 = X_numeric @ W_numeric + W_sector[sector_index]

This is exactly equivalent to multiplying a materialized sector one-hot
matrix by its slice of the first-layer weight matrix.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from alpha_mining_neural_network.m0 import (
    CLASS_VALUES,
    PreparedSplit,
    softmax_and_true_class_nll,
    stable_softmax,
)


@dataclass(frozen=True)
class NormalizedGraph:
    """One normalized daily graph in COO message-passing form."""

    node_count: int
    sender: np.ndarray
    receiver: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        if self.node_count <= 0:
            raise ValueError("node_count must be positive")
        if self.sender.ndim != 1 or self.receiver.ndim != 1 or self.weight.ndim != 1:
            raise ValueError("Graph arrays must be one-dimensional")
        if not (len(self.sender) == len(self.receiver) == len(self.weight)):
            raise ValueError("Graph arrays must have the same length")
        if not np.issubdtype(self.weight.dtype, np.floating):
            raise TypeError("Normalized graph weights must use a floating dtype")
        if len(self.sender):
            if self.sender.min() < 0 or self.sender.max() >= self.node_count:
                raise ValueError("sender contains an invalid node index")
            if self.receiver.min() < 0 or self.receiver.max() >= self.node_count:
                raise ValueError("receiver contains an invalid node index")
        if not np.isfinite(self.weight).all():
            raise ValueError("Normalized graph weights must be finite")

    def aggregate(self, values: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply the normalized adjacency, or its transpose, to node values."""

        values = np.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.node_count:
            raise ValueError(
                "values must have shape [node_count] or [node_count, features]"
            )
        one_dimensional = values.ndim == 1
        matrix = values[:, None] if one_dimensional else values
        output = np.zeros(
            (self.node_count, matrix.shape[1]),
            dtype=np.result_type(matrix.dtype, self.weight.dtype),
        )
        source = self.receiver if transpose else self.sender
        destination = self.sender if transpose else self.receiver
        np.add.at(output, destination, self.weight[:, None] * matrix[source])
        return output[:, 0] if one_dimensional else output

    def to_dense(self) -> np.ndarray:
        """Materialize a small graph for diagnostics and tests."""

        dense = np.zeros((self.node_count, self.node_count), dtype=self.weight.dtype)
        np.add.at(dense, (self.receiver, self.sender), self.weight)
        return dense


def normalized_graph(
    node_count: int,
    sender: np.ndarray | Iterable[int],
    receiver: np.ndarray | Iterable[int],
    weight: np.ndarray | Iterable[float] | None = None,
    *,
    add_reverse_edges: bool = False,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> NormalizedGraph:
    """Add one self-loop per node and apply absolute-degree normalization.

    For a directed, possibly signed edge ``u -> v``, normalization is

    ``w / sqrt(out_degree[u] * in_degree[v])``.

    Both degrees use absolute edge weights, preventing positive and negative
    correlation edges from cancelling.  Undirected graphs reduce to the usual
    symmetric GCN normalization.
    """

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    resolved_dtype = np.dtype(dtype)
    if not np.issubdtype(resolved_dtype, np.floating):
        raise TypeError("dtype must be a floating NumPy dtype")
    sender_array = np.asarray(sender, dtype=np.int64).reshape(-1)
    receiver_array = np.asarray(receiver, dtype=np.int64).reshape(-1)
    if sender_array.shape != receiver_array.shape:
        raise ValueError("sender and receiver must have the same shape")
    if weight is None:
        weight_array = np.ones(len(sender_array), dtype=resolved_dtype)
    else:
        weight_array = np.asarray(weight, dtype=resolved_dtype).reshape(-1)
    if weight_array.shape != sender_array.shape:
        raise ValueError("weight must align with sender and receiver")
    if len(sender_array):
        if sender_array.min() < 0 or sender_array.max() >= node_count:
            raise ValueError("sender contains an invalid node index")
        if receiver_array.min() < 0 or receiver_array.max() >= node_count:
            raise ValueError("receiver contains an invalid node index")
    if not np.isfinite(weight_array).all():
        raise ValueError("Edge weights must be finite")

    # Source skeletons must not supply self-loops; the model owns this step and
    # adds exactly one +1 loop to every node.
    non_loop = sender_array != receiver_array
    sender_array = sender_array[non_loop]
    receiver_array = receiver_array[non_loop]
    weight_array = weight_array[non_loop]
    if add_reverse_edges and len(sender_array):
        original_sender = sender_array
        original_receiver = receiver_array
        original_weight = weight_array
        sender_array = np.concatenate((original_sender, original_receiver))
        receiver_array = np.concatenate((original_receiver, original_sender))
        weight_array = np.concatenate((original_weight, original_weight))

    nodes = np.arange(node_count, dtype=np.int64)
    sender_array = np.concatenate((sender_array, nodes))
    receiver_array = np.concatenate((receiver_array, nodes))
    weight_array = np.concatenate(
        (weight_array, np.ones(node_count, dtype=weight_array.dtype))
    )

    # Coalesce duplicates before degree calculation.  This is important for
    # stable np.add.at semantics and makes the stored graph canonical.
    key = receiver_array * node_count + sender_array
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    sorted_weight = weight_array[order]
    unique_key, starts = np.unique(sorted_key, return_index=True)
    coalesced_weight = np.add.reduceat(sorted_weight, starts)
    nonzero = coalesced_weight != 0
    unique_key = unique_key[nonzero]
    coalesced_weight = coalesced_weight[nonzero]
    receiver_array = unique_key // node_count
    sender_array = unique_key % node_count

    absolute = np.abs(coalesced_weight).astype(np.float64, copy=False)
    out_degree = np.bincount(sender_array, weights=absolute, minlength=node_count)
    in_degree = np.bincount(receiver_array, weights=absolute, minlength=node_count)
    if (out_degree <= 0).any() or (in_degree <= 0).any():
        raise RuntimeError("Self-loops failed to give every node positive degree")
    denominator = np.sqrt(out_degree[sender_array] * in_degree[receiver_array])
    normalized_weight = (coalesced_weight / denominator).astype(
        resolved_dtype, copy=False
    )
    return NormalizedGraph(
        node_count=node_count,
        sender=sender_array.astype(np.int32, copy=False),
        receiver=receiver_array.astype(np.int32, copy=False),
        weight=normalized_weight,
    )


def sector_graph(sector_index: np.ndarray) -> NormalizedGraph:
    """Create the proposal's same-sector graph for one date."""

    sector = np.asarray(sector_index)
    if sector.ndim != 1 or not np.issubdtype(sector.dtype, np.integer):
        raise ValueError("sector_index must be a one-dimensional integer array")
    sender_parts: list[np.ndarray] = []
    receiver_parts: list[np.ndarray] = []
    for value in np.unique(sector):
        members = np.flatnonzero(sector == value).astype(np.int32)
        if len(members) <= 1:
            continue
        sender = np.repeat(members, len(members))
        receiver = np.tile(members, len(members))
        non_loop = sender != receiver
        sender_parts.append(sender[non_loop])
        receiver_parts.append(receiver[non_loop])
    sender = (
        np.concatenate(sender_parts) if sender_parts else np.empty(0, dtype=np.int32)
    )
    receiver = (
        np.concatenate(receiver_parts)
        if receiver_parts
        else np.empty(0, dtype=np.int32)
    )
    return normalized_graph(len(sector), sender, receiver)


@dataclass
class PreparedGraphSplit:
    """A normal prepared split plus one normalized graph per date."""

    data: PreparedSplit
    graphs: tuple[NormalizedGraph, ...]
    relation: str = "sector"

    def __post_init__(self) -> None:
        if len(self.graphs) != len(self.data.dates):
            raise ValueError("There must be exactly one graph per split date")
        node_count = self.data.y.shape[1]
        if any(graph.node_count != node_count for graph in self.graphs):
            raise ValueError("Every graph must align with the split node axis")


def with_sector_graphs(split: PreparedSplit) -> PreparedGraphSplit:
    """Attach same-sector daily graphs to a prepared tensor split."""

    return PreparedGraphSplit(
        data=split,
        graphs=tuple(sector_graph(day) for day in split.sector_index),
        relation="sector",
    )


@dataclass
class GCNParameters:
    """Parameters of the two-layer GCN classifier."""

    numeric_weight: np.ndarray
    sector_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def copy(self) -> GCNParameters:
        return GCNParameters(*(value.copy() for value in self.arrays()))

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.numeric_weight,
            self.sector_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
        )


@dataclass
class GCNForwardCache:
    x_numeric: np.ndarray
    sector_index: np.ndarray
    graph: NormalizedGraph
    hidden_pre_activation: np.ndarray
    hidden: np.ndarray


@dataclass
class GCNAdamState:
    first: GCNParameters
    second: GCNParameters
    step: int = 0


def initialize_gcn(
    numeric_feature_count: int,
    sector_category_count: int,
    hidden_dim: int,
    class_count: int = 5,
    *,
    seed: int = 0,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> GCNParameters:
    """Glorot-initialize a deterministic two-layer GCN."""

    for name, value in (
        ("numeric_feature_count", numeric_feature_count),
        ("sector_category_count", sector_category_count),
        ("hidden_dim", hidden_dim),
        ("class_count", class_count),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    rng = np.random.default_rng(seed)
    input_dim = numeric_feature_count + sector_category_count
    input_limit = math.sqrt(6.0 / (input_dim + hidden_dim))
    output_limit = math.sqrt(6.0 / (hidden_dim + class_count))
    return GCNParameters(
        numeric_weight=rng.uniform(
            -input_limit, input_limit, size=(numeric_feature_count, hidden_dim)
        ).astype(dtype),
        sector_weight=rng.uniform(
            -input_limit, input_limit, size=(sector_category_count, hidden_dim)
        ).astype(dtype),
        hidden_bias=np.zeros(hidden_dim, dtype=dtype),
        output_weight=rng.uniform(
            -output_limit, output_limit, size=(hidden_dim, class_count)
        ).astype(dtype),
        output_bias=np.zeros(class_count, dtype=dtype),
    )


def gcn_forward(
    parameters: GCNParameters,
    x_numeric: np.ndarray,
    sector_index: np.ndarray,
    graph: NormalizedGraph,
) -> tuple[np.ndarray, GCNForwardCache]:
    """Compute logits for one daily graph and retain a backpropagation cache."""

    x = np.asarray(x_numeric)
    sector = np.asarray(sector_index, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != graph.node_count:
        raise ValueError("x_numeric must have shape [graph nodes, numeric features]")
    if sector.shape != (graph.node_count,):
        raise ValueError("sector_index must align with graph nodes")
    if len(sector) and (
        sector.min() < 0 or sector.max() >= parameters.sector_weight.shape[0]
    ):
        raise ValueError("sector_index is outside the embedding vocabulary")
    transformed = x @ parameters.numeric_weight + parameters.sector_weight[sector]
    hidden_pre_activation = graph.aggregate(transformed) + parameters.hidden_bias
    hidden = np.maximum(hidden_pre_activation, 0)
    logits = graph.aggregate(hidden @ parameters.output_weight) + parameters.output_bias
    return logits, GCNForwardCache(
        x_numeric=x,
        sector_index=sector,
        graph=graph,
        hidden_pre_activation=hidden_pre_activation,
        hidden=hidden,
    )


def gcn_logits(
    parameters: GCNParameters,
    x_numeric: np.ndarray,
    sector_index: np.ndarray,
    graph: NormalizedGraph,
) -> np.ndarray:
    return gcn_forward(parameters, x_numeric, sector_index, graph)[0]


def _zeros_like(parameters: GCNParameters) -> GCNParameters:
    return GCNParameters(*(np.zeros_like(value) for value in parameters.arrays()))


def _add_gradients(destination: GCNParameters, source: GCNParameters) -> None:
    for left, right in zip(destination.arrays(), source.arrays(), strict=True):
        left += right


def gcn_backward(
    parameters: GCNParameters,
    cache: GCNForwardCache,
    logits_gradient: np.ndarray,
) -> GCNParameters:
    """Backpropagate through one daily graph."""

    gradient = np.asarray(logits_gradient)
    if gradient.shape != (cache.graph.node_count, parameters.output_bias.size):
        raise ValueError("logits_gradient does not align with the cached logits")
    output_bias_gradient = gradient.sum(axis=0)
    projected_hidden_gradient = cache.graph.aggregate(gradient, transpose=True)
    output_weight_gradient = cache.hidden.T @ projected_hidden_gradient
    hidden_gradient = projected_hidden_gradient @ parameters.output_weight.T
    hidden_pre_gradient = hidden_gradient * (cache.hidden_pre_activation > 0)
    hidden_bias_gradient = hidden_pre_gradient.sum(axis=0)
    transformed_gradient = cache.graph.aggregate(hidden_pre_gradient, transpose=True)
    numeric_weight_gradient = cache.x_numeric.T @ transformed_gradient
    sector_weight_gradient = np.zeros_like(parameters.sector_weight)
    np.add.at(
        sector_weight_gradient,
        cache.sector_index,
        transformed_gradient,
    )
    return GCNParameters(
        numeric_weight_gradient,
        sector_weight_gradient,
        hidden_bias_gradient,
        output_weight_gradient,
        output_bias_gradient,
    )


def daily_cross_entropy_error(
    logits: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    date_weight: float = 1.0,
) -> tuple[float, np.ndarray, float, int]:
    """Return exact daily CE and its logits gradient for one graph."""

    logits = np.asarray(logits)
    target = np.asarray(target)
    mask = np.asarray(mask)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [nodes, classes]")
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("logits must use a floating dtype")
    if target.shape != (logits.shape[0],) or mask.shape != target.shape:
        raise ValueError("target and mask must align with the logits node axis")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    if date_weight < 0 or not np.isfinite(date_weight):
        raise ValueError("date_weight must be finite and non-negative")
    selected_target = target[mask].astype(np.int64, copy=False)
    if not len(selected_target):
        raise ValueError("A supervised date must contain at least one active node")
    if selected_target.min() < 0 or selected_target.max() >= logits.shape[1]:
        raise ValueError("Active target contains an invalid class")
    probabilities, negative_log_likelihood = softmax_and_true_class_nll(
        logits[mask], selected_target
    )
    accuracy = float((probabilities.argmax(axis=1) == selected_target).mean())
    error = np.zeros_like(logits)
    selected_error = probabilities
    selected_error[np.arange(len(selected_target)), selected_target] -= 1.0
    selected_error *= date_weight / len(selected_target)
    error[mask] = selected_error
    return (
        date_weight * float(negative_log_likelihood.mean()),
        error,
        accuracy,
        len(selected_target),
    )


def batch_objective_and_gradients(
    parameters: GCNParameters,
    split: PreparedGraphSplit,
    date_indices: np.ndarray | Iterable[int],
    *,
    l2: float,
) -> tuple[float, GCNParameters]:
    """Compute a date-equal batch objective and exact gradients."""

    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    indices = np.asarray(list(date_indices), dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("date_indices must contain at least one date")
    if indices.min() < 0 or indices.max() >= len(split.data.dates):
        raise ValueError("date_indices contains an out-of-range date")
    active = split.data.loss_mask[indices].any(axis=1)
    indices = indices[active]
    if not len(indices):
        raise ValueError("Selected dates contain no supervised nodes")

    gradients = _zeros_like(parameters)
    data_loss = 0.0
    date_weight = 1.0 / len(indices)
    for date_index in indices:
        logits, cache = gcn_forward(
            parameters,
            split.data.x_numeric[date_index],
            split.data.sector_index[date_index],
            split.graphs[date_index],
        )
        loss, logits_gradient, _, _ = daily_cross_entropy_error(
            logits,
            split.data.y[date_index],
            split.data.loss_mask[date_index],
            date_weight=date_weight,
        )
        data_loss += loss
        _add_gradients(
            gradients,
            gcn_backward(parameters, cache, logits_gradient),
        )

    penalty = (
        0.5
        * l2
        * float(
            np.square(parameters.numeric_weight).sum()
            + np.square(parameters.sector_weight).sum()
            + np.square(parameters.output_weight).sum()
        )
    )
    gradients.numeric_weight += l2 * parameters.numeric_weight
    gradients.sector_weight += l2 * parameters.sector_weight
    gradients.output_weight += l2 * parameters.output_weight
    return data_loss + penalty, gradients


def strict_loss_accuracy(
    parameters: GCNParameters,
    split: PreparedGraphSplit,
) -> tuple[float, float, int]:
    """Evaluate uncapped daily-equal CE directly from logits."""

    losses: list[float] = []
    accuracies: list[float] = []
    node_count = 0
    for date_index, graph in enumerate(split.graphs):
        mask = split.data.loss_mask[date_index]
        if not mask.any():
            continue
        logits = gcn_logits(
            parameters,
            split.data.x_numeric[date_index],
            split.data.sector_index[date_index],
            graph,
        )
        target = split.data.y[date_index, mask].astype(np.int64, copy=False)
        probabilities, negative_log_likelihood = softmax_and_true_class_nll(
            logits[mask], target
        )
        if not np.isfinite(negative_log_likelihood).all():
            raise RuntimeError("Non-finite exact GCN cross-entropy")
        losses.append(float(negative_log_likelihood.mean()))
        accuracies.append(float((probabilities.argmax(axis=1) == target).mean()))
        node_count += len(target)
    if not losses:
        raise RuntimeError("Split contains no model-ready labels")
    return float(np.mean(losses)), float(np.mean(accuracies)), node_count


def _predict_probabilities(
    parameters: GCNParameters,
    split: PreparedGraphSplit,
    *,
    collect_exact_metrics: bool,
) -> tuple[np.ndarray, tuple[float, float, int] | None]:
    """Run one forward pass per graph and optionally collect exact loss."""

    class_count = parameters.output_bias.size
    output = np.empty((*split.data.y.shape, class_count), dtype=np.float32)
    daily_losses: list[float] = []
    daily_accuracies: list[float] = []
    evaluated_nodes = 0
    for date_index, graph in enumerate(split.graphs):
        logits = gcn_logits(
            parameters,
            split.data.x_numeric[date_index],
            split.data.sector_index[date_index],
            graph,
        )
        probabilities = stable_softmax(logits)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("GCN produced a non-finite probability")
        output[date_index] = probabilities.astype(np.float32, copy=False)
        if collect_exact_metrics:
            mask = split.data.loss_mask[date_index]
            if mask.any():
                target = split.data.y[date_index, mask].astype(np.int64, copy=False)
                selected_logits = logits[mask]
                shifted = selected_logits - selected_logits.max(axis=1, keepdims=True)
                normalizer = np.exp(shifted).sum(axis=1)
                negative_log_likelihood = (
                    np.log(normalizer) - shifted[np.arange(len(target)), target]
                )
                if not np.isfinite(negative_log_likelihood).all():
                    raise RuntimeError("Non-finite exact GCN cross-entropy")
                daily_losses.append(float(negative_log_likelihood.mean()))
                daily_accuracies.append(
                    float((probabilities[mask].argmax(axis=1) == target).mean())
                )
                evaluated_nodes += len(target)
    if not np.allclose(output.sum(axis=2), 1.0, atol=2e-6):
        raise RuntimeError("GCN probability rows do not sum to one")
    exact_metrics: tuple[float, float, int] | None = None
    if collect_exact_metrics:
        if not daily_losses:
            raise RuntimeError("Split contains no model-ready labels")
        exact_metrics = (
            float(np.mean(daily_losses)),
            float(np.mean(daily_accuracies)),
            evaluated_nodes,
        )
    return output, exact_metrics


def predict_probabilities(
    parameters: GCNParameters,
    split: PreparedGraphSplit,
) -> np.ndarray:
    """Predict five-class probabilities while preserving daily node axes."""

    return _predict_probabilities(parameters, split, collect_exact_metrics=False)[0]


def predict_probabilities_with_metrics(
    parameters: GCNParameters,
    split: PreparedGraphSplit,
) -> tuple[np.ndarray, tuple[float, float, int]]:
    """Predict and compute exact daily CE without a second graph forward pass."""

    probabilities, metrics = _predict_probabilities(
        parameters, split, collect_exact_metrics=True
    )
    assert metrics is not None
    return probabilities, metrics


def initialize_adam(parameters: GCNParameters) -> GCNAdamState:
    return GCNAdamState(_zeros_like(parameters), _zeros_like(parameters))


def adam_update(
    parameters: GCNParameters,
    gradients: GCNParameters,
    state: GCNAdamState,
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    """Apply one shared-step Adam update to every parameter array."""

    state.step += 1
    for parameter, gradient, first, second in zip(
        parameters.arrays(),
        gradients.arrays(),
        state.first.arrays(),
        state.second.arrays(),
        strict=True,
    ):
        first *= beta1
        first += (1.0 - beta1) * gradient
        second *= beta2
        second += (1.0 - beta2) * np.square(gradient)
        first_hat = first / (1.0 - beta1**state.step)
        second_hat = second / (1.0 - beta2**state.step)
        parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)


def clip_gradients(gradients: GCNParameters, max_norm: float | None) -> float:
    """Clip by global L2 norm and return the pre-clipping norm."""

    squared_norm = sum(
        float(np.square(value.astype(np.float64, copy=False)).sum())
        for value in gradients.arrays()
    )
    norm = math.sqrt(squared_norm)
    if max_norm is not None:
        if max_norm <= 0:
            raise ValueError("max_norm must be positive when provided")
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            for value in gradients.arrays():
                value *= scale
    return norm


def train_one_epoch(
    parameters: GCNParameters,
    state: GCNAdamState,
    split: PreparedGraphSplit,
    *,
    rng: np.random.Generator,
    learning_rate: float,
    l2: float,
    batch_dates: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    gradient_clip_norm: float | None = None,
) -> float:
    """Take one shuffled pass over active dates."""

    if batch_dates <= 0:
        raise ValueError("batch_dates must be positive")
    active_dates = np.flatnonzero(split.data.loss_mask.any(axis=1))
    if not len(active_dates):
        raise RuntimeError("Training split contains no active dates")
    order = rng.permutation(active_dates)
    batch_count = math.ceil(len(order) / batch_dates)
    objective_sum = 0.0
    objective_dates = 0
    for batch in np.array_split(order, batch_count):
        objective, gradients = batch_objective_and_gradients(
            parameters, split, batch, l2=l2
        )
        clip_gradients(gradients, gradient_clip_norm)
        adam_update(
            parameters,
            gradients,
            state,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        if not all(np.isfinite(value).all() for value in parameters.arrays()):
            raise RuntimeError("Non-finite GCN parameter encountered during training")
        objective_sum += objective * len(batch)
        objective_dates += len(batch)
    return objective_sum / objective_dates


def fit_candidate(
    train: PreparedGraphSplit,
    validation: PreparedGraphSplit,
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
) -> tuple[GCNParameters, dict[str, Any]]:
    """Train one hyperparameter candidate with validation-only early stopping."""

    parameters = initialize_gcn(
        int(config["numeric_feature_count"]),
        int(config["sector_category_count"]),
        int(candidate["hidden_dim"]),
        int(config["class_count"]),
        seed=seed,
    )
    state = initialize_adam(parameters)
    rng = np.random.default_rng(seed)
    best_parameters: GCNParameters | None = None
    best_loss = float("inf")
    best_epoch = 0
    patience_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        training_objective = train_one_epoch(
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
            gradient_clip_norm=(
                float(config["gradient_clip_norm"])
                if config.get("gradient_clip_norm") is not None
                else None
            ),
        )
        validation_loss, validation_accuracy, validation_nodes = strict_loss_accuracy(
            parameters, validation
        )
        history.append(
            {
                "epoch": epoch,
                "mean_batch_training_objective": training_objective,
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
        raise RuntimeError("No validation GCN checkpoint was selected")
    return best_parameters, {
        "candidate_id": str(candidate["id"]),
        "hidden_dim": int(candidate["hidden_dim"]),
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


def refit_model(
    refit: PreparedGraphSplit,
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    epochs: int,
    seed: int,
) -> tuple[GCNParameters, dict[str, Any]]:
    """Reinitialize and train the selected model on train plus validation."""

    parameters = initialize_gcn(
        int(config["numeric_feature_count"]),
        int(config["sector_category_count"]),
        int(candidate["hidden_dim"]),
        int(config["class_count"]),
        seed=seed,
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
            gradient_clip_norm=(
                float(config["gradient_clip_norm"])
                if config.get("gradient_clip_norm") is not None
                else None
            ),
        )
        history.append({"epoch": epoch, "mean_batch_training_objective": objective})
    return parameters, {
        "seed": seed,
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


def parameter_count(parameters: GCNParameters) -> int:
    return sum(value.size for value in parameters.arrays())


def save_model(
    path: Path,
    parameters: GCNParameters,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "numeric_weight": parameters.numeric_weight,
        "sector_weight": parameters.sector_weight,
        "hidden_bias": parameters.hidden_bias,
        "output_weight": parameters.output_weight,
        "output_bias": parameters.output_bias,
        "class_values": CLASS_VALUES.astype(np.int8),
    }
    for key, value in (metadata or {}).items():
        payload[key] = np.asarray(value)
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def load_model(path: Path) -> GCNParameters:
    with np.load(path, allow_pickle=False) as checkpoint:
        if not np.array_equal(checkpoint["class_values"], CLASS_VALUES.astype(np.int8)):
            raise RuntimeError(f"Unexpected class mapping in {path}")
        return GCNParameters(
            checkpoint["numeric_weight"].astype(np.float32, copy=True),
            checkpoint["sector_weight"].astype(np.float32, copy=True),
            checkpoint["hidden_bias"].astype(np.float32, copy=True),
            checkpoint["output_weight"].astype(np.float32, copy=True),
            checkpoint["output_bias"].astype(np.float32, copy=True),
        )
