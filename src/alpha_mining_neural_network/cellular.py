"""Construct proposal 1-cells from standardized TOP500 features."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STANDARD_DIR = REPOSITORY_ROOT / "data" / "standard" / "standardized_features"
DEFAULT_UNIVERSE = REPOSITORY_ROOT / "data" / "TOP500" / "top500_universe_21d.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "cellular"
DEFAULT_K = (3, 5, 10, 20)
DEFAULT_TAU = (0.2, 0.4, 0.6, 0.8)
WINDOW = 120


def tau_name(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def load_standard_features(path: Path) -> pd.DataFrame:
    columns = [
        "date",
        "permno",
        "rank",
        "return_t_minus_1_z",
        "beta_60_z",
        "sector_index",
    ]
    if not path.exists():
        raise FileNotFoundError(f"Standardized feature directory not found: {path}")
    frame = pd.read_parquet(path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["date", "permno"]).any():
        raise RuntimeError("Standardized features contain duplicate node keys")
    counts = frame.groupby("date", sort=False).size()
    if (counts != 500).any():
        raise RuntimeError("Each TOP500 date must contain exactly 500 nodes")
    return frame


def prepare_node_arrays(
    frame: pd.DataFrame,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates())
    assets = np.sort(frame["permno"].unique().astype(np.int32, copy=False))
    date_codes = pd.Categorical(frame["date"], categories=dates, ordered=True).codes
    asset_codes = np.searchsorted(assets, frame["permno"].to_numpy()).astype(np.int32)
    node_permnos = np.empty((len(dates), 500), dtype=np.int32)
    node_assets = np.empty((len(dates), 500), dtype=np.int32)
    asset_return = np.full((len(assets), len(dates)), np.nan, dtype=np.float64)
    asset_beta = np.full((len(assets), len(dates)), np.nan, dtype=np.float64)
    asset_sector = np.full((len(assets), len(dates)), -1, dtype=np.int32)
    for _, group in frame.groupby("date", sort=False):
        date_code = int(date_codes[group.index[0]])
        positions = group["rank"].to_numpy(dtype=np.int64) - 1
        if not np.array_equal(np.sort(positions), np.arange(500)):
            raise RuntimeError(f"Ranks are not contiguous on {date_code}")
        node_permnos[date_code, positions] = group["permno"].to_numpy(dtype=np.int32)
        node_assets[date_code, positions] = asset_codes[group.index.to_numpy()]
        group_assets = asset_codes[group.index.to_numpy()]
        asset_return[group_assets, date_code] = group["return_t_minus_1_z"].to_numpy(
            dtype=np.float64
        )
        asset_beta[group_assets, date_code] = group["beta_60_z"].to_numpy(dtype=np.float64)
        asset_sector[group_assets, date_code] = group["sector_index"].fillna(-1).to_numpy(
            dtype=np.int32
        )
    return dates, assets, node_permnos, node_assets, asset_return, asset_beta, asset_sector


def correlation_matrix(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Pearson correlation for complete rows, returning NaN otherwise."""
    if right is None:
        right = left
    valid_left = np.isfinite(left).all(axis=1)
    valid_right = np.isfinite(right).all(axis=1)
    result = np.full((left.shape[0], right.shape[0]), np.nan, dtype=np.float64)
    if not valid_left.any() or not valid_right.any():
        return result
    x = left[valid_left]
    y = right[valid_right]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x_norm = np.sqrt(np.square(x).sum(axis=1))
    y_norm = np.sqrt(np.square(y).sum(axis=1))
    denominator = x_norm[:, None] * y_norm[None, :]
    valid = denominator > 0
    correlation = np.full_like(denominator, np.nan)
    correlation[valid] = (x @ y.T)[valid] / denominator[valid]
    result[np.ix_(valid_left, valid_right)] = correlation
    return result


def undirected_top_k(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.abs(matrix).copy()
    np.fill_diagonal(score, -np.inf)
    pairs: set[tuple[int, int]] = set()
    n = matrix.shape[0]
    for source in range(n):
        valid = np.flatnonzero(np.isfinite(score[source]))
        if not len(valid):
            continue
        take = min(k, len(valid))
        selected = valid[np.argpartition(score[source, valid], -take)[-take:]]
        for target in selected:
            if source != target:
                pairs.add((min(source, int(target)), max(source, int(target))))
    if not pairs:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16), np.array([], dtype=np.float32)
    pair_array = np.asarray(sorted(pairs), dtype=np.int16)
    source = np.concatenate((pair_array[:, 0], pair_array[:, 1]))
    target = np.concatenate((pair_array[:, 1], pair_array[:, 0]))
    weight = np.concatenate((matrix[pair_array[:, 0], pair_array[:, 1]], matrix[pair_array[:, 0], pair_array[:, 1]]))
    return source, target, weight.astype(np.float32)


def undirected_threshold(matrix: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.triu(np.isfinite(matrix) & (np.abs(matrix) >= tau), k=1)
    source, target = np.where(valid)
    if not len(source):
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16), np.array([], dtype=np.float32)
    weight = matrix[source, target].astype(np.float32)
    return (
        np.concatenate((source, target)).astype(np.int16),
        np.concatenate((target, source)).astype(np.int16),
        np.concatenate((weight, weight)),
    )


def directed_top_k(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.abs(matrix)
    np.fill_diagonal(score, -np.inf)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for source in range(matrix.shape[0]):
        valid = np.flatnonzero(np.isfinite(score[source]))
        if not len(valid):
            continue
        take = min(k, len(valid))
        target = valid[np.argpartition(score[source, valid], -take)[-take:]]
        source_parts.append(np.full(len(target), source, dtype=np.int16))
        target_parts.append(target.astype(np.int16))
        weight_parts.append(matrix[source, target].astype(np.float32))
    if not source_parts:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16), np.array([], dtype=np.float32)
    return np.concatenate(source_parts), np.concatenate(target_parts), np.concatenate(weight_parts)


def directed_threshold(matrix: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix) & (np.abs(matrix) >= tau)
    np.fill_diagonal(valid, False)
    source, target = np.where(valid)
    return source.astype(np.int16), target.astype(np.int16), matrix[source, target].astype(np.float32)


EDGE_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("src_rank", pa.int16()),
        ("dst_rank", pa.int16()),
        ("weight", pa.float32()),
    ]
)


class EdgeWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer = pq.ParquetWriter(path, EDGE_SCHEMA, compression="zstd")
        self.rows = 0

    def write(self, date: pd.Timestamp, source: np.ndarray, target: np.ndarray, weight: np.ndarray) -> None:
        if not len(source):
            return
        table = pa.Table.from_pydict(
            {
                "date": np.full(len(source), date.strftime("%Y-%m-%d"), dtype=object),
                "src_rank": source,
                "dst_rank": target,
                "weight": weight,
            },
            schema=EDGE_SCHEMA,
        )
        self.writer.write_table(table)
        self.rows += len(source)

    def close(self) -> None:
        self.writer.close()


def build_cellular(
    *,
    standard_dir: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = DEFAULT_K,
    tau_values: tuple[float, ...] = DEFAULT_TAU,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace cellular output directory: {output_dir}")
    if any(k <= 0 for k in k_values) or any(k >= 500 for k in k_values):
        raise ValueError("k values must be between 1 and 499")
    if any(not 0 < tau <= 1 for tau in tau_values):
        raise ValueError("tau values must be in (0, 1]")
    frame = load_standard_features(standard_dir)
    dates, assets, node_permnos, node_assets, asset_return, asset_beta, asset_sector = prepare_node_arrays(frame)
    del assets, node_permnos

    temporary = output_dir.parent / f".{output_dir.name}.part"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    writers: dict[str, EdgeWriter] = {}
    for k in k_values:
        for relation in ("corr", "leadlag", "beta"):
            writers[f"{relation}_topk_k{k}"] = EdgeWriter(temporary / f"{relation}_topk_k{k}.parquet")
    for tau in tau_values:
        label = tau_name(tau)
        for relation in ("corr", "leadlag", "beta"):
            writers[f"{relation}_tau_tau{label}"] = EdgeWriter(temporary / f"{relation}_tau_tau{label}.parquet")
    writers["sector_fixed"] = EdgeWriter(temporary / "sector_fixed.parquet")
    writers["industry_fixed"] = EdgeWriter(temporary / "industry_fixed.parquet")

    try:
        for date_index, date in enumerate(dates):
            if date_index < WINDOW:
                continue
            current_assets = node_assets[date_index]
            current_beta = asset_beta[current_assets, date_index]
            current_sector = asset_sector[current_assets, date_index]
            corr_window = asset_return[current_assets, date_index - WINDOW : date_index]
            corr = correlation_matrix(corr_window)
            for k in k_values:
                source, target, weight = undirected_top_k(corr, k)
                writers[f"corr_topk_k{k}"].write(date, source, target, weight)
            for tau in tau_values:
                source, target, weight = undirected_threshold(corr, tau)
                writers[f"corr_tau_tau{tau_name(tau)}"].write(date, source, target, weight)

            source_window = asset_return[current_assets, date_index - 124 : date_index - 4]
            target_window = asset_return[current_assets, date_index - 120 : date_index]
            leadlag = correlation_matrix(source_window, target_window)
            for k in k_values:
                source, target, weight = directed_top_k(leadlag, k)
                writers[f"leadlag_topk_k{k}"].write(date, source, target, weight)
            for tau in tau_values:
                source, target, weight = directed_threshold(leadlag, tau)
                writers[f"leadlag_tau_tau{tau_name(tau)}"].write(date, source, target, weight)

            beta_similarity = np.full((500, 500), np.nan, dtype=np.float64)
            valid_beta = np.isfinite(current_beta) & (current_beta != 0)
            beta_similarity[np.ix_(valid_beta, valid_beta)] = np.sign(
                current_beta[valid_beta, None] * current_beta[None, valid_beta]
            )
            for k in k_values:
                source, target, weight = undirected_top_k(beta_similarity, k)
                writers[f"beta_topk_k{k}"].write(date, source, target, weight)
            for tau in tau_values:
                source, target, weight = undirected_threshold(beta_similarity, tau)
                writers[f"beta_tau_tau{tau_name(tau)}"].write(date, source, target, weight)

            same_sector = np.equal.outer(current_sector, current_sector)
            same_sector &= (current_sector[:, None] >= 0) & (current_sector[None, :] >= 0)
            np.fill_diagonal(same_sector, False)
            source, target = np.where(np.triu(same_sector, k=1))
            weights = np.ones(len(source), dtype=np.float32)
            writers["sector_fixed"].write(
                date,
                np.concatenate((source, target)).astype(np.int16),
                np.concatenate((target, source)).astype(np.int16),
                np.concatenate((weights, weights)),
            )
            writers["industry_fixed"].write(
                date,
                np.concatenate((source, target)).astype(np.int16),
                np.concatenate((target, source)).astype(np.int16),
                np.concatenate((weights, weights)),
            )
            if date_index % 250 == 0:
                print(f"Cellular dates: {date_index:,}/{len(dates):,}", flush=True)
        for writer in writers.values():
            writer.close()
        temporary.replace(output_dir)
    except Exception:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    files: list[dict[str, object]] = []
    for name, writer in writers.items():
        path = output_dir / f"{name}.parquet"
        files.append({"name": name, "path": path.name, "edges": writer.rows, "bytes": path.stat().st_size})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(standard_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "relation_edge_schema": ["date", "src_rank", "dst_rank", "weight"],
        "node_reference": "TOP500 rank is zero-based in edge files; rank+1 maps to data/TOP500 rank",
        "rolling_window": WINDOW,
        "k_values": list(k_values),
        "tau_values": list(tau_values),
        "correlation_definition": "Pearson correlation of standardized return_t_minus_1_z over prior 120 feature dates",
        "leadlag_definition": "Corr(source t-124:t-5, target t-120:t-1) using standardized return_t_minus_1_z timestamps",
        "beta_similarity_definition": "one-dimensional cosine similarity sign(beta_i * beta_j)",
        "sector_definition": "same point-in-time SICCD sector_index",
        "industry_definition": "SICCD equality proxy; no separate industry mapping is present",
        "unavailable_relations": {"supply_chain": "No supplier-customer mapping exists in available inputs"},
        "files": files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cellular_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build top-k and tau-threshold 1-cells.")
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--tau", type=float, nargs="+", default=list(DEFAULT_TAU))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_cellular(
        standard_dir=args.standard_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        k_values=tuple(args.k),
        tau_values=tuple(args.tau),
    )
    print(f"Cellular files: {len(manifest['files'])}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0
