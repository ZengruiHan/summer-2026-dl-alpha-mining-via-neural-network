"""Evaluate M0 long-short portfolio performance and cost sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
DEFAULT_PORTFOLIO_DIR = REPOSITORY_ROOT / "results" / "portfolios"
DEFAULT_READY_DIR = REPOSITORY_ROOT / "data" / "ready_for_use"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "portfolio_metrics"
DEFAULT_COST_BPS = (0.0, 5.0, 10.0, 20.0)
DEFAULT_HEADLINE_COST_BPS = 10.0
ANNUALIZATION_DAYS = 252


def aligned_turnover(
    current_permno: np.ndarray,
    current_weight: np.ndarray,
    previous_permno: np.ndarray | None,
    previous_weight: np.ndarray | None,
) -> float:
    """Compute sum |w_t-w_{t-1}| by security identifier, not array column."""

    if current_permno.shape != current_weight.shape:
        raise ValueError("Current PERMNO and weight arrays do not align")
    current_nonzero = current_weight != 0.0
    current = {
        int(asset): float(weight)
        for asset, weight in zip(
            current_permno[current_nonzero],
            current_weight[current_nonzero],
            strict=True,
        )
    }
    if previous_permno is None or previous_weight is None:
        previous: dict[int, float] = {}
    else:
        if previous_permno.shape != previous_weight.shape:
            raise ValueError("Previous PERMNO and weight arrays do not align")
        previous_nonzero = previous_weight != 0.0
        previous = {
            int(asset): float(weight)
            for asset, weight in zip(
                previous_permno[previous_nonzero],
                previous_weight[previous_nonzero],
                strict=True,
            )
        }
    assets = current.keys() | previous.keys()
    return float(
        sum(abs(current.get(asset, 0.0) - previous.get(asset, 0.0)) for asset in assets)
    )


def strict_portfolio_return(
    weights: np.ndarray, target_return: np.ndarray
) -> tuple[float, int]:
    """Return NaN unless every nonzero holding has an observed next return."""

    if weights.shape != target_return.shape:
        raise ValueError("Weights and target returns do not align")
    held = weights != 0.0
    missing_count = int((held & ~np.isfinite(target_return)).sum())
    if missing_count:
        return float("nan"), missing_count
    return float(
        np.dot(
            weights[held].astype(np.float64),
            target_return[held].astype(np.float64),
        )
    ), 0


def zero_contribution_portfolio_return(
    weights: np.ndarray, target_return: np.ndarray
) -> tuple[float, int]:
    """Keep original weights and assign zero contribution to missing returns.

    A date remains unavailable only when every nonzero holding return is
    missing.  No asset is removed and no surviving weight is renormalized.
    """

    if weights.shape != target_return.shape:
        raise ValueError("Weights and target returns do not align")
    held = weights != 0.0
    observed = held & np.isfinite(target_return)
    missing_count = int((held & ~np.isfinite(target_return)).sum())
    if not observed.any():
        return float("nan"), missing_count
    return float(
        np.dot(
            weights[observed].astype(np.float64),
            target_return[observed].astype(np.float64),
        )
    ), missing_count


def performance_summary(
    gross_return: np.ndarray,
    turnover: np.ndarray,
    *,
    transaction_cost_bps: float,
    annualization_days: int = ANNUALIZATION_DAYS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if gross_return.shape != turnover.shape:
        raise ValueError("Gross return and turnover arrays do not align")
    if transaction_cost_bps < 0 or not math.isfinite(transaction_cost_bps):
        raise ValueError("Transaction cost must be finite and nonnegative")
    valid = np.isfinite(gross_return)
    cost_rate = transaction_cost_bps / 10_000.0
    net_return = np.full(gross_return.shape, np.nan, dtype=np.float64)
    net_return[valid] = gross_return[valid] - cost_rate * turnover[valid]
    valid_net = net_return[valid]
    if len(valid_net) < 2:
        sharpe = None
    else:
        standard_deviation = float(valid_net.std(ddof=1))
        sharpe = (
            None
            if standard_deviation == 0.0 or not math.isfinite(standard_deviation)
            else float(
                math.sqrt(annualization_days)
                * float(valid_net.mean())
                / standard_deviation
            )
        )
    conditional_wealth = np.full(gross_return.shape, np.nan, dtype=np.float64)
    drawdown = np.full(gross_return.shape, np.nan, dtype=np.float64)
    wealth_level = 1.0
    running_peak = 1.0
    for index in np.flatnonzero(valid):
        daily_return = float(net_return[index])
        if daily_return <= -1.0:
            raise RuntimeError("Net return implies nonpositive wealth")
        wealth_level *= 1.0 + daily_return
        running_peak = max(running_peak, wealth_level)
        conditional_wealth[index] = wealth_level
        drawdown[index] = 1.0 - wealth_level / running_peak
    summary = {
        "transaction_cost_bps": float(transaction_cost_bps),
        "annualization_days": annualization_days,
        "sharpe": sharpe,
        "mean_daily_gross_return": float(gross_return[valid].mean()) if valid.any() else None,
        "mean_daily_gross_return_bps": (
            float(gross_return[valid].mean() * 10_000.0) if valid.any() else None
        ),
        "mean_daily_net_return": float(valid_net.mean()) if len(valid_net) else None,
        "mean_daily_net_return_bps": (
            float(valid_net.mean() * 10_000.0) if len(valid_net) else None
        ),
        "mean_daily_turnover": (
            float(turnover[valid].mean()) if valid.any() else None
        ),
        "mean_daily_turnover_all_formation_dates": float(turnover.mean()),
        "maximum_drawdown": (
            float(np.nanmax(drawdown)) if np.isfinite(drawdown).any() else None
        ),
        "cumulative_net_return": wealth_level - 1.0,
        "calendar_dates": len(gross_return),
        "valid_return_dates": int(valid.sum()),
        "missing_return_dates": int((~valid).sum()),
    }
    return summary, {
        "net_return": net_return,
        "conditional_wealth": conditional_wealth,
        "drawdown": drawdown,
    }


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


def _cost_label(cost_bps: float) -> str:
    text = f"{cost_bps:g}".replace(".", "p")
    return f"{text}bps"


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate_portfolio_metrics(
    *,
    portfolio_dir: Path,
    ready_dir: Path,
    output_dir: Path,
    cost_bps: tuple[float, ...] = DEFAULT_COST_BPS,
    headline_cost_bps: float = DEFAULT_HEADLINE_COST_BPS,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not cost_bps:
        raise ValueError("At least one transaction-cost scenario is required")
    normalized_costs = tuple(sorted(set(float(value) for value in cost_bps)))
    if headline_cost_bps not in normalized_costs:
        raise ValueError("Headline cost must be included in cost scenarios")
    if any(value < 0 or not math.isfinite(value) for value in normalized_costs):
        raise ValueError("Transaction costs must be finite and nonnegative")

    portfolio_manifest_path = portfolio_dir / "portfolio_manifest.json"
    portfolio_manifest = load_json(portfolio_manifest_path)
    if portfolio_manifest.get("status") != "complete" or portfolio_manifest.get("model") != "M0":
        raise RuntimeError("Portfolio input is not a complete M0 artifact")
    dates = np.load(portfolio_dir / "dates.npy", mmap_mode="r", allow_pickle=False)
    permno = np.load(portfolio_dir / "permno.npy", mmap_mode="r", allow_pickle=False)
    fold_index = np.load(
        portfolio_dir / "fold_index.npy", mmap_mode="r", allow_pickle=False
    )
    weights = np.load(
        portfolio_dir / "weights.npy", mmap_mode="r", allow_pickle=False
    )
    expected_node_shape = (len(dates), 500)
    if permno.shape != expected_node_shape or weights.shape != expected_node_shape:
        raise RuntimeError("Portfolio arrays do not align")
    if fold_index.shape != (len(dates),):
        raise RuntimeError("Portfolio fold index does not align")
    if not np.allclose(weights.sum(axis=1), 0.0, atol=2e-7):
        raise RuntimeError("Portfolio weights are not dollar neutral")
    if not np.allclose(np.abs(weights).sum(axis=1), 1.0, atol=2e-6):
        raise RuntimeError("Portfolio weights do not have unit gross exposure")

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
        returns = np.load(
            supervision_dir / "target_return.npy", mmap_mode="r", allow_pickle=False
        )
        return_dates = np.load(
            supervision_dir / "target_date.npy", mmap_mode="r", allow_pickle=False
        )
        stop = offset + len(year_dates)
        if not np.array_equal(dates[offset:stop], year_dates):
            raise RuntimeError(f"Portfolio date alignment failed for {year}")
        if not np.array_equal(permno[offset:stop], year_permno):
            raise RuntimeError(f"Portfolio PERMNO alignment failed for {year}")
        date_grid = np.broadcast_to(year_dates[:, None], returns.shape)
        finite_returns = np.isfinite(returns)
        if not np.all(return_dates[finite_returns] > date_grid[finite_returns]):
            raise RuntimeError(f"Non-forward portfolio target date in {year}")
        target_return[offset:stop] = returns
        target_date[offset:stop] = return_dates
        source_records[str(year)] = {
            "dates_sha256": sha256_file(feature_dir / "dates.npy"),
            "permno_sha256": sha256_file(feature_dir / "permno.npy"),
            "target_return_sha256": sha256_file(
                supervision_dir / "target_return.npy"
            ),
            "target_date_sha256": sha256_file(
                supervision_dir / "target_date.npy"
            ),
        }
        offset = stop
    if offset != len(dates):
        raise RuntimeError("Portfolio dates extend outside full supervision")

    source_payload = {
        "portfolio_manifest_sha256": sha256_file(portfolio_manifest_path),
        "full_supervision_sources": source_records,
        "transaction_cost_bps": normalized_costs,
        "headline_transaction_cost_bps": headline_cost_bps,
        "annualization_days": ANNUALIZATION_DAYS,
        "return_missing_policy": "zero_contribution_without_reweighting",
        "first_day_turnover_policy": "previous portfolio is cash/zero weights",
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    source_fingerprint = _fingerprint(source_payload)
    if output_dir.exists() and not overwrite:
        manifest_path = output_dir / "portfolio_metrics.json"
        if not manifest_path.exists():
            raise RuntimeError("Existing portfolio metric output lacks manifest")
        existing = load_json(manifest_path)
        if existing.get("source_fingerprint") != source_fingerprint:
            raise RuntimeError("Existing portfolio metrics use different inputs or assumptions")
        for filename, record in existing["files"].items():
            path = output_dir / filename
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Existing portfolio metric failed integrity: {path}")
        return existing

    turnover = np.empty(len(dates), dtype=np.float64)
    gross_return = np.full(len(dates), np.nan, dtype=np.float64)
    strict_gross_return = np.full(len(dates), np.nan, dtype=np.float64)
    held_missing_return_count = np.zeros(len(dates), dtype=np.int16)
    daily_target_date = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[D]")
    for date_index in range(len(dates)):
        turnover[date_index] = aligned_turnover(
            permno[date_index],
            weights[date_index],
            None if date_index == 0 else permno[date_index - 1],
            None if date_index == 0 else weights[date_index - 1],
        )
        gross, missing_count = zero_contribution_portfolio_return(
            weights[date_index], target_return[date_index]
        )
        strict_gross, strict_missing_count = strict_portfolio_return(
            weights[date_index], target_return[date_index]
        )
        if strict_missing_count != missing_count:
            raise RuntimeError("Missing-return policies disagree on missing count")
        gross_return[date_index] = gross
        strict_gross_return[date_index] = strict_gross
        held_missing_return_count[date_index] = missing_count
        available_dates = target_date[date_index, np.isfinite(target_return[date_index])]
        if len(available_dates):
            unique_target_dates = np.unique(available_dates)
            if len(unique_target_dates) != 1:
                raise RuntimeError(f"Multiple target dates on signal date {dates[date_index]}")
            daily_target_date[date_index] = unique_target_dates[0]
    if not np.isclose(turnover[0], 1.0, atol=2e-6):
        raise RuntimeError("First-day cash-to-portfolio turnover is not one")
    if not np.isfinite(turnover).all() or (turnover < 0).any():
        raise RuntimeError("Invalid daily turnover")
    return_valid_mask = np.isfinite(gross_return)
    strict_return_valid_mask = np.isfinite(strict_gross_return)
    if not np.array_equal(strict_return_valid_mask, held_missing_return_count == 0):
        raise RuntimeError("Strict return coverage and missing-holding count disagree")
    if not np.all(strict_return_valid_mask <= return_valid_mask):
        raise RuntimeError("Strict return coverage is not a subset of primary coverage")

    scenario_summaries: dict[str, dict[str, Any]] = {}
    scenario_arrays: dict[str, dict[str, np.ndarray]] = {}
    strict_scenario_summaries: dict[str, dict[str, Any]] = {}
    for cost in normalized_costs:
        label = _cost_label(cost)
        summary, arrays = performance_summary(
            gross_return,
            turnover,
            transaction_cost_bps=cost,
        )
        scenario_summaries[label] = summary
        scenario_arrays[label] = arrays
        strict_summary, _ = performance_summary(
            strict_gross_return,
            turnover,
            transaction_cost_bps=cost,
        )
        strict_scenario_summaries[label] = strict_summary

    fold_summaries: list[dict[str, Any]] = []
    for fold in range(17):
        mask = fold_index == fold
        scenarios: dict[str, Any] = {}
        for cost in normalized_costs:
            label = _cost_label(cost)
            summary, _ = performance_summary(
                gross_return[mask],
                turnover[mask],
                transaction_cost_bps=cost,
            )
            scenarios[label] = summary
        fold_summaries.append(
            {
                "fold_index": fold,
                "test_year": 2009 + fold,
                "calendar_dates": int(mask.sum()),
                "return_valid_dates": int(return_valid_mask[mask].sum()),
                "held_missing_return_nodes": int(
                    held_missing_return_count[mask].sum()
                ),
                "scenarios": scenarios,
            }
        )

    daily_rows: list[dict[str, Any]] = []
    for index, date in enumerate(dates):
        row: dict[str, Any] = {
            "date": str(date),
            "target_date": str(daily_target_date[index]),
            "fold_index": int(fold_index[index]),
            "return_valid": bool(return_valid_mask[index]),
            "held_missing_return_count": int(held_missing_return_count[index]),
            "gross_return": repr(float(gross_return[index])),
            "strict_gross_return": repr(float(strict_gross_return[index])),
            "turnover": repr(float(turnover[index])),
        }
        for cost in normalized_costs:
            label = _cost_label(cost)
            arrays = scenario_arrays[label]
            row[f"net_return_{label}"] = repr(float(arrays["net_return"][index]))
            row[f"conditional_wealth_{label}"] = repr(
                float(arrays["conditional_wealth"][index])
            )
            row[f"drawdown_{label}"] = repr(float(arrays["drawdown"][index]))
        daily_rows.append(row)

    fold_rows: list[dict[str, Any]] = []
    for fold_record in fold_summaries:
        for label, scenario in fold_record["scenarios"].items():
            fold_rows.append(
                {
                    "fold_index": fold_record["fold_index"],
                    "test_year": fold_record["test_year"],
                    "transaction_cost_bps": scenario["transaction_cost_bps"],
                    "calendar_dates": fold_record["calendar_dates"],
                    "valid_return_dates": scenario["valid_return_dates"],
                    "missing_return_dates": scenario["missing_return_dates"],
                    "held_missing_return_nodes": fold_record[
                        "held_missing_return_nodes"
                    ],
                    "sharpe": repr(scenario["sharpe"]),
                    "mean_daily_net_return_bps": repr(
                        scenario["mean_daily_net_return_bps"]
                    ),
                    "mean_daily_turnover": repr(scenario["mean_daily_turnover"]),
                    "maximum_drawdown": repr(scenario["maximum_drawdown"]),
                    "cumulative_net_return": repr(
                        scenario["cumulative_net_return"]
                    ),
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
            source = portfolio_dir / filename
            destination = temporary / filename
            method = hardlink_or_copy(source, destination)
            files[filename] = {
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "publication_method": method,
            }
        derived_arrays = {
            "daily_target_date.npy": daily_target_date,
            "return_valid_mask.npy": return_valid_mask,
            "strict_return_valid_mask.npy": strict_return_valid_mask,
            "held_missing_return_count.npy": held_missing_return_count,
            "gross_return.npy": gross_return,
            "strict_gross_return.npy": strict_gross_return,
            "turnover.npy": turnover,
        }
        for label, arrays in scenario_arrays.items():
            derived_arrays[f"net_return_{label}.npy"] = arrays["net_return"]
            derived_arrays[f"conditional_wealth_{label}.npy"] = arrays[
                "conditional_wealth"
            ]
            derived_arrays[f"drawdown_{label}.npy"] = arrays["drawdown"]
        for filename, values in derived_arrays.items():
            path = temporary / filename
            _save_array(path, values)
            files[filename] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }
        daily_fields = list(daily_rows[0])
        fold_fields = list(fold_rows[0])
        daily_csv = temporary / "daily_portfolio_metrics.csv"
        fold_csv = temporary / "fold_portfolio_metrics.csv"
        _write_csv(daily_csv, daily_fields, daily_rows)
        _write_csv(fold_csv, fold_fields, fold_rows)
        for path in (daily_csv, fold_csv):
            files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "derivation": "computed",
            }

        headline_label = _cost_label(headline_cost_bps)
        gross_mean = (
            float(gross_return[return_valid_mask].mean())
            if return_valid_mask.any()
            else None
        )
        turnover_mean = (
            float(turnover[return_valid_mask].mean())
            if return_valid_mask.any()
            else None
        )
        break_even_cost_bps = (
            None
            if gross_mean is None or turnover_mean in (None, 0.0)
            else float(gross_mean / turnover_mean * 10_000.0)
        )
        manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "M0",
            "artifact": "OOS long-short portfolio metrics",
            "source_fingerprint": source_fingerprint,
            "portfolio_input": str(portfolio_dir.resolve()),
            "full_supervision_input": str(
                (ready_dir / "_shared" / "supervision_full").resolve()
            ),
            "headline_transaction_cost_bps": headline_cost_bps,
            "headline": scenario_summaries[headline_label],
            "break_even_one_way_cost_bps": break_even_cost_bps,
            "scenarios": scenario_summaries,
            "strict_complete_date_sensitivity": strict_scenario_summaries,
            "folds": fold_summaries,
            "turnover_definition": (
                "sum over the union of adjacent-day PERMNOs of absolute weight "
                "change; the first date starts from zero/cash weights"
            ),
            "gross_return_definition": "sum_i weight[i,t] * target_return[i,t+1]",
            "net_return_definition": "gross_return - (cost_bps/10000) * turnover",
            "sharpe_definition": "sqrt(252) * mean(net_return) / sample_std(net_return, ddof=1)",
            "maximum_drawdown_definition": "max(1 - conditional_wealth/running_peak), including initial wealth 1",
            "cumulative_net_return_definition": "product over return-valid dates of (1+net_return) minus 1",
            "return_missing_policy": (
                "primary stale-price convention: a missing held return contributes zero "
                "with original weights unchanged and no ex-post reweighting; a date is "
                "unavailable only when every held return is missing"
            ),
            "conditional_path_policy": (
                "wealth, drawdown, Sharpe, net return, and cumulative return use the "
                "primary return-valid dates; unavailable dates remain NaN"
            ),
            "coverage": {
                "calendar_dates": len(dates),
                "valid_return_dates": int(return_valid_mask.sum()),
                "missing_return_dates": int((~return_valid_mask).sum()),
                "valid_return_fraction": float(return_valid_mask.mean()),
                "held_missing_return_nodes": int(held_missing_return_count.sum()),
                "partially_missing_return_dates": int(
                    ((held_missing_return_count > 0) & return_valid_mask).sum()
                ),
                "max_missing_held_nodes_on_one_date": int(
                    held_missing_return_count.max()
                ),
                "strict_complete_holding_dates": int(
                    strict_return_valid_mask.sum()
                ),
            },
            "full_supervision_sources": source_records,
            "files": files,
            "checks": {
                "date_alignment": "passed",
                "permno_alignment": "passed",
                "turnover_permno_union_alignment": "passed",
                "first_day_turnover": 1.0,
                "daily_gross_exposure": 1.0,
                "daily_net_exposure": 0.0,
                "zero_contribution_no_reweight_policy": "passed",
                "strict_complete_date_sensitivity": "passed",
                "scenario_cost_monotonicity": "passed",
            },
        }
        mean_net = [
            scenario_summaries[_cost_label(cost)]["mean_daily_net_return"]
            for cost in normalized_costs
        ]
        if any(
            mean_net[index] < mean_net[index + 1]
            for index in range(len(mean_net) - 1)
        ):
            raise RuntimeError("Net return is not monotone in transaction cost")
        write_json(temporary / "portfolio_metrics.json", manifest)
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M0 OOS long-short portfolio metrics."
    )
    parser.add_argument("--portfolio-dir", type=Path, default=DEFAULT_PORTFOLIO_DIR)
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cost-bps",
        type=float,
        action="append",
        help="One-way cost scenario in bps; repeat. Defaults to 0,5,10,20.",
    )
    parser.add_argument("--headline-cost-bps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = evaluate_portfolio_metrics(
        portfolio_dir=args.portfolio_dir.expanduser().resolve(),
        ready_dir=args.ready_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        cost_bps=tuple(args.cost_bps) if args.cost_bps else DEFAULT_COST_BPS,
        headline_cost_bps=args.headline_cost_bps,
        overwrite=args.overwrite,
    )
    headline = manifest["headline"]
    print(
        f"Headline ({manifest['headline_transaction_cost_bps']:g} bps): "
        f"Sharpe={headline['sharpe']:.6f}, "
        f"net={headline['mean_daily_net_return_bps']:.6f} bps/day, "
        f"turnover={headline['mean_daily_turnover']:.6f}, "
        f"MDD={headline['maximum_drawdown']:.6f}, "
        f"cumulative={headline['cumulative_net_return']:.6f}",
        flush=True,
    )
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
