#!/usr/bin/env python
"""Draw the proposal-defined M0 OOS diagnostic figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO_METRICS = REPOSITORY_ROOT / "results" / "portfolio_metrics"
DEFAULT_PREDICTION_METRICS = REPOSITORY_ROOT / "results" / "prediction_metrics"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "figures"
COLORS = {
    "blue": "#2F6690",
    "light_blue": "#81B1D2",
    "green": "#2A9D8F",
    "red": "#D1495B",
    "orange": "#E9A23B",
    "gray": "#5E6472",
    "grid": "#D9DEE7",
}


@dataclass(frozen=True)
class PlotData:
    dates: np.ndarray
    realization_dates: np.ndarray
    fold_index: np.ndarray
    net_return: np.ndarray
    wealth: np.ndarray
    test_years: np.ndarray
    boundary_dates: np.ndarray
    fold_sharpe: np.ndarray
    fold_rank_ic: np.ndarray
    fold_mdd: np.ndarray
    fold_standalone_return: np.ndarray
    fold_wealth_contribution: np.ndarray
    transaction_cost_bps: float
    overall_cumulative_return: float
    overall_mdd: float
    source_assumption: str


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cost_label(cost_bps: float) -> str:
    return f"{cost_bps:g}".replace(".", "p") + "bps"


def compute_fold_contributions(
    net_return: np.ndarray, fold_index: np.ndarray, fold_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return standalone fold returns and additive final-wealth contributions."""

    if net_return.shape != fold_index.shape:
        raise ValueError("Net-return and fold-index arrays must align")
    standalone = np.empty(fold_count, dtype=np.float64)
    contribution = np.empty(fold_count, dtype=np.float64)
    wealth_before = 1.0
    for fold in range(fold_count):
        values = net_return[(fold_index == fold) & np.isfinite(net_return)]
        if len(values) == 0:
            raise RuntimeError(f"Fold {fold} has no valid net returns")
        factor = float(np.prod(1.0 + values))
        standalone[fold] = factor - 1.0
        contribution[fold] = wealth_before * (factor - 1.0)
        wealth_before *= factor
    if not np.isclose(contribution.sum(), wealth_before - 1.0, atol=1e-12):
        raise RuntimeError("Fold contributions do not add to final cumulative return")
    return standalone, contribution


def load_plot_data(
    portfolio_metric_dir: Path,
    prediction_metric_dir: Path,
) -> tuple[PlotData, dict[str, Any]]:
    portfolio_manifest_path = portfolio_metric_dir / "portfolio_metrics.json"
    prediction_manifest_path = prediction_metric_dir / "prediction_metrics.json"
    portfolio_manifest = load_json(portfolio_manifest_path)
    prediction_manifest = load_json(prediction_manifest_path)
    if portfolio_manifest.get("status") != "complete":
        raise RuntimeError("Portfolio metrics are incomplete")
    if prediction_manifest.get("status") != "complete":
        raise RuntimeError("Prediction metrics are incomplete")
    if portfolio_manifest.get("model") != "M0" or prediction_manifest.get("model") != "M0":
        raise RuntimeError("Plot inputs are not M0 artifacts")

    cost_bps = float(portfolio_manifest["headline_transaction_cost_bps"])
    label = cost_label(cost_bps)
    dates = np.load(
        portfolio_metric_dir / "dates.npy", mmap_mode="r", allow_pickle=False
    )
    fold_index = np.load(
        portfolio_metric_dir / "fold_index.npy", mmap_mode="r", allow_pickle=False
    )
    net_return = np.load(
        portfolio_metric_dir / f"net_return_{label}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    wealth = np.load(
        portfolio_metric_dir / f"conditional_wealth_{label}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    realization_dates = np.load(
        portfolio_metric_dir / "daily_target_date.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if not (
        dates.shape
        == realization_dates.shape
        == fold_index.shape
        == net_return.shape
        == wealth.shape
    ):
        raise RuntimeError("Daily plot arrays do not align")
    if not np.array_equal(np.unique(fold_index), np.arange(17, dtype=np.int8)):
        raise RuntimeError("Expected exactly 17 ordered OOS folds")

    test_years = np.arange(2009, 2026, dtype=np.int16)
    boundary_dates = np.empty(17, dtype="datetime64[D]")
    fold_sharpe = np.empty(17, dtype=np.float64)
    fold_rank_ic = np.empty(17, dtype=np.float64)
    fold_mdd = np.empty(17, dtype=np.float64)
    prediction_by_fold = {
        int(record["fold_index"]): record for record in prediction_manifest["folds"]
    }
    portfolio_by_fold = {
        int(record["fold_index"]): record for record in portfolio_manifest["folds"]
    }
    for fold in range(17):
        positions = np.flatnonzero(fold_index == fold)
        if not len(positions):
            raise RuntimeError(f"Missing fold {fold}")
        valid_positions = positions[np.isfinite(net_return[positions])]
        if not len(valid_positions):
            raise RuntimeError(f"Fold {fold} has no realized return dates")
        boundary_dates[fold] = realization_dates[valid_positions[0]]
        portfolio_record = portfolio_by_fold[fold]
        prediction_record = prediction_by_fold[fold]
        if int(portfolio_record["test_year"]) != 2009 + fold:
            raise RuntimeError("Portfolio fold/year mapping mismatch")
        if int(prediction_record["test_year"]) != 2009 + fold:
            raise RuntimeError("Prediction fold/year mapping mismatch")
        scenario = portfolio_record["scenarios"][label]
        fold_sharpe[fold] = float(scenario["sharpe"])
        fold_mdd[fold] = float(scenario["maximum_drawdown"])
        fold_rank_ic[fold] = float(prediction_record["rank_ic"])

    standalone, contribution = compute_fold_contributions(
        net_return, fold_index, fold_count=17
    )
    for fold in range(17):
        expected = float(portfolio_by_fold[fold]["scenarios"][label]["cumulative_net_return"])
        if not np.isclose(standalone[fold], expected, atol=1e-12):
            raise RuntimeError(f"Fold cumulative return mismatch for fold {fold}")
    headline = portfolio_manifest["headline"]
    overall_cumulative = float(headline["cumulative_net_return"])
    if not np.isclose(contribution.sum(), overall_cumulative, atol=1e-12):
        raise RuntimeError("Fold contributions do not match overall cumulative return")
    finite_wealth = wealth[np.isfinite(wealth)]
    if not len(finite_wealth) or not np.isclose(
        finite_wealth[-1] - 1.0, overall_cumulative, atol=1e-12
    ):
        raise RuntimeError("Wealth endpoint does not match cumulative return")

    data = PlotData(
        dates=np.asarray(dates),
        realization_dates=np.asarray(realization_dates),
        fold_index=np.asarray(fold_index),
        net_return=np.asarray(net_return),
        wealth=np.asarray(wealth),
        test_years=test_years,
        boundary_dates=boundary_dates,
        fold_sharpe=fold_sharpe,
        fold_rank_ic=fold_rank_ic,
        fold_mdd=fold_mdd,
        fold_standalone_return=standalone,
        fold_wealth_contribution=contribution,
        transaction_cost_bps=cost_bps,
        overall_cumulative_return=overall_cumulative,
        overall_mdd=float(headline["maximum_drawdown"]),
        source_assumption=portfolio_manifest["return_missing_policy"],
    )
    provenance = {
        "portfolio_metrics": str(portfolio_manifest_path.resolve()),
        "portfolio_metrics_sha256": sha256_file(portfolio_manifest_path),
        "prediction_metrics": str(prediction_manifest_path.resolve()),
        "prediction_metrics_sha256": sha256_file(prediction_manifest_path),
        "headline_transaction_cost_bps": cost_bps,
    }
    return data, provenance


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["gray"],
            "axes.labelcolor": "#20242B",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 10,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def add_fold_boundaries(ax: plt.Axes, data: PlotData, *, label_folds: bool) -> None:
    for fold in range(17):
        start = data.boundary_dates[fold]
        end = (
            data.boundary_dates[fold + 1]
            if fold < 16
            else data.realization_dates[np.isfinite(data.wealth)][-1]
            + np.timedelta64(1, "D")
        )
        if fold % 2:
            ax.axvspan(start, end, color=COLORS["light_blue"], alpha=0.08, zorder=0)
        if fold > 0:
            ax.axvline(start, color=COLORS["gray"], linewidth=0.65, alpha=0.55)
        if label_folds:
            midpoint = start + (end - start) / 2
            ax.text(
                midpoint,
                0.985,
                f"F{fold:02d}\n{data.test_years[fold]}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
                color=COLORS["gray"],
            )


def draw_wealth_axis(ax: plt.Axes, data: PlotData, *, label_folds: bool = True) -> None:
    valid = np.isfinite(data.wealth)
    wealth_dates = np.concatenate((data.dates[:1], data.realization_dates[valid]))
    wealth_values = np.concatenate((np.array([1.0]), data.wealth[valid]))
    add_fold_boundaries(ax, data, label_folds=label_folds)
    ax.plot(
        wealth_dates,
        wealth_values,
        color=COLORS["blue"],
        linewidth=1.7,
        label=f"Net wealth ({data.transaction_cost_bps:g} bps one-way)",
        zorder=3,
    )
    ax.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.9, alpha=0.8)
    endpoint = float(data.wealth[valid][-1])
    endpoint_date = data.realization_dates[valid][-1]
    ax.scatter(endpoint_date, endpoint, color=COLORS["red"], s=28, zorder=4)
    ax.annotate(
        f"Final wealth {endpoint:.3f}\nCum. net {data.overall_cumulative_return:.1%}",
        xy=(endpoint_date, endpoint),
        xytext=(-95, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 0.8},
        fontsize=8.5,
        ha="right",
    )
    ax.set_title("M0 Overall Out-of-Sample Cumulative Net Wealth")
    ax.set_ylabel("Wealth (initial = 1.0)")
    ax.set_xlabel("Return realization date (t+1); initial anchor is first signal date")
    ax.grid(True, axis="y")
    ax.legend(loc="lower left")


def draw_fold_metric_axes(
    axes: tuple[plt.Axes, plt.Axes, plt.Axes] | list[plt.Axes], data: PlotData
) -> None:
    years = data.test_years
    colors = np.where(data.fold_sharpe >= 0.0, COLORS["green"], COLORS["red"])
    axes[0].bar(years, data.fold_sharpe, color=colors, width=0.72)
    axes[0].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[0].set_title("Fold Sharpe")
    axes[0].set_ylabel("Annualized")

    axes[1].plot(
        years,
        data.fold_rank_ic,
        marker="o",
        markersize=4,
        linewidth=1.4,
        color=COLORS["blue"],
    )
    axes[1].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_title("Fold Rank IC")
    axes[1].set_ylabel("Mean daily Spearman")

    axes[2].bar(years, data.fold_mdd, color=COLORS["orange"], width=0.72)
    axes[2].set_title("Fold Maximum Drawdown")
    axes[2].set_ylabel("Drawdown")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))

    for axis in axes:
        axis.grid(True, axis="y")
        axis.set_xticks(years)
        axis.tick_params(axis="x", labelrotation=60, labelsize=8)
        axis.set_xlabel("Test year")


def draw_contribution_axis(ax: plt.Axes, data: PlotData) -> None:
    values = data.fold_wealth_contribution
    colors = np.where(values >= 0.0, COLORS["green"], COLORS["red"])
    ax.bar(data.test_years, values, color=colors, width=0.72)
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(data.test_years)
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax.set_xlabel("Test year")
    ax.set_ylabel("Contribution to final wealth")
    ax.set_title(
        "Fold-Level Cumulative Net Return Contribution "
        f"(sum = {values.sum():.1%})"
    )
    ax.grid(True, axis="y")


def _save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    written: list[Path] = []
    for suffix, options in (
        (".png", {"dpi": 220}),
        (".pdf", {}),
    ):
        final = output_base.with_suffix(suffix)
        temporary = final.with_name(final.name + ".part")
        fig.savefig(temporary, format=suffix[1:], bbox_inches="tight", **options)
        temporary.replace(final)
        written.append(final)
    return written


def create_figures(data: PlotData, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(18, 7), constrained_layout=True)
    draw_wealth_axis(ax, data, label_folds=True)
    fig.suptitle(
        "M0 OOS | 2009–2025 | "
        f"{data.transaction_cost_bps:g} bps one-way cost",
        fontsize=14,
        fontweight="bold",
    )
    written.extend(
        _save_figure(fig, output_dir / f"M0_oos_cumulative_wealth_{cost_label(data.transaction_cost_bps)}")
    )
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, constrained_layout=True)
    draw_fold_metric_axes(list(axes), data)
    fig.suptitle(
        f"M0 Fold Diagnostics ({data.transaction_cost_bps:g} bps net portfolio metrics)",
        fontsize=14,
        fontweight="bold",
    )
    written.extend(
        _save_figure(fig, output_dir / f"M0_fold_metrics_over_time_{cost_label(data.transaction_cost_bps)}")
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)
    draw_contribution_axis(ax, data)
    written.extend(
        _save_figure(
            fig,
            output_dir
            / f"M0_fold_cumulative_net_return_contribution_{cost_label(data.transaction_cost_bps)}",
        )
    )
    plt.close(fig)

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.7, 1.0, 1.15))
    wealth_axis = fig.add_subplot(grid[0, :])
    metric_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    contribution_axis = fig.add_subplot(grid[2, :])
    draw_wealth_axis(wealth_axis, data, label_folds=False)
    draw_fold_metric_axes(metric_axes, data)
    draw_contribution_axis(contribution_axis, data)
    fig.suptitle(
        "M0 Out-of-Sample Diagnostic Dashboard\n"
        f"2009–2025 | {data.transaction_cost_bps:g} bps one-way cost | "
        "missing held returns contribute zero without reweighting",
        fontsize=15,
        fontweight="bold",
    )
    written.extend(
        _save_figure(fig, output_dir / f"M0_oos_diagnostic_dashboard_{cost_label(data.transaction_cost_bps)}")
    )
    plt.close(fig)
    return written


def build_figures(
    *,
    portfolio_metric_dir: Path,
    prediction_metric_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    data, provenance = load_plot_data(portfolio_metric_dir, prediction_metric_dir)
    written = create_figures(data, output_dir)
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "M0",
        "headline_transaction_cost_bps": data.transaction_cost_bps,
        "date_start": str(data.dates[0]),
        "date_end": str(data.dates[-1]),
        "wealth_date_start": str(data.realization_dates[np.isfinite(data.wealth)][0]),
        "wealth_date_end": str(data.realization_dates[np.isfinite(data.wealth)][-1]),
        "fold_count": 17,
        "test_years": data.test_years.tolist(),
        "fold_boundary_dates": [str(value) for value in data.boundary_dates],
        "overall_cumulative_net_return": data.overall_cumulative_return,
        "overall_maximum_drawdown": data.overall_mdd,
        "fold_sharpe": data.fold_sharpe.tolist(),
        "fold_rank_ic": data.fold_rank_ic.tolist(),
        "fold_maximum_drawdown": data.fold_mdd.tolist(),
        "fold_standalone_cumulative_net_return": data.fold_standalone_return.tolist(),
        "fold_additive_final_wealth_contribution": data.fold_wealth_contribution.tolist(),
        "fold_contribution_sum": float(data.fold_wealth_contribution.sum()),
        "contribution_definition": (
            "chronological wealth-before-fold times standalone fold return; "
            "contributions add exactly to final cumulative net return"
        ),
        "return_missing_assumption": data.source_assumption,
        "sources": provenance,
        "matplotlib_version": matplotlib.__version__,
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in written
        },
        "checks": {
            "wealth_endpoint_matches_summary": "passed",
            "fold_metric_mapping": "passed",
            "fold_standalone_returns_match_summary": "passed",
            "fold_contributions_sum_to_overall_return": "passed",
        },
    }
    manifest_path = output_dir / "M0_figure_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".part")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw M0 OOS diagnostic figures.")
    parser.add_argument(
        "--portfolio-metrics", type=Path, default=DEFAULT_PORTFOLIO_METRICS
    )
    parser.add_argument(
        "--prediction-metrics", type=Path, default=DEFAULT_PREDICTION_METRICS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_figures(
        portfolio_metric_dir=args.portfolio_metrics.expanduser().resolve(),
        prediction_metric_dir=args.prediction_metrics.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(f"Figures: {len(manifest['files'])}", flush=True)
    print(f"Output: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
