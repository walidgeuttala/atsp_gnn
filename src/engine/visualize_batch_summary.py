"""Visualization utilities for batch_test_search_models results.

Loads a CSV summary (e.g., jobs/search/batch_test_summary.csv) and generates
diagnostic plots covering:

- Final vs. initial gap by ATSP size and (relations, agg) combo.
- Total GNN/GLS runtime budget comparison.
- Trade-off between total time and final gap with a fitted trend line.

Outputs are PNG figures saved to an output directory for easy inspection.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass
class SummaryRow:
    slurm_dir: str
    model_file: str
    relations: str
    agg: str
    framework: str
    model: str
    atsp_size: int
    avg_init_gap: float
    avg_final_gap: float
    total_model_time: float
    total_gls_time: float

    @property
    def combo(self) -> str:
        return f"{self.relations} | {self.agg}"

    @property
    def total_time(self) -> float:
        return (self.total_model_time or 0.0) + (self.total_gls_time or 0.0)


def load_summary(csv_path: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append(
                    SummaryRow(
                        slurm_dir=raw.get("slurm_dir", ""),
                        model_file=raw.get("model_file", ""),
                        relations=raw.get("relations", ""),
                        agg=raw.get("agg", ""),
                        framework=raw.get("framework", ""),
                        model=raw.get("model", ""),
                        atsp_size=int(float(raw.get("atsp_size", "0") or 0)),
                        avg_init_gap=float(raw.get("avg_init_gap", "nan")),
                        avg_final_gap=float(raw.get("avg_final_gap", "nan")),
                        total_model_time=float(raw.get("total_model_time", "nan")),
                        total_gls_time=float(raw.get("total_gls_time", "nan")),
                    )
                )
            except ValueError:
                continue
    return [r for r in rows if not math.isnan(r.avg_final_gap)]


def pivot_metric(rows: List[SummaryRow], metric: str) -> Tuple[np.ndarray, List[str], List[int]]:
    combos = sorted({r.combo for r in rows})
    sizes = sorted({r.atsp_size for r in rows})
    combo_to_idx = {c: i for i, c in enumerate(combos)}
    size_to_idx = {s: j for j, s in enumerate(sizes)}
    matrix = np.full((len(combos), len(sizes)), np.nan, dtype=float)

    for r in rows:
        i = combo_to_idx[r.combo]
        j = size_to_idx[r.atsp_size]
        if metric == "avg_final_gap":
            matrix[i, j] = r.avg_final_gap
        elif metric == "avg_init_gap":
            matrix[i, j] = r.avg_init_gap
        elif metric == "total_model_time":
            matrix[i, j] = r.total_model_time
        elif metric == "total_gls_time":
            matrix[i, j] = r.total_gls_time
        elif metric == "total_time":
            matrix[i, j] = r.total_time
    return matrix, combos, sizes


def plot_heatmap(matrix: np.ndarray, combos: List[str], sizes: List[int], *, title: str, cmap: str, out_path: Path, value_fmt: str = "{:.2f}") -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(sizes) * 1.5), max(4, len(combos) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_yticks(np.arange(len(combos)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yticklabels(combos)
    ax.set_xlabel("ATSP size")
    ax.set_title(title)

    # Annotate cells with values when finite.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, value_fmt.format(val), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_vs_time(rows: List[SummaryRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    combos = sorted({r.combo for r in rows})
    color_map = {combo: plt.cm.tab20(i % 20) for i, combo in enumerate(combos)}

    times = np.array([r.total_time for r in rows])
    gaps = np.array([r.avg_final_gap for r in rows])
    slope, intercept, r_value, p_value, _ = stats.linregress(times, gaps)

    for r in rows:
        ax.scatter(r.total_time, r.avg_final_gap, color=color_map[r.combo], label=r.combo)

    x_fit = np.linspace(times.min(), times.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="black", linestyle="--", label=f"Trend (R²={r_value**2:.3f})")

    ax.set_xlabel("Total time (s) [Model + GLS]")
    ax.set_ylabel("Average final gap (%)")
    ax.set_title("Gap vs. total time")
    # Deduplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, bool] = {}
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_handles.append(h)
            uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_improvement_vs_time(rows: List[SummaryRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    combos = sorted({r.combo for r in rows})
    color_map = {combo: plt.cm.Paired(i % 12) for i, combo in enumerate(combos)}
    improvement = np.array([r.avg_init_gap - r.avg_final_gap for r in rows])
    times = np.array([r.total_time for r in rows])
    sizes = np.array([r.atsp_size for r in rows])

    # Fit linear regression
    slope, intercept, r_value, p_value, _ = stats.linregress(times, improvement)
    x_fit = np.linspace(times.min(), times.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color="black", linestyle="--", label=f"Trend (R²={r_value**2:.3f})")

    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    size_cmap = plt.cm.get_cmap("viridis")
    for r in rows:
        ax.scatter(
            r.total_time,
            r.avg_init_gap - r.avg_final_gap,
            color=color_map[r.combo],
            edgecolor=size_cmap(norm(r.atsp_size)),
            linewidths=1.2,
            label=r.combo,
        )

    ax.set_xlabel("Total time (s) [Model + GLS]")
    ax.set_ylabel("Gap improvement (init - final)")
    ax.set_title("Gap improvement vs. total time")

    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, bool] = {}
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_handles.append(h)
            uniq_labels.append(l)
    legend = ax.legend(uniq_handles, uniq_labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    legend.set_title("relations | agg")

    sm = plt.cm.ScalarMappable(cmap=size_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("ATSP size")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sizewise_scatter(rows: List[SummaryRow], out_path: Path) -> None:
    sizes = sorted({r.atsp_size for r in rows})
    n_cols = 2
    n_rows = int(math.ceil(len(sizes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    combos = sorted({r.combo for r in rows})
    color_map = {combo: plt.cm.tab10(i % 10) for i, combo in enumerate(combos)}

    for idx, size in enumerate(sizes):
        ax = axes[idx // n_cols][idx % n_cols]
        subset = [r for r in rows if r.atsp_size == size]
        times = [r.total_time for r in subset]
        gaps = [r.avg_final_gap for r in subset]
        improvements = [r.avg_init_gap - r.avg_final_gap for r in subset]
        for r, t, g, imp in zip(subset, times, gaps, improvements):
            ax.scatter(t, g, s=max(20, imp * 4), color=color_map[r.combo], label=r.combo, alpha=0.8)
            ax.annotate(r.model_file.replace('best_model_rel_', ''), (t, g), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7)
        ax.set_title(f"ATSP size {size}")
        ax.set_xlabel("Total time (s)")
        ax.set_ylabel("Avg final gap (%)")
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for j in range(len(sizes), n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
    fig.legend(uniq_handles, uniq_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_time_stacked(rows: List[SummaryRow], out_path: Path) -> None:
    combos = sorted({r.combo for r in rows})
    sizes = sorted({r.atsp_size for r in rows})
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8, 3 * len(sizes)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, size in zip(axes, sizes):
        subset = [r for r in rows if r.atsp_size == size]
        subset.sort(key=lambda r: r.total_time)
        labels = [r.combo for r in subset]
        model_times = [r.total_model_time for r in subset]
        gls_times = [r.total_gls_time for r in subset]
        x = np.arange(len(subset))
        ax.barh(x, model_times, color="#1f77b4", label="Model time")
        ax.barh(x, gls_times, left=model_times, color="#ff7f0e", label="GLS time")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Total time (s)")
        ax.set_title(f"Time budget breakdown @ ATSP {size}")

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize batch test summary results")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "jobs" / "search" / "batch_test_summary.csv",
        help="Path to batch_test_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "jobs" / "search" / "plots",
        help="Directory to write output figures",
    )
    args = parser.parse_args()

    rows = load_summary(args.summary_csv)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.summary_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Heatmaps for gaps and total runtime.
    final_gap_matrix, combos, sizes = pivot_metric(rows, "avg_final_gap")
    plot_heatmap(
        final_gap_matrix,
        combos,
        sizes,
        title="Average final gap (%)",
        cmap="YlOrRd_r",
        out_path=args.output_dir / "heatmap_final_gap.png",
    )

    init_gap_matrix, _, _ = pivot_metric(rows, "avg_init_gap")
    plot_heatmap(
        init_gap_matrix,
        combos,
        sizes,
        title="Average initial gap (%)",
        cmap="Blues",
        out_path=args.output_dir / "heatmap_init_gap.png",
    )

    total_time_matrix, _, _ = pivot_metric(rows, "total_time")
    plot_heatmap(
        total_time_matrix,
        combos,
        sizes,
        title="Total inference + GLS time (s)",
        cmap="Greens",
        out_path=args.output_dir / "heatmap_total_time.png",
        value_fmt="{:.1f}",
    )

    # Stacked time bars per size.
    plot_time_stacked(rows, args.output_dir / "stacked_time_breakdown.png")

    # Scatter with regression.
    plot_gap_vs_time(rows, args.output_dir / "gap_vs_total_time.png")
    plot_gap_improvement_vs_time(rows, args.output_dir / "gap_improvement_vs_total_time.png")
    plot_sizewise_scatter(rows, args.output_dir / "sizewise_gap_scatter.png")

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
