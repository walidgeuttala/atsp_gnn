"""Analyse hybrid GLS + model results and compare with baselines."""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


_RELATION_ORDER = [
    'pp', 'ss', 'st', 'tt', 'ts', 'sp', 'ps', 'tp', 'pt'
]


def _canonicalize_rel(rel: str) -> str:
    if not isinstance(rel, str):
        return ''
    parts = [p.strip() for p in re.split(r'[+_]', rel) if p.strip()]
    unique_parts: list[str] = []
    for p in parts:
        if p not in unique_parts:
            unique_parts.append(p)
    order_index = {name: idx for idx, name in enumerate(_RELATION_ORDER)}
    unique_parts.sort(key=lambda p: order_index.get(p, len(_RELATION_ORDER)))
    return '_'.join(unique_parts)


@dataclass
class HybridResult:
    atsp: int
    iterations: int
    aggregation: str
    relation: str
    avg_gap_pct: float
    total_time_s: float
    source: str


def _validate_hybrid_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "atsp",
        "iterations",
        "aggregation",
        "relation",
        "avg_gap_pct",
        "total_time_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hybrid dataframe missing columns: {sorted(missing)}")
    df = df.copy()
    df["atsp"] = df["atsp"].astype(int)
    df["iterations"] = df["iterations"].astype(int)
    df["aggregation"] = df["aggregation"].astype(str)
    df["relation"] = df["relation"].astype(str)
    df["avg_gap_pct"] = df["avg_gap_pct"].astype(float)
    df["total_time_s"] = df["total_time_s"].astype(float)
    if "source" not in df.columns:
        df["source"] = "hybrid.csv"
    df["relation_key"] = df["relation"].apply(_canonicalize_rel)
    df["relation"] = df["relation_key"]
    return df


def parse_hybrid_logs(paths: Iterable[Path]) -> pd.DataFrame:
    line_re = re.compile(
        r"ATSP(?P<size>\d+)\s+\|\s+iterations:\s+(?P<iterations>\d+)\s+\|"
        r"\s+Agg:\s+(?P<agg>\S+)\s+\|\s+Rel:\s+(?P<rel>[^|]+?)\s+\|"
        r"\s+Instances:\s+\d+\s+\|\s+Avg Gap:\s+(?P<gap>[0-9.]+)\s+%\s+\|\s+Total Time:\s+(?P<time>[0-9.]+)\s+s"
    )

    rows: list[HybridResult] = []
    for path in paths:
        if not path.exists():
            print(f"warning: {path} missing, skipping")
            continue
        for line in path.read_text().splitlines():
            match = line_re.search(line)
            if match:
                rows.append(
                    HybridResult(
                        atsp=int(match.group("size")),
                        iterations=int(match.group("iterations")),
                        aggregation=match.group("agg"),
                        relation=match.group("rel").strip(),
                        avg_gap_pct=float(match.group("gap")),
                        total_time_s=float(match.group("time")),
                        source=path.name,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        raise ValueError("No hybrid runs parsed; check log paths")
    df["relation_key"] = df["relation"].apply(_canonicalize_rel)
    df["relation"] = df["relation_key"]
    return df.sort_values(["aggregation", "relation", "atsp", "iterations"], ignore_index=True)


def load_batch_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "model_file",
        "atsp_size",
        "relations",
        "agg",
        "avg_final_gap",
        "total_model_time",
        "total_gls_time",
        "avg_opt_cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing from {csv_path}: {sorted(missing)}")
    df["relations"] = df["relations"].fillna("")
    df["relation_key"] = df["relations"].apply(_canonicalize_rel)
    return df


def select_best_combos(
    hybrid: pd.DataFrame,
    batch: pd.DataFrame,
    target_sizes: Iterable[int],
    max_iterations: Optional[int] = None,
    keep_all_iterations: bool = False,
) -> pd.DataFrame:
    target_sizes = list(target_sizes)
    batch = batch.copy()

    merged_rows = []
    grouped = hybrid.groupby(["aggregation", "relation", "atsp"], as_index=False)

    for (agg, rel, size), group in grouped:
        if size not in target_sizes:
            continue
        if max_iterations is not None:
            group = group[group["iterations"] <= max_iterations]
        if group.empty:
            continue
        ranked_hybrid = group.sort_values(["avg_gap_pct", "total_time_s"])  # best first
        selected_rows = ranked_hybrid.itertuples(index=False)
        if not keep_all_iterations:
            selected_rows = [ranked_hybrid.iloc[0]]

        candidates = batch.query(
            "agg == @agg and relation_key == @rel and atsp_size == @size"
        )
        if candidates.empty:
            continue
        best_model = candidates.sort_values("avg_final_gap").iloc[0]

        for entry in selected_rows:
            # `entry` is a namedtuple when coming from itertuples and a Series otherwise.
            if hasattr(entry, "_fields"):
                iter_count = getattr(entry, "iterations")
                avg_gap = getattr(entry, "avg_gap_pct")
                total_time = getattr(entry, "total_time_s")
                source = getattr(entry, "source", "")
            else:  # pandas Series from list indexing
                iter_count = entry["iterations"]
                avg_gap = entry["avg_gap_pct"]
                total_time = entry["total_time_s"]
                source = entry.get("source", "")

            merged_rows.append(
                {
                    "relations": rel,
                    "agg": agg,
                    "atsp_size": size,
                    "iterations": int(iter_count),
                    "three_opt_gap_pct": float(avg_gap),
                    "gnn_gap_pct": float(best_model["avg_init_gap"]),
                    "two_opt_gap_pct": float(best_model["avg_final_gap"]),
                    "model_file": best_model["model_file"],
                    "slurm_dir": best_model.get("slurm_dir", ""),
                    "framework": best_model.get("framework", ""),
                    "model": best_model.get("model", ""),
                    "gnn_time_s": float(best_model["total_model_time"]),
                    "two_opt_total_time_s": float(best_model["total_gls_time"]),
                    "three_opt_time_s": float(total_time),
                    "combined_two_opt_time_s": float(
                        best_model["total_model_time"] + best_model["total_gls_time"]
                    ),
                    "combined_three_opt_time_s": float(
                        best_model["total_model_time"] + total_time
                    ),
                    # Preserve legacy name for downstream code until consumers switch over.
                    "combined_total_time_s": float(
                        best_model["total_model_time"] + total_time
                    ),
                    "avg_opt_cost": float(best_model["avg_opt_cost"]),
                    "model_param_count": best_model.get("model_param_count"),
                    "hidden_dim": best_model.get("hidden_dim"),
                    "num_heads": best_model.get("num_heads"),
                    "num_gnn_layers": best_model.get("num_gnn_layers"),
                    "three_opt_source": source,
                }
            )

    result = pd.DataFrame(merged_rows)
    if result.empty:
        raise ValueError("No matching combinations between hybrid logs and batch summary")

    result = result.sort_values(
        ["atsp_size", "relations", "agg", "three_opt_gap_pct", "iterations"],
        ignore_index=True,
    )
    return result


def format_for_table(result: pd.DataFrame) -> pd.DataFrame:
    formatted = result.copy()
    formatted["GNN Time (s)"] = formatted["gnn_time_s"].round(2)
    formatted["Edge Builder + 2-Opt Time (s)"] = formatted["two_opt_total_time_s"].round(2)
    formatted["Edge Builder + 3-Opt Time (s)"] = formatted["three_opt_time_s"].round(2)
    formatted["Total Time 2-Opt (s)"] = formatted["combined_two_opt_time_s"].round(2)
    formatted["Total Time 3-Opt (s)"] = formatted["combined_three_opt_time_s"].round(2)
    formatted["Avg Gap GNN (%)"] = formatted["gnn_gap_pct"].round(2)
    formatted["Avg Gap 2-Opt (%)"] = formatted["two_opt_gap_pct"].round(2)
    formatted["Avg Gap 3-Opt (%)"] = formatted["three_opt_gap_pct"].round(2)
    formatted["Avg Opt Cost (M)"] = (formatted["avg_opt_cost"] / 1e6).round(4)
    formatted = formatted[
        [
            "atsp_size",
            "relations",
            "agg",
            "iterations",
            "model_file",
            "GNN Time (s)",
            "Edge Builder + 2-Opt Time (s)",
            "Edge Builder + 3-Opt Time (s)",
            "Total Time 2-Opt (s)",
            "Total Time 3-Opt (s)",
            "Avg Gap GNN (%)",
            "Avg Gap 2-Opt (%)",
            "Avg Gap 3-Opt (%)",
            "Avg Opt Cost (M)",
        ]
    ]
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best hybrid + model combos")
    parser.add_argument(
        "--hybrid-logs",
        type=Path,
        nargs="+",
        default=None,
        help="Paths to 3-opt hybrid log files",
    )
    parser.add_argument(
        "--hybrid-csv",
        type=Path,
        default=None,
        help="Optional CSV containing hybrid summary (columns: atsp, iterations, aggregation, relation, avg_gap_pct, total_time_s, ...)",
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        required=True,
        help="Path to batch_test_summary.csv",
    )
    parser.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=[100, 150, 250, 500],
        help="ATSP sizes to include",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional cap on iterations when selecting hybrid runs",
    )
    parser.add_argument(
        "--keep-all-iterations",
        action="store_true",
        help="Keep every hybrid iteration entry instead of only the best gap per group",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("jobs/search/output.csv"),
        help="Where to store merged best combos",
    )
    args = parser.parse_args()

    if args.hybrid_csv is not None:
        hybrid_df = _validate_hybrid_df(pd.read_csv(args.hybrid_csv)).sort_values(
            ["aggregation", "relation", "atsp", "iterations"], ignore_index=True
        )
    elif args.hybrid_logs:
        hybrid_df = parse_hybrid_logs(args.hybrid_logs)
    else:
        raise ValueError("Provide either --hybrid-csv or --hybrid-logs")
    batch_df = load_batch_summary(args.batch_summary)
    merged = select_best_combos(
        hybrid_df,
        batch_df,
        args.target_sizes,
        args.max_iterations,
        keep_all_iterations=args.keep_all_iterations,
    )

    detail_path = args.out_csv
    merged.to_csv(detail_path, index=False)
    print("Detailed merged combinations:")
    print(merged)
    print()
    print(f"Wrote detailed summary to {detail_path}")

    table_df = format_for_table(merged)
    table_path = detail_path.with_name(detail_path.stem + "_table" + detail_path.suffix)
    table_df.to_csv(table_path, index=False)
    print(f"Wrote formatted table to {table_path}")


if __name__ == "__main__":
    main()
