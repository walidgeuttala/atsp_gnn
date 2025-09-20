#!/usr/bin/env python3
"""Estimate LKH runtime for representative ATSP instances.

For each target ATSP size we solve a single instance with LKH and
approximate the cost of processing 30 instances by simple scaling.
Output is written as a CSV under ``jobs/plots`` and also echoed to stdout.
"""

from __future__ import annotations

import csv
import pickle
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import lkh
import tsplib95

from src.utils.atsp_utils import get_adj_matrix_string


TARGET_SIZES: Tuple[int, ...] = (100, 150, 250, 500, 1000)


def find_instance(dataset_dir: Path) -> Path:
    """Return the first test instance path for the dataset."""
    test_file = dataset_dir / "test.txt"
    candidate: Path | None = None
    if test_file.exists():
        with test_file.open("r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    candidate = dataset_dir / line
                    break
    if candidate and candidate.exists():
        return candidate
    # Fallback: first *.pkl in directory
    for path in sorted(dataset_dir.glob("*.pkl")):
        if path.name != "scalers.pkl":
            return path
    raise FileNotFoundError(f"No instances found under {dataset_dir}")


def load_graph(instance_path: Path):
    with instance_path.open("rb") as fh:
        return pickle.load(fh)


def solve_with_lkh(tsplib_str: str, lkh_binary: Path) -> float:
    problem = tsplib95.loaders.parse(tsplib_str)
    start = time.perf_counter()
    # ``lkh.solve`` returns the tour but we only need the duration here.
    lkh.solve(str(lkh_binary), problem=problem)
    return time.perf_counter() - start


def estimate_for_sizes(sizes: Iterable[int]) -> List[dict]:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "saved_dataset"
    lkh_binary = project_root / "LKH-3.0.9" / "LKH"
    if not lkh_binary.exists():
        raise FileNotFoundError(f"LKH binary not found at {lkh_binary}")

    estimates: List[dict] = []
    for size in sizes:
        dataset_dir = dataset_root / f"ATSP_30x{size}"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder missing: {dataset_dir}")

        instance_path = find_instance(dataset_dir)
        graph = load_graph(instance_path)
        tsplib_str = get_adj_matrix_string(graph, weight="weight")

        elapsed = solve_with_lkh(tsplib_str, lkh_binary)
        estimates.append(
            {
                "atsp_size": size,
                "instance": instance_path.name,
                "single_run_seconds": round(elapsed, 4),
                "estimated_30_runs_seconds": round(elapsed * 30.0, 4),
            }
        )
    return estimates


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["atsp_size", "instance", "single_run_seconds", "estimated_30_runs_seconds"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = estimate_for_sizes(TARGET_SIZES)
    out_path = Path(__file__).resolve().parents[2] / "../jobs/plots/lkh_runtime_estimate.csv"
    write_csv(rows, out_path)

    print("Estimated LKH runtimes (seconds):")
    for row in rows:
        single = row["single_run_seconds"]
        thirty = row["estimated_30_runs_seconds"]
        print(
            f"ATSP {row['atsp_size']:>4} | instance {row['instance']:>32} | "
            f"1-run {single:6.2f}s | 30-run {thirty:7.2f}s"
        )
    print(f"\nCSV written to {out_path}")


if __name__ == "__main__":
    main()
