import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


def parse_instance_file(path: Path, n: int) -> Dict:
    """Parse an exported instance*.txt file produced by run_export/test_dgl_export.

    Expected blocks:
      edge_weight:\n <NxN floats>\n
      regret:\n <NxN floats>\n
      regret_pred:\n <NxN floats>\n
      opt_cost: <float>\n
      num_iterations: <int>\n
      init_cost: <float>\n
      best_cost: <float>\n
    Returns dict with numpy arrays and metadata.
    """
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    def read_matrix(start_idx: int) -> Tuple[np.ndarray, int]:
        rows: List[List[float]] = []
        i = start_idx
        while i < len(lines) and lines[i].strip() != '':
            rows.append([float(x) for x in lines[i].split()])
            i += 1
        M = np.array(rows, dtype=np.float64)
        return M, i

    i = 0
    # edge_weight
    while i < len(lines) and lines[i].strip() != 'edge_weight:':
        i += 1
    assert i < len(lines), f"edge_weight header not found in {path}"
    i += 1
    W, i = read_matrix(i)
    i += 1  # skip blank line

    # regret
    assert lines[i].strip() == 'regret:', f"regret header missing in {path}"
    i += 1
    R, i = read_matrix(i)
    i += 1

    # regret_pred
    assert lines[i].strip() == 'regret_pred:', f"regret_pred header missing in {path}"
    i += 1
    RP, i = read_matrix(i)
    i += 1

    # Optional blocks: regret_true, in_solution (order may vary if appended)
    RT = None
    IS = None
    # Attempt to read up to two optional matrices
    for _ in range(2):
        # skip blank lines
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        if i >= len(lines):
            break
        hdr = lines[i].strip()
        if hdr == 'regret_true:':
            i += 1
            RT, i = read_matrix(i)
            i += 1
        elif hdr == 'in_solution:':
            i += 1
            IS, i = read_matrix(i)
            i += 1
        else:
            break

    # tail metrics (order-insensitive)
    meta = {}
    for j in range(i, len(lines)):
        ln = lines[j].strip()
        if not ln:
            continue
        if ':' in ln:
            k, v = ln.split(':', 1)
            meta[k.strip()] = v.strip()

    return {
        'edge_weight': W,
        'regret': R,
        'regret_pred': RP,
        'regret_true': RT,
        'in_solution': IS,
        'meta': meta,
    }


def zeros_cycle_from_regret(R: np.ndarray, tol: float = 1e-6) -> Tuple[bool, List[int], str]:
    """Check if off-diagonal zeros in regret define a single Hamiltonian cycle.

    Returns (ok, tour, message). `tour` includes the return to start.
    """
    n = R.shape[0]
    # One zero per row and per column (excluding diagonal)
    row_zero_idx = []
    for i in range(n):
        candidates = [j for j in range(n) if i != j and abs(R[i, j]) <= tol]
        if len(candidates) != 1:
            return False, [], f"Row {i}: expected 1 zero off-diagonal, found {len(candidates)}"
        row_zero_idx.append(candidates[0])

    col_counts = [0] * n
    for j in row_zero_idx:
        col_counts[j] += 1
    bad_cols = [j for j, c in enumerate(col_counts) if c != 1]
    if bad_cols:
        return False, [], f"Columns not balanced (must be 1 incoming per node): {bad_cols}"

    # Follow cycle starting at 0
    tour = [0]
    seen = {0}
    cur = 0
    for _ in range(n):
        nxt = row_zero_idx[cur]
        if nxt in seen:
            # Close only at the end
            if nxt == 0 and len(tour) == n:
                tour.append(0)
                break
            return False, [], f"Early cycle closure or repeated node at {nxt}"
        tour.append(nxt)
        seen.add(nxt)
        cur = nxt
    if len(tour) != n + 1 or tour[-1] != 0:
        return False, [], "Did not form a single n-node cycle"
    return True, tour, "ok"


def cycle_from_in_solution(IS: np.ndarray) -> Tuple[bool, List[int], str]:
    n = IS.shape[0]
    # exactly one outgoing and one incoming per node
    if not np.allclose(IS.sum(axis=1), np.ones(n)):
        return False, [], "Each row must have exactly one 1 (one outgoing edge)"
    if not np.allclose(IS.sum(axis=0), np.ones(n)):
        return False, [], "Each column must have exactly one 1 (one incoming edge)"
    # follow cycle from 0
    nxt_of = {i: int(np.argmax(IS[i])) for i in range(n)}
    tour = [0]
    seen = {0}
    cur = 0
    for _ in range(n):
        nxt = nxt_of[cur]
        if nxt in seen:
            if nxt == 0 and len(tour) == n:
                tour.append(0)
                break
            return False, [], f"Early closure or repeated node {nxt}"
        tour.append(nxt)
        seen.add(nxt)
        cur = nxt
    if len(tour) != n + 1 or tour[-1] != 0:
        return False, [], "Did not form a single n-node cycle"
    return True, tour, "ok"


def tour_cost_from_matrix(W: np.ndarray, tour: List[int]) -> float:
    return float(sum(W[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def validate_dir(instances_dir: Path, n: int, write_report: bool = True) -> Dict:
    files = sorted(instances_dir.glob('instance*.txt'))
    assert files, f"No instance*.txt found under {instances_dir}"

    summary = {
        'checked': 0,
        'valid_cycles': 0,
        'regret_zero_edges_per_row_ok': 0,
        'opt_cost_match': 0,
        'errors': [],
    }

    for fp in files:
        try:
            rec = parse_instance_file(fp, n)
            W = rec['edge_weight']
            R = rec['regret']
            RT = rec.get('regret_true', None)
            IS = rec.get('in_solution', None)

            if W.shape != (n, n) or R.shape != (n, n):
                summary['errors'].append(f"{fp.name}: shape mismatch W={W.shape}, R={R.shape}")
                continue

            # Prefer in_solution; else use regret_true zeros; else regret zeros
            if IS is not None:
                ok, tour, msg = cycle_from_in_solution(IS)
            elif RT is not None:
                ok, tour, msg = zeros_cycle_from_regret(RT)
            else:
                ok, tour, msg = zeros_cycle_from_regret(R)
            summary['checked'] += 1
            if ok:
                summary['valid_cycles'] += 1
            else:
                summary['errors'].append(f"{fp.name}: {msg}")

            # If cycle valid, check opt_cost matches cost along zero-regret edges
            if ok:
                calc_cost = tour_cost_from_matrix(W, tour)
                opt_cost = float(rec['meta'].get('opt_cost', 'nan'))
                if not np.isnan(opt_cost) and abs(calc_cost - opt_cost) <= 1e-4:
                    summary['opt_cost_match'] += 1
                else:
                    summary['errors'].append(
                        f"{fp.name}: opt_cost mismatch (calc={calc_cost:.6f}, file={opt_cost})"
                    )
                # Also verify regret_true edges are ~0 on the tour if present
                if RT is not None:
                    max_abs = max(abs(RT[tour[i], tour[i + 1]]) for i in range(len(tour) - 1))
                    if max_abs > 1e-6:
                        summary['errors'].append(
                            f"{fp.name}: regret_true on tour not ~0 (max_abs={max_abs:.3e})"
                        )
        except Exception as e:
            summary['errors'].append(f"{fp.name}: parse/validation error: {e}")

    if write_report:
        out = instances_dir / 'validation_report.json'
        with open(out, 'w') as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description='Validate exported ATSP instances (regret-based)')
    ap.add_argument('--instances_dir', type=str, required=True,
                    help='Path to folder containing instance*.txt files (trial_0/test_atsp{N})')
    ap.add_argument('--atsp_size', type=int, required=True)
    args = ap.parse_args()

    instances_dir = Path(args.instances_dir)
    res = validate_dir(instances_dir, args.atsp_size, write_report=True)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
