import sys
import pathlib
import argparse
from typing import Tuple, List

import numpy as np


def _ensure_repo_path() -> None:
    """Ensure the repository's `src` package is importable when run as a script.

    Adds the repo root (the directory that contains `tsp/atsp_gnn`) to sys.path
    so `import src.data...` works whether invoked from repo root or elsewhere.
    """
    this_file = pathlib.Path(__file__).resolve()
    # repo_root/.../tsp/atsp_gnn/src/data/validate_dataset_50.py
    # repo_root is two parents up from `tsp` (i.e., `.../<repo_root>`)
    tsp_dir = this_file.parents[3]  # .../tsp
    repo_root = tsp_dir.parent      # repo root that contains `tsp`
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_path()

from src.data.dataset_dgl import ATSPDatasetDGL  # noqa: E402


DEFAULT_DATA_DIR = \
    "../saved_dataset/ATSP_3000x50"


def check_graph_raw(nxG, atsp_size: int, eps: float = 1e-8) -> Tuple[bool, List[str]]:
    """Validate one raw NetworkX ATSP instance.

    - Exactly `atsp_size` edges should have in_solution == 1 (Hamiltonian tour)
    - Every in_solution edge must have regret == 0 (within `eps`)
    Returns (ok, messages).
    """
    msgs: List[str] = []
    selected = []
    for u, v, data in nxG.edges(data=True):
        if data.get("in_solution", 0.0) >= 0.5:
            selected.append((u, v, data))

    ok_count = (len(selected) == atsp_size)
    if not ok_count:
        msgs.append(
            f"Selected edges count {len(selected)} != {atsp_size}"
        )

    ok_regret = True
    for (u, v, data) in selected:
        r = float(data.get("regret", 0.0))
        if abs(r) > eps:
            ok_regret = False
            msgs.append(
                f"Regret non-zero on selected edge ({u},{v}): regret={r}"
            )
            # don't spam; keep scanning to report a few more
            if len(msgs) > 10:
                break

    return (ok_count and ok_regret), msgs


def check_graph_dgl(g_dgl, atsp_size: int, eps: float = 1e-8) -> Tuple[bool, List[str]]:
    """Cross-check using the DGL graph features (unscaled in_solution, scaled regret).

    Since regret is scaled in the DGL dataset, only verify the in_solution count here.
    Returns (ok, messages).
    """
    msgs: List[str] = []
    in_sol = g_dgl.ndata.get("in_solution", None)
    if in_sol is None:
        return False, ["Missing ndata['in_solution'] in DGL graph"]

    # in_solution stored as shape (m, 1)
    try:
        in_sol_np = in_sol.squeeze(1).detach().cpu().numpy()
    except Exception:
        in_sol_np = np.asarray(in_sol).reshape(-1)

    selected_count = int((in_sol_np >= 0.5).sum())
    ok = (selected_count == atsp_size)
    if not ok:
        msgs.append(
            f"DGL in_solution count {selected_count} != {atsp_size}"
        )
    return ok, msgs


def main():
    parser = argparse.ArgumentParser(
        description="Validate ATSP 50 dataset: in-solution edges have regret 0 and exactly N selected."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to ATSP_3000x50 dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["val", "val", "test"],
        help="Dataset split to validate",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="ATSP size (number of nodes)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Optional cap on number of instances to check",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}")
        sys.exit(2)

    ds = ATSPDatasetDGL(
        data_dir=data_dir,
        split=args.split,
        atsp_size=args.size,
        device="cpu",
        undirected=True,
        load_once=False,  # get raw NetworkX graph along with processed DGL
    )

    total = len(ds)
    limit = args.max if args.max is not None else total
    failures = 0
    details: List[str] = []

    for idx in range(limit):
        try:
            g_dgl, g_nx = ds[idx]  # __getitem__ returns (H, graph) when load_once=False
        except Exception as e:
            failures += 1
            details.append(f"[{idx}] load error: {e}")
            continue

        ok_raw, msgs_raw = check_graph_raw(g_nx, args.size)
        ok_dgl, msgs_dgl = check_graph_dgl(g_dgl, args.size)
        if not (ok_raw and ok_dgl):
            failures += 1
            name = getattr(g_nx, "graph", {}).get("name", "<unknown>")
            prefix = f"[{idx}] {name}"
            for m in (msgs_raw + msgs_dgl):
                details.append(f"{prefix}: {m}")

    passed = limit - failures
    print(f"Checked {passed}/{limit} instances in split '{args.split}'.")
    if failures:
        print(f"FAILURES: {failures}")
        # limit output to avoid flooding
        for line in details[:50]:
            print(line)
        sys.exit(1)
    else:
        print("All checks passed: each tour selects exactly N edges and all have regret 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()

