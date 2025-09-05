import argparse
import multiprocessing as mp
import os
import pathlib
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable
import pickle 

import networkx as nx
import numpy as np
import tsplib95
import lkh
from scipy.sparse.csgraph import floyd_warshall

def compute_edge_regret_wrapper(args):
    """Wrapper function for multiprocessing - unpacks arguments."""
    return compute_edge_regret(*args)

def compute_edge_regret(edge, adj, base_cost, lkh_path):
    """Compute regret for one edge (i,j)."""
    i, j = edge
    inst = ATSPInstance(adj.copy())
    _, cost = inst.solve_lkh_with_fixed_edge((i, j), lkh_path=lkh_path)
    value = 1.0 if base_cost >= 0 else -1.0
    regret = ((cost - base_cost) / base_cost) * value
    return (i, j), regret


# ATSPInstance: one problem (adjacency, tour, cost, regrets)
@dataclass
class ATSPInstance:
    adj: np.ndarray                   # (n, n) metric adjacency (float)
    tour: Optional[List[int]] = None  # tour as 0-based cycle [.., start]
    cost: Optional[float] = None      # cost of 'tour'
    regrets: Optional[np.ndarray] = None  # (n, n) edge labels (float)

    # ---------- Construction ----------

    @classmethod
    def from_random(
        cls,
        n: int,
        weight_min: int = 100,
        weight_max: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> "ATSPInstance":
        """Create random complete directed instance and metric-close via Floyd–Warshall."""
        if rng is None:
            rng = np.random.default_rng()
        weights = rng.integers(weight_min, weight_max + 1, size=(n, n)).astype(float)
        np.fill_diagonal(weights, 0.0)
        dist = floyd_warshall(weights, directed=True)  # float matrix; enforces triangle inequality
        return cls(adj=dist)

    @classmethod
    def from_tsplib_file(cls, file_path: os.PathLike) -> "ATSPInstance":
        """Parse ATSP FULL_MATRIX instance (keeps floats if present)."""
        with open(file_path, "r") as f:
            lines = f.readlines()

        dim = 0
        in_weights = False
        vals: List[float] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("DIMENSION"):
                dim = int(s.split()[-1])
            elif s.startswith("EDGE_WEIGHT_SECTION"):
                in_weights = True
            elif in_weights:
                if s == "EOF":
                    break
                vals.extend(map(float, s.split()))

        if dim <= 0:
            raise ValueError("DIMENSION not found or invalid in TSPLIB file.")

        adj = np.array(vals, dtype=float).reshape(dim, dim)
        return cls(adj=adj)

    # ---------- TSPLIB / LKH interface ----------

    def _to_tsplib_string(self, fixed_edge: Optional[Tuple[int, int]] = None) -> str:
        """Build TSPLIB string (ATSP, FULL_MATRIX). Optionally include FIXED_EDGES_SECTION."""
        n = self.adj.shape[0]
        header = (
            f"NAME: ATSP\n"
            f"COMMENT: {n}-city problem\n"
            f"TYPE: ATSP\n"
            f"DIMENSION: {n}\n"
            f"EDGE_WEIGHT_TYPE: EXPLICIT\n"
            f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
            f"EDGE_WEIGHT_SECTION\n"
        )
        rows = []
        for i in range(n):
            rows.append(" ".join(str(int(self.adj[i, j])) for j in range(n)))
        body = "\n".join(rows)

        if fixed_edge is None:
            return f"{header}{body}\nEOF\n"

        # LKH uses 1-based indexing
        i, j = fixed_edge
        fixed = (
            "\nFIXED_EDGES_SECTION\n"
            f"{i+1} {j+1}\n"
            "-1\n"
            "EOF\n"
        )
        return f"{header}{body}{fixed}"

    def solve_lkh(self, lkh_path: str = "../LKH-3.0.9/LKH") -> Tuple[List[int], float]:
        """Solve instance via LKH and cache tour/cost."""
        tsp_str = self._to_tsplib_string()
        problem = tsplib95.parse(tsp_str)
        solution = lkh.solve(lkh_path, problem=problem)
        tour = [v - 1 for v in solution[0]] + [solution[0][0] - 1]  # close cycle
        cost = self._tour_cost(tour)
        self.tour, self.cost = tour, cost
        return tour, cost

    def solve_lkh_with_fixed_edge(
        self, edge: Tuple[int, int], lkh_path: str = "../LKH-3.0.9/LKH"
    ) -> Tuple[List[int], float]:
        """Solve with edge forced into the tour using FIXED_EDGES_SECTION."""
        tsp_str = self._to_tsplib_string(fixed_edge=edge)
        problem = tsplib95.parse(tsp_str)
        solution = lkh.solve(lkh_path, problem=problem)
        tour = [v - 1 for v in solution[0]] + [solution[0][0] - 1]
        cost = self._tour_cost(tour)
        return tour, cost

    def _tour_cost(self, tour: List[int]) -> float:
        return float(sum(self.adj[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    # ---------- Labeling ----------

    def label_regrets(
        self,
        mode: str = "row_best",
        lkh_path: str = "../LKH-3.0.9/LKH",
        base_tour: Optional[List[int]] = None,
        base_cost: Optional[float] = None,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute edge regrets.
        - mode="row_best": fast heuristic: regret(i->j) = w(i,j) - min_{k!=i} w(i,k)
        - mode="fixed_edge_lkh": accurate but heavy; regret = (opt_cost_with_edge - opt_cost) / opt_cost
        """
        n = self.adj.shape[0]
        regrets = np.zeros((n, n), dtype=float)

        if mode == "row_best":
            row_min = np.copy(self.adj)
            np.fill_diagonal(row_min, np.inf)
            best = np.min(row_min, axis=1)  # (n,)
            regrets = self.adj - best[:, None]
            np.fill_diagonal(regrets, 0.0)
            self.regrets = regrets
            return regrets

        if mode == "fixed_edge_lkh":
            # ensure base tour/cost
            if base_tour is None or base_cost is None:
                base_tour, base_cost = self.solve_lkh(lkh_path=lkh_path)
            base_cost = float(base_cost if base_cost != 0 else 1e-6)

            edge_list = [(i, j) for i in range(n) for j in range(n) if i != j]

            def work(e: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
                _, c = self.solve_lkh_with_fixed_edge(e, lkh_path=lkh_path)
                # normalized regret as in your code:
                value = 1.0 if base_cost >= 0 else -1.0
                regret = ((c - base_cost) / base_cost) * value
                return e, regret

            if parallel:
                args_list = [(e, self.adj, base_cost, lkh_path) for e in edge_list]
                with mp.Pool(processes=processes) as pool:
                    for (i, j), r in pool.imap_unordered(compute_edge_regret_wrapper, args_list):
                        regrets[i, j] = r
            else:
                for e in edge_list:
                    (i, j), r = work(e)
                    regrets[i, j] = r

            self.regrets = regrets
            return regrets

        raise ValueError(f"Unknown regret mode: {mode}")

    # ---------- Export ----------

    def to_networkx(self) -> nx.DiGraph:
        """Edge attributes: weight, regret, in_solution."""
        n = self.adj.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        in_sol = set()
        if self.tour is not None:
            in_sol = {(self.tour[i], self.tour[i + 1]) for i in range(len(self.tour) - 1)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                attrs = {"weight": float(self.adj[i, j])}
                if self.regrets is not None:
                    attrs["regret"] = float(self.regrets[i, j])
                attrs["in_solution"] = (i, j) in in_sol
                G.add_edge(i, j, **attrs)
        if self.tour is not None:
            G.graph["tour"] = self.tour
            G.graph["cost"] = float(self.cost)
        return G


# ATSPDatasetBuilder: many instances, no noisy globals
class ATSPDatasetBuilder:
    def __init__(
        self,
        n_nodes: int,
        n_instances: int,
        output_dir: pathlib.Path,
        weight_min: int = 100,
        weight_max: int = 1000,
        lkh_path: str = "../LKH-3.0.9/LKH",
    ):
        self.n_nodes = n_nodes
        self.n_instances = n_instances
        self.output_dir = pathlib.Path(output_dir)
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.lkh_path = lkh_path

    # --- generators ---

    def _iter_random_instances(self) -> Iterable[ATSPInstance]:
        rng = np.random.default_rng()
        for _ in range(self.n_instances):
            yield ATSPInstance.from_random(
                self.n_nodes, self.weight_min, self.weight_max, rng=rng
            )

    def _iter_tsplib_dir(self, dir_path: pathlib.Path) -> Iterable[ATSPInstance]:
        for name in sorted(os.listdir(dir_path)):
            p = dir_path / name
            if p.is_file():
                yield ATSPInstance.from_tsplib_file(p)

    # --- build pipeline ---

    def build_and_save(
        self,
        from_tsplib_dir: Optional[pathlib.Path] = None,
        regret_mode: str = "row_best",
        parallel: bool = False,
        processes: Optional[int] = None,
        save_graph_pickles: bool = True,
        save_summary_csv: bool = True,
    ) -> None:
        """
        End-to-end:
          - create/parse instances
          - solve with LKH
          - label regrets (row_best or fixed_edge_lkh)
          - save pickles + summary
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_lines = ["id,n_nodes,cost\n"]

        if from_tsplib_dir is None:
            src_iter = self._iter_random_instances()
        else:
            src_iter = self._iter_tsplib_dir(pathlib.Path(from_tsplib_dir))

        for inst in src_iter:
            # Solve base tour
            tour, cost = inst.solve_lkh(self.lkh_path)

            # Labels
            inst.label_regrets(
                mode=regret_mode,
                lkh_path=self.lkh_path,
                base_tour=tour,
                base_cost=cost,
                parallel=parallel,
                processes=processes,
            )

            # Save
            uid = uuid.uuid4().hex
            summary_lines.append(f"{uid},{inst.adj.shape[0]},{cost}\n")

            if save_graph_pickles:
                G = inst.to_networkx()
                filepath = self.output_dir / f"{uid}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(G, f)

        if save_summary_csv:
            with open(self.output_dir / "summary.csv", "w") as f:
                f.writelines(summary_lines)



# =========================================================
# CLI (instances only; dataset building)
# =========================================================

def _main():
    parser = argparse.ArgumentParser(description="Generate/label ATSP dataset.")
    parser.add_argument("n_samples", type=int, help="Number of instances to generate")
    parser.add_argument("n_nodes", type=int, help="Number of nodes per instance")
    parser.add_argument("out_dir", type=pathlib.Path, help="Output directory")
    parser.add_argument("--weight_min", type=int, default=100)
    parser.add_argument("--weight_max", type=int, default=1000)
    parser.add_argument("--lkh_path", type=str, default="../LKH-3.0.9/LKH")
    parser.add_argument("--regret_mode", type=str, default="fixed_edge_lkh",
                        choices=["row_best", "fixed_edge_lkh"])
    parser.add_argument("--parallel", action="store_true",
                        help="Parallelize fixed-edge LKH (heavy).")
    parser.add_argument("--processes", type=int, default=None, help="Pool size for edge parallelism")
    parser.add_argument("--from_tsplib_dir", type=pathlib.Path, default=None,
                        help="If set, read instances from this directory instead of random generation.")
    args = parser.parse_args()

    subdir_name = f"ATSP_{args.n_nodes}x{args.n_samples}"
    args.out_dir = args.out_dir / subdir_name
    os.makedirs(args.out_dir, exist_ok=True)

    builder = ATSPDatasetBuilder(
        n_nodes=args.n_nodes,
        n_instances=args.n_samples,
        output_dir=args.out_dir,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        lkh_path=args.lkh_path,
    )

    builder.build_and_save(
        from_tsplib_dir=args.from_tsplib_dir,
        regret_mode=args.regret_mode,
        parallel=args.parallel,
        processes=args.processes,
        save_graph_pickles=True,
        save_summary_csv=True,
    )

if __name__ == "__main__":
    _main()
