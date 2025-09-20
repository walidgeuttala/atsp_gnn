"""
Updated ATSP dataset builder and instance utilities.

Main changes vs your original:
- Add support for reading a single .pt file with shape (num_samples, n, n) OR (n, n).
  If .pt supplied, its contents drive the instances (we ignore positional n_samples/n_nodes).
- Add CLI flag --from_pt_file to load adjacency matrices from a PyTorch .pt.
- Add CLI flag --no_regret to compute tours/costs but skip expensive regret labeling.
- Keep existing TSPLIB / random generation code paths.
- Improve fixed-edge LKH regret computation:
  - skip edges already in the base tour (no need to force them)
  - robust base_cost handling (avoid divide-by-zero)
  - optional multiprocessing (safe argument packing for Pool)
  - worker logs exceptions and returns inf on failures
- Clear logging and careful error messages.

Notes:
- Requires: numpy, scipy, networkx, tsplib95, lkh wrapper you used previously.
- Optional: torch (only required if using --from_pt_file).
"""

import argparse
import multiprocessing as mp
import os
import pathlib
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable
import pickle
import logging
import sys

import networkx as nx
import numpy as np
import tsplib95
import lkh
from scipy.sparse.csgraph import floyd_warshall

#TODO: add new alternatives for ATSPInstance
#TODO: add conversion functions

# optional dependency for .pt files
try:
    import torch  # type: ignore
except Exception:
    torch = None  # only required if you use --from_pt_file

# --- basic logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


# -------------------------
# Multiprocessing worker
# -------------------------
def compute_edge_regret_worker(args):
    """
    Worker for Pool.imap_unordered.
    args: (adj, edge, base_cost, lkh_path)
    Returns: ((i, j), regret)
    Robust: logs exceptions and returns (i,j), inf if LKH fails for that edge.
    """
    adj, edge, base_cost, lkh_path = args
    i, j = edge
    try:
        inst = ATSPInstance(adj.copy())
        _, cost = inst.solve_lkh_with_fixed_edge((i, j), lkh_path=lkh_path)
    except Exception as e:
        logging.exception("LKH failed for fixed edge %s -> %s: %s", i, j, e)
        return (i, j), float("inf")
    value = 1.0 if base_cost >= 0 else -1.0
    # base_cost is expected non-zero; caller should have ensured fallback
    regret = ((cost - base_cost) / base_cost) * value
    return (i, j), regret


# -------------------------
# ATSP instance definition
# -------------------------
@dataclass
class ATSPInstance:
    adj: np.ndarray                   # (n, n) adjacency (float)
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
        enforce_metric: bool = True,
    ) -> "ATSPInstance":
        """
        Create random complete directed instance.
        If enforce_metric=True, run Floyd–Warshall on random integer weights to ensure triangle inequality.
        """
        if rng is None:
            rng = np.random.default_rng()
        weights = rng.integers(weight_min, weight_max + 1, size=(n, n)).astype(float)
        np.fill_diagonal(weights, 0.0)
        if enforce_metric:
            dist = floyd_warshall(weights, directed=True)
        else:
            dist = weights
        return cls(adj=dist)

    @classmethod
    def from_tsplib_file(cls, file_path: os.PathLike) -> "ATSPInstance":
        """
        Parse an ATSP FULL_MATRIX TSPLIB file.
        This parser is conservative: it will parse numeric values from EDGE_WEIGHT_SECTION.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        dim = 0
        in_weights = False
        vals: List[float] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # accept "DIMENSION" (maybe "DIMENSION : 100") robustly
            if s.upper().startswith("DIMENSION"):
                # split by non-digits - do best-effort
                parts = s.replace(":", " ").split()
                for token in parts[::-1]:
                    if token.isdigit():
                        dim = int(token)
                        break
            elif s.upper().startswith("EDGE_WEIGHT_SECTION"):
                in_weights = True
                continue
            elif in_weights:
                if s.upper() == "EOF":
                    break
                # extend numeric tokens
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except Exception:
                        # ignore non-numeric tokens, but warn
                        logging.debug("Non-numeric token in EDGE_WEIGHT_SECTION: %s", tok)

        if dim <= 0:
            raise ValueError("DIMENSION not found or invalid in TSPLIB file.")

        if len(vals) != dim * dim:
            raise ValueError(f"Edge weight section had {len(vals)} numbers but expected {dim*dim}")

        adj = np.array(vals, dtype=float).reshape(dim, dim)
        return cls(adj=adj)

    # ---------- TSPLIB / LKH interface ----------

    def _to_tsplib_string(self, fixed_edge: Optional[Tuple[int, int]] = None) -> str:
        """
        Build a TSPLIB formatted string for an ATSP FULL_MATRIX problem.
        NOTE: TSPLIB expects integer weights in explicit matrices in practice; we round entries.
        """
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
        # Round adjacency to ints for TSPLIB (typical expectation). Use int(round()) to avoid truncation bias.
        for i in range(n):
            rows.append(" ".join(str(int(round(self.adj[i, j]))) for j in range(n)))
        body = "\n".join(rows) + "\n"

        if fixed_edge is None:
            return f"{header}{body}EOF\n"

        # LKH uses 1-based indexing for fixed edges
        i, j = fixed_edge
        fixed = (
            "FIXED_EDGES_SECTION\n"
            f"{i+1} {j+1}\n"
            "-1\n"
        )
        return f"{header}{body}{fixed}EOF\n"

    def solve_lkh(self, lkh_path: str = "../../lib/LKH-3") -> Tuple[List[int], float]:
        """
        Solve instance via LKH and cache tour/cost.
        Returns tour as zero-based closed cycle [v0, v1, ..., v0] and cost.
        """
        tsp_str = self._to_tsplib_string()
        problem = tsplib95.parse(tsp_str)
        solution = lkh.solve(lkh_path, problem=problem)
        # solution[0] is a list of 1-based node indices representing a cycle
        tour = [v - 1 for v in solution[0]] + [solution[0][0] - 1]
        cost = self._tour_cost(tour)
        self.tour, self.cost = tour, cost
        return tour, cost

    def solve_lkh_with_fixed_edge(
        self, edge: Tuple[int, int], lkh_path: str = "../LKH-3.0.9/LKH"
    ) -> Tuple[List[int], float]:
        """
        Solve forcing a directed edge (i->j) to be part of the tour using FIXED_EDGES_SECTION.
        Does not cache tour/cost in self (keeps caller in control).
        """
        tsp_str = self._to_tsplib_string(fixed_edge=edge)
        problem = tsplib95.parse(tsp_str)
        solution = lkh.solve(lkh_path, problem=problem)
        tour = [v - 1 for v in solution[0]] + [solution[0][0] - 1]
        cost = self._tour_cost(tour)
        return tour, cost

    def _tour_cost(self, tour: List[int]) -> float:
        """Sum adjacency along closed tour."""
        # tour is expected to have last element equal to first element (closed)
        if len(tour) < 2:
            return 0.0
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
        - mode="row_best": heuristic: regret(i->j) = w(i,j) - min_{k!=i} w(i,k)
        - mode="fixed_edge_lkh": accurate but heavy; regret = (opt_cost_with_edge - opt_cost) / opt_cost
        Important:
          - base_tour/base_cost will be computed if not provided (cost fallback to small eps for numeric safety).
          - in fixed_edge_lkh, edges that are already in base_tour get regret 0 and are skipped.
        """
        n = self.adj.shape[0]
        regrets = np.zeros((n, n), dtype=float)

        if mode == "row_best":
            # For each row i, compute min outgoing excluding self-loop
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
            # numeric safety: avoid divide-by-zero
            base_cost = float(base_cost if abs(base_cost) > 1e-12 else 1e-6)

            # Determine edges in base tour to skip (their regret is 0)
            in_sol = set()
            if base_tour is not None:
                # base_tour is closed (last == first)
                in_sol = {(base_tour[i], base_tour[i + 1]) for i in range(len(base_tour) - 1)}

            # Prepare list of edges to evaluate (all directed except self-loops and base edges)
            edge_list = [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in in_sol]

            # pre-fill regrets for edges in the base tour with 0.0
            regrets = np.zeros((n, n), dtype=float)
            for (i, j) in in_sol:
                regrets[i, j] = 0.0

            # If there are no edges to evaluate, just return
            if len(edge_list) == 0:
                self.regrets = regrets
                return regrets

            if parallel:
                # Build worker args list
                args_list = [(self.adj, e, base_cost, lkh_path) for e in edge_list]
                # choose reasonable pool size
                cpu = mp.cpu_count()
                pool_size = processes or min(cpu, max(1, len(args_list)))
                logging.info("Launching Pool with %d workers to evaluate %d edges (skipped %d base edges).",
                             pool_size, len(args_list), len(in_sol))
                with mp.Pool(processes=pool_size) as pool:
                    for (i, j), r in pool.imap_unordered(compute_edge_regret_worker, args_list):
                        regrets[i, j] = r
            else:
                logging.info("Evaluating %d edges sequentially (skipped %d base edges).", len(edge_list), len(in_sol))
                for e in edge_list:
                    (i, j), r = compute_edge_regret_worker((self.adj, e, base_cost, lkh_path))
                    regrets[i, j] = r

            self.regrets = regrets
            return regrets

        raise ValueError(f"Unknown regret mode: {mode}")

    # ---------- Export ----------

    def to_networkx(self) -> nx.DiGraph:
        """
        Export instance to a directed NetworkX graph.
        Edge attributes: weight, optional 'regret', 'in_solution'
        Graph attributes: 'tour' (list) and 'cost' if available.
        """
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

# -------------------------
# HCP, HPP, K-TSP, Multi-agent TSP instance definitions
# -------------------------

@dataclass
class HCPInstance:
    adj: np.ndarray  # (n, n) adjacency (bool)
    hasCycle: Optional[bool] = None
    cycle: Optional[List[int]] = None  # List of nodes in cycle

    @classmethod
    def from_random(
        cls,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> "HCPInstance":
        """
        Generate a random boolean adjacency matrix for HCP.
        edge_prob: probability of an edge between any two nodes (excluding self-loops).
        """
        edge_prob = np.log(n) / n
        if rng is None:
            rng = np.random.default_rng()
        adj = rng.random((n, n)) < edge_prob
        np.fill_diagonal(adj, False)  # No self-loops
        return cls(adj=adj)

    def to_atsp(self) -> "ATSPInstance":
        """
        Convert this HPPInstance to ATSPInstance.
        False -> 1000*n, True -> 1, diagonal -> 0
        """
        n = self.adj.shape[0]
        adj = np.where(self.adj, 1, 1_000 * n).astype(float)
        np.fill_diagonal(adj, 0.0)
        return ATSPInstance(adj=adj)
    
    @classmethod
    def from_atsp_instance(cls, atsp_instance: "ATSPInstance") -> "HCPInstance":
        """
        Convert an ATSPInstance to an HCPInstance.
        - Converts the weight matrix (adj) to a boolean adjacency matrix.
        - Extracts the Hamiltonian cycle from the ATSP tour.
        - Validates the cycle and returns an HCPInstance.

        Args:
            atsp_instance (ATSPInstance): The ATSP instance containing the adjacency matrix and solution.

        Returns:
            HCPInstance: The corresponding HCP instance.
        """
        n = atsp_instance.adj.shape[0]

        # Convert the weight matrix to a boolean adjacency matrix
        adj = (atsp_instance.adj == 1)  # True for edges with weight exactly 1, False otherwise
        # Extract the cycle from the ATSP tour
        if atsp_instance.tour is None:
            raise ValueError("ATSPInstance does not have a valid tour.")

        # Validate the cycle forms a valid Hamiltonian cycle
        if len(atsp_instance.tour) != n:
            raise ValueError("ATSP tour does not form a valid Hamiltonian cycle.")

        # Check if the ATSP tour contains any edge that is not True in the boolean adjacency
        for i in range(len(atsp_instance.tour)):
            u, v = atsp_instance.tour[i], atsp_instance.tour[(i + 1)%len(atsp_instance.tour)]
            if not adj[u, v]:
            # If any edge in the tour is not present, no cycle exists
                return cls(adj=adj, hasCycle=False, cycle=None)

        
        cycle = atsp_instance.tour.copy()

        # Return the HCPInstance
        return cls(adj=adj, hasCycle=True, cycle=cycle)

@dataclass
class HPPInstance:
    adj: np.ndarray  # (n, n) adjacency (bool)
    hasPath: Optional[bool] = None
    path: Optional[List[int]] = None  # List of nodes in path

    @classmethod
    def from_random(
        cls,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> "HCPInstance":
        """
        Generate a random boolean adjacency matrix for HCP.
        edge_prob: probability of an edge between any two nodes (excluding self-loops).
        """
        edge_prob = np.log(n) / n
        if rng is None:
            rng = np.random.default_rng()
        adj = rng.random((n, n)) < edge_prob
        np.fill_diagonal(adj, False)  # No self-loops
        return cls(adj=adj)

    def to_atsp(self) -> "ATSPInstance":
        """
        Convert this HPPInstance to ATSPInstance.
        False -> 1000*n, True -> 1, diagonal -> 0
        """
        n = self.adj.shape[0]
        adj = np.where(self.adj, 1, 1_000 * n).astype(float)
        np.fill_diagonal(adj, 0.0)
        return ATSPInstance(adj=adj)
    
    @classmethod
    def from_atsp_instance(cls, atsp_instance: ATSPInstance) -> HCPInstance:
        """
        Convert an ATSPInstance to an HCPInstance.
        - Converts the weight matrix (adj) to a boolean adjacency matrix.
        - Extracts the Hamiltonian cycle from the ATSP tour.
        - Validates the cycle and returns an HCPInstance.
        A valid path is found if at most one edge in the tour is False.
        If an edge is False, the path should start with the node the edge targets.
        """
        n = atsp_instance.adj.shape[0]
        adj = (atsp_instance.adj == 1)  # True for edges with weight exactly 1, False otherwise

        if atsp_instance.tour is None:
            raise ValueError("ATSPInstance does not have a valid tour.")

        # Validate the cycle forms a valid Hamiltonian cycle
        if len(atsp_instance.tour) != n:
            raise ValueError("ATSP tour does not form a valid Hamiltonian cycle.")

        # Count the number of False edges in the tour
        false_edges = []
        for i in range(len(atsp_instance.tour)):
            u, v = atsp_instance.tour[i], atsp_instance.tour[(i + 1) % len(atsp_instance.tour)]
            if not adj[u, v]:
                false_edges.append((u, v))

        if len(false_edges) == 0:
            # All edges are True, valid cycle
            cycle = atsp_instance.tour.copy()
            return cls(adj=adj, hasCycle=True, cycle=cycle)
        elif len(false_edges) == 1:
            # One edge is False, valid path starting at the target of the False edge
            _, start_node = false_edges[0]
            # Rotate tour so it starts at start_node
            tour = atsp_instance.tour.copy()
            idx = tour.index(start_node)
            path = tour[idx:] + tour[:idx]
            return cls(adj=adj, hasCycle=False, cycle=path)
        else:
            # More than one False edge, not a valid path/cycle
            return cls(adj=adj, hasCycle=False, cycle=None)
        
@dataclass
class KTSPInstance:
    adj: np.ndarray  # (n, n) adjacency (float)
    k: int = 1
    tours: Optional[List[List[int]]] = None  # List of k tours
    costs: Optional[List[float]] = None

    @classmethod
    def from_random(
        cls,
        n: int,
        k: int,
        weight_min: int = 100,
        weight_max: int = 1000,
        rng: Optional[np.random.Generator] = None,
        enforce_metric: bool = True,
    ) -> "KTSPInstance":
        atsp = ATSPInstance.from_random(n, weight_min, weight_max, rng, enforce_metric)
        return cls(adj=atsp.adj.copy(), k=k)
    
    def to_atsp(self) -> "ATSPInstance":
        """
        Convert KTSPInstance to ATSPInstance by copying the first node k times.
        - Edges between the copies are set to 1.
        - All other edges are increased by n * 1000.
        """
        n = self.adj.shape[0]
        k = self.k
        new_n = n + k - 1
        # Create expanded adjacency matrix
        expanded = np.full((new_n, new_n), n * 1000, dtype=float)
        # Copy original adjacency
        expanded[k-1:, k-1:] = self.adj[1:, 1:]
        # Copy edges from original node 0 to all nodes
        for i in range(k):
            expanded[i, k-1:] = self.adj[0, 1:]
            expanded[k-1:, i] = self.adj[1:, 0]
        # Set edges between the k copies to 1, except self-loops
        for i in range(k):
            for j in range(k):
                if i != j:
                    expanded[i, j] = 1.0
        # Set diagonal to 0
        np.fill_diagonal(expanded, 0.0)
        return ATSPInstance(adj=expanded)
    
    @classmethod
    def from_atsp_instance(cls, atsp_instance: ATSPInstance, k: int) -> "KTSPInstance":
        """
        Convert an ATSPInstance (expanded from KTSP) back to KTSPInstance with solution.
        - Validates that the ATSP tour is of correct size.
        - Validates that the k copies of the main node have weight 1.0 (except diagonal).
        - Validates that the k copies have the same weights to/from other nodes.
        - Extracts k tours from the ATSP tour (if available).
        - Computes individual costs for each tour.
        - Returns KTSPInstance with tours and costs.
        """
        n = atsp_instance.adj.shape[0]
        orig_n = n - k + 1

        # Validate that the k copies of the main node have weight 1.0 (except diagonal)
        for i in range(k):
            for j in range(k):
                if i != j and not atsp_instance.adj[i, j] == 1.0:
                    raise ValueError(f"Edge ({i},{j}) between main node copies does not have weight 1.0.")

        # Validate that the k copies have the same weights to/from other nodes
        for i in range(k):
            for v in range(k, n):
                ref_out = atsp_instance.adj[i, v]
                ref_in = atsp_instance.adj[v, i]
                for j in range(k):
                    if atsp_instance.adj[j, v] != ref_out:
                        raise ValueError(f"Copy node {j} has different outgoing weight to node {v} than copy {i}.")
                    if atsp_instance.adj[v, j] != ref_in:
                        raise ValueError(f"Copy node {j} has different incoming weight from node {v} than copy {i}.")

        # Validate that the ATSP tour is of correct size
        if atsp_instance.tour is None:
            return cls(adj=atsp_instance.adj[k-1:, k-1:].copy(), k=k, tours=None, costs=None)
        if len(atsp_instance.tour) != n:
            raise ValueError(f"ATSP tour length {len(atsp_instance.tour)} does not match expected {n}.")

        # Extract the original adjacency matrix
        adj = atsp_instance.adj[k-1:, k-1:].copy()

        # Split the ATSP tour into k separate tours
        tours = []
        costs = []
        current_tour = []
        current_cost = 0.0
        # Map each value to u - (k-1), central nodes < k set to 0
        mapped_tour = []
        for u in atsp_instance.tour:
            if u < k:
                mapped_tour.append(0)
            else:
                mapped_tour.append(u - (k - 1))

        # Split the mapped tour into segments at 0, creating an embedded array (list of lists)
        split_indices = [i for i, v in enumerate(mapped_tour) if v == 0]
        
        # If the first value is not 0, merge last and first segments
        segments = []
        for i in range(len(split_indices)):
            start = split_indices[i]
            end = split_indices[i + 1] if i + 1 < len(split_indices) else len(mapped_tour)
            seg = mapped_tour[start:end]
            if seg:
                segments.append(seg)

        # Merge last and first if needed
        if segments and segments[0][0] != 0:
            segments[0] = segments[-1] + segments[0]
            segments = segments[:-1]

        # Evaluate each tour segment
        costs = []
        for seg in segments:
            # Calculate cost for this segment
            cost = 0.0
            for i in range(len(seg)):
                cost += adj[seg[i], seg[(i + 1)%len(seg)]]
            costs.append(cost)
        # Ensure all k tours are extracted
        if len(tours) != k:
            raise ValueError("ATSP tour does not match expected k tours.")
        # Validate total cost matches
        if not np.isclose(sum(costs), atsp_instance.cost):
            raise ValueError("ATSP cost does not match sum of k-tour costs.")
        return cls(adj=adj, k=k, tours=tours, costs=costs)

@dataclass
class MultiAgentTSPInstance:
    adj: np.ndarray  # (n, n) adjacency (float)
    agent_tours: Optional[List[List[int]]] = None
    agent_costs: Optional[List[float]] = None

    @classmethod
    def from_random(
        cls,
        n: int,
        weight_min: int = 100,
        weight_max: int = 1000,
        rng: Optional[np.random.Generator] = None,
        enforce_metric: bool = True,
    ) -> "MultiAgentTSPInstance":
        atsp = ATSPInstance.from_random(n, weight_min, weight_max, rng, enforce_metric)
        return cls(adj=atsp.adj.copy())

    
    def to_ktsp_instances(self, max_agents: Optional[int] = None) -> List["KTSPInstance"]:
        """
        Create KTSPInstance objects for agent counts from 1 up to max_agents (inclusive).
        If max_agents is None, use n (number of nodes).
        """
        n = self.adj.shape[0]
        if max_agents is None:
            max_agents = n
        ktsp_list = []
        for k in range(1, max_agents + 1):
            ktsp_list.append(KTSPInstance(adj=self.adj.copy(), k=k))
        return ktsp_list

# -------------------------
# Dataset builder
# -------------------------
class ATSPDatasetBuilder:
    def __init__(
        self,
        n_nodes: int,
        n_instances: int,
        output_dir: pathlib.Path,
        weight_min: int = 100,
        weight_max: int = 1000,
        lkh_path: str = "../LKH-3.0.9/LKH",
        seed: Optional[int] = None,
        problem_type: str = "ATSP",  # <-- new attribute
        k_agents: Optional[int] = None,  # for KTSP
    ):
        self.n_nodes = n_nodes
        self.n_instances = n_instances
        self.output_dir = pathlib.Path(output_dir)
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.lkh_path = lkh_path
        self.seed = seed
        self.problem_type = problem_type  # <-- store problem type
        self.k_agents = k_agents

    # --- generators ---

    def _iter_random_instances(self) -> Iterable[ATSPInstance]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_instances):
            if self.problem_type == "HCP":
                hcp_instance = HCPInstance.from_random(self.n_nodes, rng=rng)
                yield hcp_instance.to_atsp()
            elif self.problem_type == "HPP":
                hpp_instance = HPPInstance.from_random(self.n_nodes, rng=rng)
                yield hpp_instance.to_atsp()
            elif self.problem_type == "KTSP":
                k = self.k_agents if self.k_agents is not None else 2
                ktsp_instance = KTSPInstance.from_random(
                    self.n_nodes, k, self.weight_min, self.weight_max, rng=rng
                )
                yield ktsp_instance.to_atsp()
            else:
                yield ATSPInstance.from_random(
                    self.n_nodes, self.weight_min, self.weight_max, rng=rng
                )

    def _iter_tsplib_dir(self, dir_path: pathlib.Path) -> Iterable[ATSPInstance]:
        for name in sorted(os.listdir(dir_path)):
            p = dir_path / name
            if p.is_file():
                yield ATSPInstance.from_tsplib_file(p)

    def _iter_pt_file(self, pt_path: pathlib.Path) -> Iterable[ATSPInstance]:
        """
        Load a .pt file (torch.save) expected to contain either:
          - a single (n, n) matrix, or
          - an array/tensor with shape (num_samples, n, n)
        Yields ATSPInstance objects for each sample.
        This function will raise informative errors if torch is not installed or shape is unexpected.
        """
        if torch is None:
            raise ImportError(
                "PyTorch (torch) is required to load .pt files. Install it with 'pip install torch'."
            )

        pt_path = pathlib.Path(pt_path)
        if not pt_path.is_file():
            raise FileNotFoundError(f".pt file not found: {pt_path}")

        logging.info("Loading .pt file from %s", pt_path)
        data = torch.load(str(pt_path), map_location="cpu", weights_only=True)  # try to load on CPU
        arr = None
        if hasattr(data, "numpy"):
            # torch.Tensor -> numpy
            arr = data.numpy()
        else:
            # maybe saved as numpy array already or list; coerce to numpy
            arr = np.array(data)

        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            # single adjacency matrix
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(f"Loaded matrix shape {arr.shape} is not square.")
            logging.info("Loaded a single adjacency matrix with shape %s", arr.shape)
            yield ATSPInstance(adj=arr)
            return
        if arr.ndim == 3:
            num = arr.shape[0]
            logging.info("Loaded %d adjacency matrices with shape %s", num, arr.shape[1:])
            for k in range(num):
                mat = arr[k]
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError(f"Sample {k} is not square: {mat.shape}")
                yield ATSPInstance(adj=np.array(mat, dtype=float))
            return
        raise ValueError(f"Unsupported tensor shape {arr.shape}; expected (n,n) or (num,n,n)")

    # --- build pipeline ---

    def build_and_save(
        self,
        from_tsplib_dir: Optional[pathlib.Path] = None,
        from_pt_file: Optional[pathlib.Path] = None,
        regret_mode: str = "row_best",
        compute_regrets: bool = True,
        parallel: bool = False,
        processes: Optional[int] = None,
        save_graph_pickles: bool = True,
        save_summary_csv: bool = True,
    ) -> None:
        """
        Main pipeline:
          - create/parse instances
          - solve with LKH (always)
          - optionally label regrets (row_best or fixed_edge_lkh)
          - save pickles + summary
        Notes:
          - if from_pt_file is supplied, we iterate over that file's contents and ignore self.n_instances.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_lines = ["id,n_nodes,cost,regrets_computed,regret_mode\n"]

        # choose source iterator
        if from_pt_file is not None:
            src_iter = self._iter_pt_file(pathlib.Path(from_pt_file))
        elif from_tsplib_dir is None:
            src_iter = self._iter_random_instances()
        else:
            src_iter = self._iter_tsplib_dir(pathlib.Path(from_tsplib_dir))

        total = None
        # If using .pt, we can try to determine total from the file (not strictly required)
        if from_pt_file is not None and torch is not None:
            try:
                data = torch.load(str(from_pt_file), map_location="cpu", weights_only=True)
                arr = np.asarray(data.numpy() if hasattr(data, "numpy") else data)
                if arr.ndim == 3:
                    total = int(arr.shape[0])
            except Exception:
                total = None

        # iterate and process
        count = 0
        for inst in src_iter:
            count += 1
            logging.info("Processing instance #%d", count)

            # Solve base tour/cost — always required (user asked: "for sure the solution should be done")
            try:
                tour, cost = inst.solve_lkh(self.lkh_path)
            except Exception as e:
                logging.exception("LKH failed on instance #%d: %s", count, e)
                # Still attempt to save instance with no tour
                tour, cost = None, float("inf")
                inst.tour, inst.cost = tour, cost

            # Labels (only if requested)
            if compute_regrets:
                try:
                    inst.label_regrets(
                        mode=regret_mode,
                        lkh_path=self.lkh_path,
                        base_tour=tour,
                        base_cost=cost,
                        parallel=parallel,
                        processes=processes,
                    )
                    regrets_computed = True
                except Exception as e:
                    logging.exception("Regret labeling failed on instance #%d: %s", count, e)
                    inst.regrets = None
                    regrets_computed = False
            else:
                inst.regrets = None
                regrets_computed = False

            # Save
            uid = uuid.uuid4().hex
            n_nodes = inst.adj.shape[0]
            # if cost is None, coerce to inf in summary
            cost_val = float(inst.cost) if inst.cost is not None else float("inf")
            summary_lines.append(f"{uid},{n_nodes},{cost_val},{int(regrets_computed)},{regret_mode}\n")

            if save_graph_pickles:
                G = inst.to_networkx()
                filepath = self.output_dir / f"{uid}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(G, f)

            # Optional: break if we were given a fixed count and not using a .pt file
            if (from_pt_file is None) and (self.n_instances is not None) and (count >= self.n_instances):
                break

        if save_summary_csv:
            with open(self.output_dir / "summary.csv", "w") as f:
                f.writelines(summary_lines)

        logging.info("Finished building dataset: saved %d instances to %s", count, self.output_dir)

# =========================================================
# CLI (instances only; dataset building)
# =========================================================

def _main():
    parser = argparse.ArgumentParser(description="Generate/label ATSP dataset.")
    # keep positional args for backward compatibility; they will be ignored when from_pt_file is used.
    parser.add_argument("n_samples", type=int, help="Number of instances to generate (ignored if --from_pt_file provided)")
    parser.add_argument("n_nodes", type=int, help="Number of nodes per instance (ignored if --from_pt_file provided)")
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
    parser.add_argument("--from_pt_file", type=pathlib.Path, default=None,
                        help="If set, read instances from a single .pt file (numpy array or torch tensor saved with torch.save).")
    parser.add_argument("--no_regret", action="store_true",
                        help="If set, skip regret labeling (still solves each instance to get base tour/cost).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for instance generation (optional).")
    parser.add_argument("--problem_type", type=str, default="ATSP", choices=["ATSP", "HCP", "HPP", "KTSP"],
                        help="Type of problem to generate: ATSP (default), HCP, HPP, or KTSP.")
    parser.add_argument("--n_agents", type=int, default=None,
                        help="Number of agents (for KTSP problems).")
    args = parser.parse_args()

    # Adjust subdir name based on problem type and number of agents (if KTSP)
    if args.problem_type == "KTSP":
        k_agents = args.n_agents if args.n_agents is not None else 2
        subdir_name = f"{args.problem_type}_{args.n_nodes}x{args.n_samples}_k{k_agents}"
    else:
        subdir_name = f"{args.problem_type}_{args.n_nodes}x{args.n_samples}"
    out_dir = args.out_dir / subdir_name
    os.makedirs(out_dir, exist_ok=True)
    builder = ATSPDatasetBuilder(
        n_nodes=args.n_nodes,
        n_instances=args.n_samples,
        output_dir=out_dir,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        lkh_path=args.lkh_path,
        seed=args.seed,
        problem_type=args.problem_type,
        k_agents=args.n_agents if args.problem_type == "KTSP" else None,
    )

    builder.build_and_save(
        from_tsplib_dir=args.from_tsplib_dir,
        from_pt_file=args.from_pt_file,
        regret_mode=args.regret_mode ,
        compute_regrets= not args.no_regret,
        parallel=args.parallel if args.problem_type == "ATSP" else None,
        processes=args.processes if args.problem_type == "ATSP" else None,
        save_graph_pickles=True,
        save_summary_csv=True,
    )


if __name__ == "__main__":
    _main()
