import pathlib
import pickle
from typing import Tuple, List, Set, Iterable, Optional
import networkx as nx
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from pathlib import Path

from .scalers import FeatureScaler
from .template_manager import TemplateManager


class BaseATSPDataset:
    """Base utilities for ATSP datasets.

    Responsibilities
    - load instance list
    - produce a covering of node pairs by subgraphs of fixed size
    - build induced subgraph views for processing
    """

    def __init__(self, data_dir: pathlib.Path, atsp_size: int):
        self.data_dir = pathlib.Path(data_dir)
        self.atsp_size = atsp_size
    
    def _load_instance_list(self, split: str) -> List[str]:
        split_file = self.data_dir / f"{split}.txt"
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def greedy_cover_subsets(self, n: int, s: int) -> List[Set[int]]:
        """Greedy algorithm to cover all unordered node pairs by subsets of size s.

        Returns list of node-index sets. Each unordered pair {i,j} appears in at least
        one returned subset.

        Notes on complexity: for n large this can be memory heavy because the set of
        uncovered pairs has size n(n-1)/2. Tradeoffs: increase s to reduce number of
        subsets, or generate subsets on the fly.
        """
        if s < 2:
            raise ValueError('s must be at least 2')

        # Represent unordered pair with tuple (min, max)
        uncovered = set((i, j) for i in range(n) for j in range(i + 1, n))
        subsets: List[Set[int]] = []

        # Precompute adjacency helper for speed
        while uncovered:
            current: Set[int] = set()
            # Maintain counts of potential gains for candidate nodes
            # At start, pick node that participates in most uncovered pairs
            # To save compute, initialize by degree on uncovered
            # Simple heuristic: pick node with largest remaining uncovered degree
            # compute uncovered degree
            deg = [0] * n
            for (a, b) in uncovered:
                deg[a] += 1
                deg[b] += 1

            # Build current subset greedily up to size s
            while len(current) < s:
                # choose node not in current with max gain defined as number of uncovered
                # pairs it would cover when added to current
                best_node = None
                best_gain = -1
                for v in range(n):
                    if v in current:
                        continue
                    # gain is count of uncovered pairs between v and nodes already in current
                    if not current:
                        gain = deg[v]
                    else:
                        gain = 0
                        for u in current:
                            a, b = (u, v) if u < v else (v, u)
                            if (a, b) in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_node = v
                if best_node is None:
                    # no candidate found, fill randomly until we reach size s
                    for v in range(n):
                        if v not in current:
                            current.add(v)
                            break
                else:
                    current.add(best_node)

                if len(current) == n:
                    break

            # remove covered pairs
            to_remove = set()
            current_list = sorted(current)
            for i_idx in range(len(current_list)):
                for j_idx in range(i_idx + 1, len(current_list)):
                    a = current_list[i_idx]
                    b = current_list[j_idx]
                    pair = (a, b)
                    if pair in uncovered:
                        to_remove.add(pair)
            uncovered.difference_update(to_remove)
            subsets.append(current)

        return subsets

    def induced_subgraph(self, G: nx.DiGraph, nodes: Iterable[int]) -> nx.DiGraph:
        """Return induced directed subgraph of G on nodes preserving edge data."""
        sub = G.subgraph(nodes).copy()
        return sub


class ATSPDatasetDGL(BaseATSPDataset):
    """ATSP dataset optimized for DGL models with optional large-graph handling.

    Key extension compared to the previous class
    - support for covering a large graph by smaller subgraphs of size sub_size
    - ability to load a small DGL template once and reuse it for all subgraphs
    - helper to apply a model to every subgraph and merge predicted regrets back
      to the original graph
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        atsp_size: int,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        device: str = 'cpu',
        undirected: bool = True,
        load_once: bool = True,
        sub_size: Optional[int] = None,
        template_dir: Optional[pathlib.Path] = None,   # <--- add this
    ):
        super().__init__(data_dir, atsp_size)
        self.data_dir = pathlib.Path(data_dir)
        self.template_dir = pathlib.Path(template_dir) if template_dir is not None else self.data_dir
        self.split = split
        self.atsp_size = atsp_size
        self.device = device
        self.relation_types = sorted(relation_types)
        self.undirected = undirected
        self.load_once = load_once
        self.sub_size = sub_size

        self.instances = self._load_instance_list(split)
        self.scalers = FeatureScaler.load(self.data_dir / 'scalers.pkl')

        self.num_edges = atsp_size * (atsp_size - 1)

        if self.load_once:
            self.template = self._load_template(full_size=self.atsp_size)
            if self.sub_size and self.sub_size != self.atsp_size:
                self.template_small = self._load_template(full_size=self.sub_size)
            else:
                self.template_small = self.template
            self.graphs = [self._process_graph(self._load_instance_graph(f)) for f in self.instances]
        else:
            self.template = None
            self.template_small = None
            self.graphs = None

    def _cover_complete_by_block_pairs(self, n: int, sub_size: int) -> List[List[int]]:
        """Fast pair-cover for complete graphs using union of half-size blocks.

        - Partition nodes into disjoint blocks of size b=sub_size//2 (last may be smaller).
        - For every unordered pair of blocks (i<j), form subset = blocks[i] ∪ blocks[j].
        - If subset has fewer than sub_size nodes (due to final short block), fill by
          appending nodes from subsequent blocks (wrapping) until reaching sub_size.

        Returns an ordered list of node lists (length == sub_size each).
        """
        b = max(2, sub_size // 2)
        blocks: List[List[int]] = []
        for start in range(0, n, b):
            end = min(n, start + b)
            blocks.append(list(range(start, end)))
        subsets: List[List[int]] = []
        B = len(blocks)
        for i in range(B):
            for j in range(i + 1, B):
                union_nodes = list(blocks[i]) + [v for v in blocks[j] if v not in blocks[i]]
                if len(union_nodes) < sub_size:
                    k = (j + 1) % B
                    while len(union_nodes) < sub_size and B > 0:
                        for v in blocks[k]:
                            if v not in union_nodes:
                                union_nodes.append(v)
                                if len(union_nodes) == sub_size:
                                    break
                        k = (k + 1) % B
                        if k == i:
                            break
                subsets.append(union_nodes[:sub_size])
        return subsets

    def _load_template(self, full_size: int) -> dgl.DGLGraph:
        # prefer sub template dir for sub_size, and data dir for full size
        candidate_dirs = (
            [self.data_dir, self.template_dir] if full_size == self.atsp_size
            else [self.template_dir, self.data_dir]
        )
        last_path = None
        for base in candidate_dirs:
            template_path = TemplateManager.get_template_path(base, full_size, self.relation_types)
            last_path = template_path
            if Path(template_path).exists():
                graphs, _ = dgl.load_graphs(str(template_path))
                g = graphs[0]
                if self.undirected:
                    g = dgl.add_reverse_edges(g)
                return g.to(self.device)
        raise FileNotFoundError(f'No template for size {full_size} in {candidate_dirs}. Last tried {last_path}')



    def _load_instance_graph(self, instance_file: str) -> nx.DiGraph:
        with open(self.data_dir / instance_file, 'rb') as f:
            return pickle.load(f)

    def _extract_features_from_graph(self, G: nx.DiGraph, node_order: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract edge features following a consistent ordering derived from node_order.

        node_order lists nodes in the same order as the DGL template expects.
        Returns tensors for weight, regret, in_solution with shape (m,) where m = s(s-1)
        """
        s = len(node_order)
        m = s * (s - 1)
        weights = torch.empty(m, dtype=torch.float32)
        regrets = torch.empty(m, dtype=torch.float32)
        in_solution = torch.empty(m, dtype=torch.float32)
        edge_idx = 0
        for i in range(s):
            for j in range(s):
                if i == j:
                    continue
                u = node_order[i]
                v = node_order[j]
                edge_data = G.get_edge_data(u, v, default=None)
                if edge_data is None:
                    # if missing edge, default values
                    weights[edge_idx] = 0.0
                    regrets[edge_idx] = 0.0
                    in_solution[edge_idx] = 0.0
                else:
                    weights[edge_idx] = edge_data.get('weight', 0.0)
                    regrets[edge_idx] = edge_data.get('regret', 0.0)
                    in_solution[edge_idx] = edge_data.get('in_solution', 0.0)
                edge_idx += 1
        return weights, regrets, in_solution

    def _process_graph(self, G: nx.DiGraph) -> dgl.DGLGraph:
        weights, regrets, in_solution = self._extract_features_from_graph(G, list(range(self.atsp_size)))
        graph = self.template.clone() if self.template is not None else self._load_template(self.atsp_size)
        graph.ndata['weight'] = self.scalers.transform(weights, 'weight').unsqueeze(1).to(self.device)
        graph.ndata['regret'] = self.scalers.transform(regrets, 'regret').unsqueeze(1).to(self.device)
        graph.ndata['in_solution'] = in_solution.unsqueeze(1).to(self.device)
        if hasattr(G, 'graph'):
            graph.graph_attr = dict(G.graph)
        return graph

    def cover_and_predict_full_graph(self, model, G: nx.DiGraph, sub_size: Optional[int] = None, batch_size: int = 16) -> None:
        """Cover the node pairs of G with subgraphs of size sub_size, run model on each
        subgraph and write predicted regret values back into G as attribute 'regret_pred'.

        If sub_size is None, use self.sub_size or self.atsp_size.
        """
        if sub_size is None:
            sub_size = self.sub_size or self.atsp_size
        n = self.atsp_size
        if sub_size >= n:
            # simple path: process full graph in one shot
            H = self.get_scaled_features(G)
            x = H.ndata['weight']
            with torch.no_grad():
                y = model(H, x)
            regret_pred = self.scalers.inverse_transform(y.cpu().flatten(), 'regret')
            # assign
            edge_idx = 0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if (i, j) in G.edges:
                        G.edges[(i, j)]['regret_pred'] = max(float(regret_pred[edge_idx]), 0.0)
                    edge_idx += 1
            return

        # Build covering subsets. For complete graphs, use fast block-pair cover.
        if len(G.edges) == n * (n - 1) and (sub_size % 2 == 0):
            subset_orders = self._cover_complete_by_block_pairs(n, sub_size)
        else:
            subset_orders = [sorted(list(s)) for s in self.greedy_cover_subsets(n, sub_size)]

        # Verify directed-pair coverage (diagnostic, cheap)
        cov_counts = np.zeros((n, n), dtype=np.int32)
        for node_order in subset_orders:
            s = len(node_order)
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    u, v = node_order[i], node_order[j]
                    cov_counts[u, v] += 1
        uncovered = np.argwhere((cov_counts == 0) & (~np.eye(n, dtype=bool)))
        if uncovered.size > 0:
            # raise explicit error to avoid silent bad merges
            raise RuntimeError(f"Coverage incomplete: found {uncovered.shape[0]} uncovered directed pairs out of {n*(n-1)}")

        # Ensure we have a small template loaded
        template_small = self.template_small or self._load_template(sub_size)

        # Prepare storage for aggregated predictions and counts (dense arrays for speed/memory)
        pred_accum = np.zeros((n, n), dtype=np.float32)
        pred_count = np.zeros((n, n), dtype=np.int32)

        # Process subsets in batches
        batch_graphs = []
        batch_node_orders = []
        for node_order in subset_orders:
            subG = self.induced_subgraph(G, node_order)
            weights, regrets, in_solution = self._extract_features_from_graph(subG, node_order)
            g = template_small.clone()
            g.ndata['weight'] = self.scalers.transform(weights, 'weight').unsqueeze(1).to(self.device)
            g.ndata['regret'] = self.scalers.transform(regrets, 'regret').unsqueeze(1).to(self.device)
            g.ndata['in_solution'] = in_solution.unsqueeze(1).to(self.device)
            batch_graphs.append(g)
            batch_node_orders.append(node_order)

            if len(batch_graphs) >= batch_size:
                self._run_batch_and_merge(model, batch_graphs, batch_node_orders, pred_accum, pred_count)
                batch_graphs = []
                batch_node_orders = []

        if batch_graphs:
            self._run_batch_and_merge(model, batch_graphs, batch_node_orders, pred_accum, pred_count)

        # Write averaged predictions into G
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c = pred_count[i, j]
                if c > 0:
                    G.edges[(i, j)]['regret_pred'] = float(pred_accum[i, j] / c)
                else:
                    G.edges[(i, j)]['regret_pred'] = float(G.edges[(i, j)].get('regret', 0.0))

    def _run_batch_and_merge(self, model, batch_graphs: List[dgl.DGLGraph], batch_node_orders: List[List[int]], pred_accum: np.ndarray, pred_count: np.ndarray):
        batch = dgl.batch(batch_graphs)
        x = batch.ndata['weight']
        with torch.no_grad():
            y = model(batch, x)
        y = y.cpu().flatten()

        # Split per graph and merge
        offset = 0
        for idx, node_order in enumerate(batch_node_orders):
            s = len(node_order)
            m = s * (s - 1)
            y_sub = y[offset: offset + m]
            offset += m
            # inverse transform
            y_sub_inv = self.scalers.inverse_transform(y_sub, 'regret')
            edge_idx = 0
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    u = node_order[i]
                    v = node_order[j]
                    val = float(max(y_sub_inv[edge_idx], 0.0))
                    pred_accum[u, v] += val
                    pred_count[u, v] += 1
                    edge_idx += 1

    def get_scaled_features(self, G: nx.DiGraph) -> dgl.DGLGraph:
        if self.load_once and self.template is not None:
            return self._process_graph(G)
        else:
            # lazy path
            template = self._load_template(self.atsp_size)
            weights, regrets, in_solution = self._extract_features_from_graph(G, list(range(self.atsp_size)))
            graph = template
            graph.ndata['weight'] = self.scalers.transform(weights, 'weight').unsqueeze(1).to(self.device)
            graph.ndata['regret'] = self.scalers.transform(regrets, 'regret').unsqueeze(1).to(self.device)
            graph.ndata['in_solution'] = in_solution.unsqueeze(1).to(self.device)
            if hasattr(G, 'graph'):
                graph.graph_attr = dict(G.graph)
            return graph

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        if self.load_once:
            return self.graphs[idx]
        else:
            graph = self._load_instance_graph(self.instances[idx])
            H = self._process_graph(graph)
            import gc
            gc.collect()
            return H, graph
