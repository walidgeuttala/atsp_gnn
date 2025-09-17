import time
import gc
import os
from pathlib import Path
import numpy as np
import torch
import networkx as nx
import tsplib95
import lkh
from utils.algorithms import guided_local_search, nearest_neighbor
from utils.atsp_utils import tour_cost, optimal_cost, get_adj_matrix_string, is_valid_tour
from data.dataset_dgl_large import ATSPDatasetDGL
import json


class ATSPTesterDGLLarge:
    """Testing manager for large ATSP instances with subgraph covering and batch prediction."""

    def __init__(self, args, get_model_fn=None):
        self.args = args
        self.full_graph_path = args.data_path
        self.template_path = args.template_path  # new
        self.sub_size = args.sub_size
        self.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    

    def create_test_dataset(self):
        return ATSPDatasetDGL(
            data_dir=self.args.data_path,
            split='test',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            undirected=self.args.undirected,
            device=self.device,
            load_once=False,
            template_dir=self.args.template_path,   # key fix
        )

    def _predict_regrets_large(self, model, test_dataset, G):
        """Use subgraph covering if sub_size < full size, else full graph."""
        sub_size = getattr(self.args, 'sub_size', None)
        if sub_size is not None and sub_size < self.args.atsp_size:
            # Cover the large graph with subgraphs and predict regrets
            test_dataset.cover_and_predict_full_graph(
                model, G, sub_size=sub_size,
                batch_size=getattr(self.args, 'batch_size', 16)
            )
        else:
            # Fall back to full-graph prediction
            H = test_dataset.get_scaled_features(G)
            x = H.ndata['weight']
            with torch.no_grad():
                y_pred = model(H, x)
            regret_pred = test_dataset.scalers.inverse_transform(
                y_pred.cpu().flatten(), 'regret'
            )
            edge_idx = 0
            for i in range(self.args.atsp_size):
                for j in range(self.args.atsp_size):
                    if i == j:
                        continue
                    if (i, j) in G.edges:
                        G.edges[(i, j)]['regret_pred'] = max(regret_pred[edge_idx], 0.0)
                    edge_idx += 1

    def test_instance(self, model, test_dataset, instance_idx: int):
        # load only the NetworkX graph, no DGL template
        G = test_dataset._load_instance_graph(test_dataset.instances[instance_idx])
        n = self.args.atsp_size

        # Try to compute a trustworthy reference optimal cost
        opt_cost = self._reference_opt_cost(G)

        start_time = time.time()
        self._predict_regrets_large(model, test_dataset, G)
        model_time = time.time() - start_time

        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        # Basic validity assertion
        if not is_valid_tour(G, init_tour):
            raise RuntimeError(f"Init tour invalid (len={len(init_tour)}) for instance {test_dataset.instances[instance_idx]}")
        init_cost = tour_cost(G, init_tour)

        gls_start = time.time()
        time_limit = gls_start + self.args.time_limit
        best_tour, final_cost, search_progress, num_iterations = guided_local_search(
            G, init_tour, init_cost, time_limit,
            weight='weight',
            guides=['regret_pred'],
            perturbation_moves=self.args.perturbation_moves,
            first_improvement=False
        )
        gls_time = time.time() - gls_start

        # Optional deeper refinement with LKH
        if getattr(self.args, 'lkh_runs', 0) > 0 or getattr(self.args, 'lkh_time_limit', 0.0) > 0.0:
            lkh_best_tour, lkh_best_cost = self._solve_with_lkh(G, time_limit=getattr(self.args, 'lkh_time_limit', 0.0), runs=getattr(self.args, 'lkh_runs', 0))
            if lkh_best_cost < final_cost:
                best_tour, final_cost = lkh_best_tour, lkh_best_cost

        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (final_cost / opt_cost - 1) * 100

        gc.collect()
        torch.cuda.empty_cache()

        return {
            'opt_cost': opt_cost,
            'init_cost': init_cost,
            'final_cost': final_cost,
            'init_gap': init_gap,
            'final_gap': final_gap,
            'num_iteration': num_iterations,
            'model_time': model_time,
            'gls_time': gls_time,
            'instance': test_dataset.instances[instance_idx]
        }

    def _reference_opt_cost(self, G) -> float:
        """Return a reliable optimal cost for the given graph under 'weight'.

        Strategy:
        1) If edges marked with in_solution define a valid Hamiltonian cycle, use that cost.
        2) Otherwise or if the computed gap would be negative later, call LKH to solve ATSP.
        """
        # 1) Attempt to reconstruct a tour from in_solution flags
        succ = {}
        pred = {}
        for u, v, data in G.edges(data=True):
            if data.get('in_solution', False):
                # tolerate multiple flags; keep first seen
                if u not in succ:
                    succ[u] = v
                if v not in pred:
                    pred[v] = u
        n = G.number_of_nodes()
        if len(succ) == n and len(pred) == n and all(u in pred for u in G.nodes) and all(v in succ for v in G.nodes):
            # try to walk the cycle from 0
            tour = [0]
            visited = set([0])
            cur = 0
            ok = True
            for _ in range(n):
                nxt = succ.get(cur)
                if nxt is None or nxt in visited:
                    ok = False
                    break
                tour.append(nxt)
                visited.add(nxt)
                cur = nxt
            if ok and cur == 0:
                # already closed; ensure closure
                if tour[-1] != 0:
                    tour.append(0)
                return float(tour_cost(G, tour, weight='weight'))

        # 2) Fall back to LKH to solve ATSP under 'weight'
        try:
            tsplib_str = get_adj_matrix_string(G, weight='weight')
            problem = tsplib95.loaders.parse(tsplib_str)
            # default bundled LKH path
            root = Path(__file__).resolve().parents[2]
            lkh_path = str(root / 'LKH-3.0.9' / 'LKH')
            solution = lkh.solve(lkh_path, problem=problem)
            tour_nodes = [v - 1 for v in solution[0]]  # to 0-index
            tour_nodes.append(tour_nodes[0])
            return float(tour_cost(G, tour_nodes, weight='weight'))
        except Exception:
            # As a last resort, approximate with nearest neighbor cost
            approx_tour = nearest_neighbor(G, start=0, weight='weight')
            return float(tour_cost(G, approx_tour, weight='weight'))

    def _solve_with_lkh(self, G, time_limit: float = 0.0, runs: int = 0):
        """Call LKH on the current graph (weight) and return (tour, cost).

        Deeper search: set larger 'runs' and/or 'time_limit'. Returns best found tour.
        """
        tsplib_str = get_adj_matrix_string(G)
        problem = tsplib95.loaders.parse(tsplib_str)
        root = Path(__file__).resolve().parents[2]
        lkh_path = str(root / 'LKH-3.0.9' / 'LKH')
        params = {}
        if runs and runs > 0:
            params['RUNS'] = runs
        if time_limit and time_limit > 0:
            params['TIME_LIMIT'] = time_limit
        try:
            if params:
                solution = lkh.solve(lkh_path, problem=problem, parameters=params)
            else:
                solution = lkh.solve(lkh_path, problem=problem)
            tour_nodes = [v - 1 for v in solution[0]]
            tour_nodes.append(tour_nodes[0])
            if not is_valid_tour(G, tour_nodes):
                raise RuntimeError("LKH returned invalid tour")
            return tour_nodes, float(tour_cost(G, tour_nodes, weight='weight'))
        except Exception as e:
            # Fall back: return existing NN or GLS
            nn_tour = nearest_neighbor(G, start=0, weight='weight')
            return nn_tour, float(tour_cost(G, nn_tour, weight='weight'))


    def test_all(self, model, test_dataset):
        results = {
            'instance_results': [],
            'opt_costs': [],
            'init_costs': [],
            'final_costs': [],
            'init_gaps': [],
            'final_gaps': [],
            'num_iterations': [],
            'model_times': [],
            'gls_times': []
        }

        total = len(test_dataset)
        limit = getattr(self.args, 'limit_instances', None)
        if isinstance(limit, int) and limit > 0:
            total = min(total, limit)
        for idx in range(total):
            instance_result = self.test_instance(model, test_dataset, idx)
            results['instance_results'].append(instance_result)

            for key in ['opt_costs', 'init_costs', 'final_costs',
                        'init_gaps', 'final_gaps', 'num_iterations',
                        'model_times', 'gls_times']:
                field = key[:-1] if key.endswith('s') else key
                results[key].append(instance_result[field])

        # Aggregate statistics
        for key in ['opt_costs', 'init_costs', 'final_costs',
                    'init_gaps', 'final_gaps', 'num_iterations',
                    'model_times', 'gls_times']:
            values = results[key]
            results[f'avg_{key}'] = np.mean(values)
            results[f'std_{key}'] = np.std(values)
            results[f'total_{key}'] = np.sum(values) if 'time' in key else None

        return results
    
    def run_test(self, model):
        test_dataset = self.create_test_dataset()
        results = self.test_all(model, test_dataset)
        # save next to model_path for consistency with run_large print
        # Save next to the checkpoint directory (model_path may be a file)
        base_dir = self.args.model_path
        if isinstance(base_dir, str) and base_dir.endswith('.pt'):
            base_dir = os.path.dirname(base_dir)
        out_dir = os.path.join(base_dir, f"test_atsp{self.args.atsp_size}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        return results
