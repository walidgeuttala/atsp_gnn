import time
import gc
import numpy as np
import torch
from utils.algorithms import guided_local_search, nearest_neighbor
from utils.atsp_utils import tour_cost, optimal_cost
from data.dataset_dgl_large import ATSPDatasetDGL
import json
import os


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

        opt_cost = optimal_cost(G)

        start_time = time.time()
        self._predict_regrets_large(model, test_dataset, G)
        model_time = time.time() - start_time

        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
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

        for idx in range(len(test_dataset)):
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
        out_dir = os.path.join(self.args.model_path, f"test_atsp{self.args.atsp_size}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        return results
