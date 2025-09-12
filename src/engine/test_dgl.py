import json
import time
import os
from typing import Dict, Any, Callable
import numpy as np
import torch
import pickle
import tqdm
import gc

from data.dataset_dgl import ATSPDatasetDGL
from utils.algorithms import guided_local_search, nearest_neighbor
from utils.atsp_utils import tour_cost, optimal_cost

class ATSPTesterDGL:
    """Testing manager for DGL-based ATSP models with Guided Local Search."""
    
    def __init__(self, args, get_model_fn: Callable = None):
        self.args = args
        self.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    def create_test_dataset(self):
        """Create test dataset with training parameters."""
        return ATSPDatasetDGL(
            data_dir=self.args.data_path,
            split='test',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            undirected=self.args.undirected,
            device=self.device,
            load_once=False
        )
    
    def test_instance(self, model, test_dataset, instance_idx: int) -> Dict[str, Any]:
        """Test single instance with GLS."""
        # Load original NetworkX graph
        H, G = test_dataset[instance_idx]
        # Get optimal cost
        opt_cost = optimal_cost(G)
        
        # Get scaled features and predict regrets
        x = H.ndata['weight']
        start_time = time.time()
        with torch.no_grad():
            y_pred = model(H, x)
        model_time = time.time() - start_time
        # Inverse transform predictions
        regret_pred = test_dataset.scalers.inverse_transform(
            y_pred.cpu().numpy().flatten(), 'regret'
        )
        
        # Add predictions to graph
        edge_idx = 0
        for i in range(self.args.atsp_size):
            for j in range(self.args.atsp_size):
                if i == j:
                    continue
                
                edge = (i, j)
                if edge in G.edges:
                    G.edges[edge]['regret_pred'] = max(regret_pred[edge_idx], 0.0)
                edge_idx += 1
        
        # Initial tour using predicted regrets
        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour)
        # Guided Local Search
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
        
        # Calculate gaps
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (final_cost / opt_cost - 1) * 100
        result = {
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
    
        # Free GPU memory
        del H, x, y_pred, regret_pred
        gc.collect()
        torch.cuda.empty_cache()
        return result
        
    
    def test_all(self, model, test_dataset) -> Dict[str, Any]:
        """Test all instances in dataset."""
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
        
        pbar = tqdm.tqdm(range(len(test_dataset)), desc="Testing instances")
        
        for idx in pbar:
            instance_result = self.test_instance(model, test_dataset, idx)
            results['instance_results'].append(instance_result)
            
            # Aggregate results
            for key in ['opt_costs', 'init_costs', 'final_costs', 'init_gaps', 
                       'final_gaps', 'num_iterations', 'model_times', 'gls_times']:
                field = key[:-1] if key.endswith('s') else key  # Remove 's' for field name
                results[key].append(instance_result[field])
            
            # Update progress bar
            pbar.set_postfix({
                'avg_init_gap': f"{np.mean(results['init_gaps']):.2f}%",
                'avg_final_gap': f"{np.mean(results['final_gaps']):.2f}%",
                'avg_iterations': f"{np.mean(results['num_iterations']):.1f}"
            })
        
        # Calculate aggregated statistics
        for key in ['opt_costs', 'init_costs', 'final_costs', 'init_gaps', 
                   'final_gaps', 'num_iterations', 'model_times', 'gls_times']:
            values = results[key]
            results[f'avg_{key}'] = np.mean(values)
            results[f'std_{key}'] = np.std(values)
            results[f'total_{key}'] = np.sum(values) if 'time' in key else None
        
        return results
    
    def run_test(self, model) -> Dict[str, Any]:
        """Main testing pipeline."""
        # Load model and parameters
        
        # Create test dataset
        test_dataset = self.create_test_dataset()
        
        print(f"Testing model on {len(test_dataset)} instances of size {self.args.atsp_size}")
        print(f"Using relation types: {self.args.relation_types}")
        
        # ---- Warm-up ----
        print("Running warm-up forward pass...")
        model.eval()
        with torch.no_grad():
            dummy_idx = 0
            G = pickle.load(open(test_dataset.data_dir / test_dataset.instances[dummy_idx], 'rb'))
            H = test_dataset.get_scaled_features(G)
            x = H.ndata['weight']
            gc.collect()
            torch.cuda.empty_cache()
            _ = model(H, x)  # run once without measuring time
            del H, x, G
            gc.collect()
            torch.cuda.empty_cache()
        # Run tests
        results = self.test_all(model, test_dataset)
        
        # Save results
        output_dir = os.path.join(
            os.path.dirname(self.args.model_path),
            f'test_atsp{self.args.atsp_size}'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f'results_test_{self.args.atsp_size}.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results