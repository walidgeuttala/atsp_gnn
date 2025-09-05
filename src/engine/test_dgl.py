import json
import time
import os
from typing import Dict, Any
import numpy as np
import torch
import pickle
import tqdm

from ..data.datasets_dgl import ATSPDatasetDGL
from ..algorithms import guided_local_search, nearest_neighbor
from ..utils import tour_cost, optimal_cost


class ATSPTesterDGL:
    """Testing manager for DGL-based ATSP models with Guided Local Search."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_class, checkpoint_path: str):
        """Load trained model from checkpoint."""
        # Load training parameters
        params_path = os.path.dirname(checkpoint_path) + '/params.json'
        with open(params_path, 'r') as f:
            train_params = json.load(f)
        
        # Create model with training parameters
        model = model_class(train_params).to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, train_params
    
    def create_test_dataset(self, train_params: Dict[str, Any]):
        """Create test dataset with training parameters."""
        return ATSPDatasetDGL(
            dataset_dir=self.args.data_path,
            split='test',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(train_params.get('relation_types', ['ss', 'st', 'tt', 'pp'])),
            device=self.device
        )
    
    def test_instance(self, model, test_dataset, instance_idx: int) -> Dict[str, Any]:
        """Test single instance with GLS."""
        # Load original NetworkX graph
        instance_path = test_dataset.dataset_dir / test_dataset.instances[instance_idx]
        with open(instance_path, 'rb') as f:
            G = pickle.load(f)
        
        # Get optimal cost
        opt_cost = optimal_cost(G, weight='weight')
        
        # Get scaled features and predict regrets
        start_time = time.time()
        H = test_dataset.get_scaled_features(G).to(self.device)
        x = H.ndata['weight']
        
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
        
        best_tour, best_cost, search_progress, num_iterations = guided_local_search(
            G, init_tour, init_cost, time_limit,
            weight='weight',
            guides=['regret_pred'],
            perturbation_moves=self.args.perturbation_moves,
            first_improvement=False
        )
        
        gls_time = time.time() - gls_start
        
        # Calculate gaps
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100
        
        return {
            'opt_cost': opt_cost,
            'init_cost': init_cost,
            'best_cost': best_cost,
            'init_gap': init_gap,
            'final_gap': final_gap,
            'num_iterations': num_iterations,
            'model_time': model_time,
            'gls_time': gls_time,
            'instance': test_dataset.instances[instance_idx]
        }
    
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
    
    def run_test(self, model_class, checkpoint_path: str) -> Dict[str, Any]:
        """Main testing pipeline."""
        # Load model and parameters
        model, train_params = self.load_model(model_class, checkpoint_path)
        
        # Create test dataset
        test_dataset = self.create_test_dataset(train_params)
        
        print(f"Testing model on {len(test_dataset)} instances of size {self.args.atsp_size}")
        print(f"Using relation types: {train_params.get('relation_types', 'default')}")
        
        # Run tests
        results = self.test_all(model, test_dataset)
        
        # Save results
        output_dir = os.path.join(
            os.path.dirname(checkpoint_path),
            f'test_atsp{self.args.atsp_size}'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
