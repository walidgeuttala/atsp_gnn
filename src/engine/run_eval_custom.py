import os
import torch
import numpy as np
from scipy.stats import pearsonr
import gc 
import pickle

from src.engine import args
from src.engine.run import locate, parse_args, fix_seed
from utils.algorithms import guided_local_search, nearest_neighbor
from utils.atsp_utils import tour_cost, optimal_cost

def eval_model(model_path, args, n_samples=30):
    # Load model and tester
    framework = args.framework.lower()
    tester_module = f'src.engine.test_{framework}'
    models_mod = f'src.models.models_{framework}'
    TesterClass = getattr(locate(tester_module), f'ATSPTester{framework.upper()}', None)
    get_model = getattr(locate(models_mod), f'get_{framework}_model', None)
    assert TesterClass and get_model, "Tester/model not found"
    tester = TesterClass(args)
    model = get_model(args)
    model.eval()
    # Load validation set
    val_dataset = tester.create_test_dataset()
    losses, gaps, corrs = [], [], []
    index = 100
    for idx in range(index, index+30):
        if idx == index+30:
            break
        with torch.no_grad():
            H, G = val_dataset[idx]
            x = H.ndata['weight']
            y_true = H.ndata['regret']
            gc.collect()
            torch.cuda.empty_cache()
            y_pred = model(H, x)  # run once without measuring time

        # MAE loss
        loss = torch.nn.functional.l1_loss(y_pred, y_true, reduction='mean').item()
        regret_pred = val_dataset.scalers.inverse_transform(
            y_pred.cpu().flatten(), 'regret'
        )
        
        # Add predictions to graph
        edge_idx = 0
        for i in range(args.atsp_size):
            for j in range(args.atsp_size):
                if i == j:
                    continue
                
                edge = (i, j)
                if edge in G.edges:
                    G.edges[edge]['regret_pred'] = max(regret_pred[edge_idx], 0.0)
                edge_idx += 1
        opt_cost = optimal_cost(G)
        
        # Initial tour using predicted regrets
        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour)
        init_gap = (init_cost / opt_cost - 1) * 100
        # GNN gap: difference in cost between predicted and optimal tour
        # (Assume tester has a method to compute cost from predicted regret, else skip)
        # Pearson correlation
        corr = pearsonr(y_pred.cpu().numpy().flatten(), y_true.cpu().numpy().flatten())[0]
        losses.append(loss)
        gaps.append(init_gap)
        corrs.append(corr)
    # Aggregate results
    result = {
        'val_loss': np.mean(losses),
        'gnn_gap': np.mean(gaps),
        'pearson_r': np.mean(corrs)
    }
    return result

def main():
    # Model paths
    base_dir = '../jobs/search'
    combos = [
        ('12201357', 'attn'),
        ('12201358', 'concat'),
        ('12201359', 'sum')
    ]
    rel_types = [
        ['pp', 'ss', 'st', 'tt'],
    ]
    results = []
    for folder, agg in combos:
        for rel in rel_types:
            model_path = f'{base_dir}/{folder}/best_model_rel_{"_".join(rel)}_{agg}.pt'
            args = parse_args()
            args.framework = 'dgl'
            args.agg = agg
            args.model_path = model_path
            args.relation_types = rel
            args.atsp_size = 50
            args.data_dir = '../saved_dataset/ATSP_3000x50'
            args.data_path = '../saved_dataset/ATSP_3000x50'
            args.model = 'HetroGAT'
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fix_seed(args.seed)
            res = eval_model(model_path, args, n_samples=30)
            results.append({
                'model': os.path.basename(model_path),
                'relations': rel,
                'agg': agg,
                **res
            })
    # Print results
    for r in results:
        print(f"Model: {r['model']}, Relations: {r['relations']}, Agg: {r['agg']}")
        print(f"  Val Loss: {r['val_loss']:.6f}, GNN Gap: {r['gnn_gap']}, Pearson r: {r['pearson_r']:.4f}")

if __name__ == '__main__':
    main()