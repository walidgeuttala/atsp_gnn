#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import time
import uuid
import os 
import networkx as nx
import numpy as np
import torch
import tqdm.auto as tqdm
import pickle 
import gnngls
from gnngls import algorithms, datasets
from gnngls.model import get_model
from utils import *
from args import *

# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(args_test):
    params = json.load(open(f'{args_test.model_path}/params.json'))
    args = parse_args()
    args = load_params(args, params)
    args.device = 'cuda'
    print(args, flush=True)
    print(args_test, flush=True)
    args.atsp_size = args_test.atsp_size
    args_test.relation_types = args.relation_types
    args_test.half_st = args.half_st
    print_gpu_memory('before loading the dataset')
    test_set = datasets.TSPDataset(f'{args_test.data_path}/test.txt', args)
    print_gpu_memory('after')
    output_path = f'{args_test.model_path}/trial_0/test_atsp{args_test.atsp_size}'
    os.makedirs(output_path, exist_ok=True)
    args.device = torch.device('cuda' if args.device  == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f'model: {args.model} trained in ATSP{args.atsp_size} for {args.n_epochs} and tested in ATSP{args_test.atsp_size} for {len(test_set.instances)}')
    print_gpu_memory('before the model')
    model = get_model(args).to(args.device)
    print_gpu_memory('after the model')
    print('device =', args.device, flush=True)

    checkpoint = torch.load(f'{args_test.model_path}/checkpoint_best_val.pt', map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pbar = tqdm.tqdm(test_set.instances)
    result = {
        'init_gaps': [],
        'final_gaps': [],
        'init_costs': [],
        'final_costs': [],
        'opt_costs': [],
        'gaps': [],
        'avg_corr_cosine': [],
        'avg_corr_normal': [],
        'avg_cnt_search': [],
        'total_model_time': 0,
        'total_gls_time': 0
        }

    for instance in pbar:
        print_gpu_memory('before loading the orignal graph')
        with open(test_set.root_dir / instance, 'rb') as f:
            G = pickle.load(f)
        opt_cost = gnngls.optimal_cost(G, weight='weight')
        print_gpu_memory('after loading the orignal graph')        
        H = test_set.get_scaled_features(G).to(args.device)
        print('after getting the H')
        x = H.ndata['weight']
        result['model_start_time'] = time.time()
        print('before the grad')
        with torch.no_grad():
            y_pred = model(H, x)
        print('after the grad')
        regret_pred = test_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
        idx = 0
        for idx_i in range(args_test.atsp_size):
            for idx_j in range(args_test.atsp_size):
                if idx_i == idx_j:
                    continue
                G.edges[(idx_i, idx_j)]['regret_pred'] = np.maximum(regret_pred[idx].item(), 0)
                idx += 1
        init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')

        result['total_model_time'] += (time.time() - result['model_start_time'])

        result['gls_start_time'] = time.time()
        result['num_nodes'] = len(init_tour) - 1
        init_cost = gnngls.tour_cost(G, init_tour)
        t = time.time()
        best_tour, best_cost, search_progress_i, cnt_ans = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args_test.time_limit, weight='weight',
                                                                                 guides=['regret_pred'],
                                                                                 perturbation_moves=args_test.perturbation_moves,
                                                                                 first_improvement=False)
        
        result['total_gls_time'] += (time.time() - result['gls_start_time'])

        best_cost = gnngls.tour_cost(G, best_tour)
        
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        regret, _ = nx.attr_matrix(G, 'regret')
        regret_pred, _ = nx.attr_matrix(G, 'regret_pred')
        with open(f"{output_path}/instance{instance}.txt", "w") as f:
            # Save array1
            f.write("edge_weight:\n")
            np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array2
            f.write("regret:\n")
            np.savetxt(f, add_diag(result['num_nodes'], H.ndata['regret'].cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array3
            f.write("regret_pred:\n")
            np.savetxt(f, add_diag(result['num_nodes'], y_pred.cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write(f"opt_cost: {opt_cost}\n")
            f.write(f"num_iterations: {cnt_ans}\n")
            f.write(f"init_cost: {init_cost}\n")
            f.write(f"best_cost: {best_cost}\n")
            
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100
        result['init_costs'].append(init_cost)
        result['final_costs'].append(best_cost)
        result['opt_costs'].append(opt_cost)
        result['init_gaps'].append(init_gap)
        result['final_gaps'].append(final_gap)
        # result['avg_corr_normal'].append(correlation_matrix(y_pred.cpu(), H.ndata['regret'].cpu()))
        # result['avg_corr_cosine'].append(cosine_similarity(y_pred.cpu().view(-1), H.ndata['regret'].cpu().view(-1)))
        result['avg_cnt_search'].append(cnt_ans)

        pbar.set_postfix({ 
                'Sum_GNN_time:': '{:.4f}'.format(np.sum(result['total_model_time'])),
                'Sum_Search_time:': '{:.4f}'.format(np.sum(result['total_gls_time'])),
                'Avg_Gap_init:': '{:.4f}'.format(np.mean(result['init_gaps'])),
                'Avg_Gap_best:': '{:.4f}'.format(np.mean(result['final_gaps'])),
                'Avg_Cost_init:': '{:.4f}'.format(np.mean(result['init_costs'])),
                'Avg_Cost_best:': '{:.4f}'.format(np.mean(result['final_costs'])),
                'Avg_Otpimal_best:': '{:.4f}'.format(np.mean(result['opt_costs'])),
                'Avg_counts: ': '{:.4f}'.format(np.mean(result['avg_cnt_search'])),
            })
    # Add the Average of such list key values
    keys = list(result.keys())
    for key in keys:
        if isinstance(result[key], list) and result[key]:  # Check if the value is a non-empty list
            avg_value = sum(result[key]) / len(result[key])  # Calculate the average
            result[f'avg_{key}'] = avg_value  # Add new key with average

    with open(f'{output_path}/results.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)


    print(f"Total time for model prediction: {result['total_model_time']:.4f} seconds")
    print(f"Total time for guided local search: {result['total_gls_time']:.4f} seconds")

if __name__ == '__main__':
    args_test = parse_args_test()

    atsp_sizes = [1000]
    data_path = args_test.data_path
    for atsp_size in atsp_sizes:
        args_test.atsp_size = atsp_size
        args_test.data_path = data_path+str(atsp_size)
        main(args_test)