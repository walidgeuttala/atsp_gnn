#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import pathlib
import time
import uuid
import os 
import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

import gnngls
from gnngls import algorithms, models, datasets
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('output_path', type=pathlib.Path)
    parser.add_argument('--time_limit', type=float, default=10.) # time limit is in the seconds
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    params = json.load(open(args.model_path.parent / 'params.json'))

    test_set = datasets.TSPDataset(args.data_path)

    os.makedirs(args.output_path, exist_ok=True)
    if 'regret_pred' in args.guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        print('device =', device)

        feat_dim = 1
        model = models.RGCN4(
            feat_dim,
            params['embed_dim'],
            1,
            test_set.etypes,
            params['n_layers'],
            params['n_heads']
        ).to(device)

        checkpoint = torch.load(args.model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.tqdm(test_set.instances)
    init_gaps = []
    final_gaps = []
    init_costs = []
    final_costs = []
    opt_costs = []
    search_progress = []
    cnt = 0
    gaps = []
    search_progress = []
    avg_corr_cosine=[]
    avg_corr_normal=[]
    avg_cnt_ans = []
    cnt = 0
    corr_all = 0.
    total_model_time = 0
    total_gls_time = 0
    for instance in pbar:
        G = nx.read_gpickle(test_set.root_dir / instance)
        opt_cost = gnngls.optimal_cost(G, weight='weight')

        t = time.time()
        search_progress.append({
            'instance': instance,
            'time': t,
            'opt_cost': opt_cost
        })

        model_start_time = time.time()
        if 'regret_pred' in args.guides:
            H = test_set.get_test_scaled_features_not_samesize_graphs(G).to(device)
            x = H.ndata['weight']
            with torch.no_grad():
                y_pred = model(H, x)
            regret_pred = test_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
            es = H.ndata['e'].cpu().numpy()
            for e, regret_pred_i in zip(es, regret_pred):
                G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
        else:
            init_tour = algorithms.nearest_neighbor(G, 0, weight='weight')

        model_time = time.time() - model_start_time
        total_model_time += model_time

        gls_start_time = time.time()
        num_nodes = len(init_tour) - 1
        init_cost = gnngls.tour_cost(G, init_tour)
        
        best_tour, best_cost, search_progress_i, cnt_ans = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args.time_limit, weight='weight',
                                                                                 guides=args.guides,
                                                                                 perturbation_moves=args.perturbation_moves,
                                                                                 first_improvement=False)
        
        gls_time = time.time() - gls_start_time

        best_cost2 = gnngls.tour_cost(G, best_tour)
        # for row in search_progress_i:
        #     row.update({
        #         'instance': instance,
        #         'opt_cost': opt_cost
        #     })
        
        #     search_progress.append(row)
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        regret, _ = nx.attr_matrix(G, 'regret')
        regret_pred, _ = nx.attr_matrix(G, 'regret_pred')
        print(f"initial cost {len(init_tour)} best_tour {len(best_tour)}, optimal_tour {opt_cost}",flush=True)
        print(f"initial cost {init_cost} best_cost {best_cost}, optimal_cost {opt_cost}",flush=True)
        with open(args.output_path / f"instance{cnt}.txt", "w") as f:
            # Save array1
            f.write("edge_weight:\n")
            np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array2
            f.write("regret:\n")
            np.savetxt(f, add_diag(num_nodes, H.ndata['regret'].cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array3
            f.write("regret_pred:\n")
            np.savetxt(f, add_diag(num_nodes, y_pred.cpu()).numpy(), fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write(f"opt_cost: {opt_cost}\n")
            f.write(f"num_iterations: {cnt_ans}\n")
            f.write(f"init_cost: {init_cost}\n")
            f.write(f"best_cost: {best_cost}\n")
            
        
        cnt += 1
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100
        init_costs.append(init_cost)
        final_costs.append(best_cost)
        opt_costs.append(opt_cost)
        init_gaps.append(init_gap)
        final_gaps.append(final_gap)
        avg_corr_normal.append(correlation_matrix(y_pred.cpu(), H.ndata['regret'].cpu()))
        avg_corr_cosine.append(cosine_similarity(y_pred.cpu().view(-1), H.ndata['regret'].cpu().view(-1)))
        avg_cnt_ans.append(cnt_ans)

        pbar.set_postfix({ 
                'Avg Gap init:': '{:.4f}'.format(np.mean(init_gaps)),
                'Avg Gap best:': '{:.4f}'.format(np.mean(final_gaps)),
                'Avg Cost init:': '{:.4f}'.format(np.mean(init_costs)),
                'Avg Cost best:': '{:.4f}'.format(np.mean(final_costs)),
                'Avg Otpimal best:': '{:.4f}'.format(np.mean(opt_costs)),
                #'Avg corr normal ': '{:.4f}'.format(np.mean(avg_corr_normal)*100),
                #'Avg cosine normal ': '{:.4f}'.format(np.mean(avg_corr_cosine)*100),
                #'Avg counts ': '{:.4f}'.format(np.mean(avg_cnt_ans)),
            })
        

    search_progress_df = pd.DataFrame.from_records(search_progress)
    search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
        'min')

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    if not args.run_dir.exists():
        args.run_dir.mkdir()
    search_progress_df.to_pickle(args.run_dir / run_name)

    print(f"Total time for model prediction: {total_model_time:.4f} seconds")
    print(f"Total time for guided local search: {total_gls_time:.4f} seconds")







    # gap = (best_cost / opt_cost - 1) * 100
    #     gaps.append(gap)
    #     print(f'best_cost {best_cost} opt_cost {opt_cost}', flush=True)
    #     print('Avg Gap: {:.4f}'.format(np.mean(gaps)), flush=True)

    #     print(corr_all/cnt)
    #     print(cnt_ans)

    #     edge_weight, _ = nx.attr_matrix(G, 'weight')
    #     corr = correlation_matrix(y_pred.cpu(),H.ndata['regret'].cpu())
    #     corr_all += corr

    #     with open(args.output_path / f"instance{cnt}.txt", "w") as f:
    #         # Save array1
    #         f.write("edge_weight:\n")
    #         np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
    #         f.write("\n")

    #         # Save array2
    #         f.write("regret:\n")
    #         np.savetxt(f, add_diag(H.ndata['regret'].cpu()).numpy(), fmt="%.8f", delimiter=" ")
    #         f.write("\n")

    #         # Save array3
    #         f.write("regret_pred:\n")
    #         np.savetxt(f, add_diag(y_pred.cpu()).numpy(), fmt="%.8f", delimiter=" ")
    #         f.write("\n")

    #         f.write(f"opt_cost: {opt_cost}\n")
    #         f.write(f"num_iterations: {cnt_ans}\n")
    #         f.write(f"init_cost: {init_cost}\n")
    #         f.write(f"best_cost: {best_cost}\n")
            

    #     cnt += 1