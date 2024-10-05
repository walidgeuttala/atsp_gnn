#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import json
import os
import random
import pathlib
import uuid
import numpy as np
import dgl.nn
import torch
import tqdm.auto as tqdm
from torch.utils.data import DataLoader
import itertools

from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import gnngls
from gnngls import models, datasets, algorithms
import torch.nn.functional as F
# Suppress FutureWarnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def loss3(y_pred, y, batch_size = 32):
    num_edges = int(y.shape[0]//batch_size)
    y = y.view(batch_size, num_edges) ** 2
    y = F.normalize(y, p=2, dim=1)
    
    y_pred = y_pred.view(batch_size, num_edges) ** 2
    y_pred = F.normalize(y_pred, p=2, dim=1)
    
    cos_similarities = F.cosine_similarity(y, y_pred, dim=1)

    return 1 - cos_similarities.mean()

def train(model, train_loader, target, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x = batch.ndata['weight']
        y = batch.ndata[target]

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss


def test(model, data_loader, target, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['weight']
            y = batch.ndata[target]

            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        return epoch_loss


def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'val_loss': val_loss
    }, save_path)

def correlation_matrix(tensor1, tensor2):
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(flat_tensor1, flat_tensor2)[0, 1]
    return corr_matrix

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def tsp_to_atsp_instance(G1):
    num_nodes = G1.number_of_nodes() // 2
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])

    first_edge = list(G1.edges)[0]

    # Get the attribute names of the first edge
    attribute_names = G1[first_edge[0]][first_edge[1]].keys()
    attribute_names_list = list(attribute_names)
    for attribute_name in attribute_names_list:
        attribute, _ = nx.attr_matrix(G1, attribute_name)
        attribute = attribute[num_nodes:, :num_nodes]
        for u, v in G2.edges():
            G2[u][v][attribute_name] = attribute[u, v]
    
    return G2

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--embed_dim', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of message passing steps')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads for GAT')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint frequency')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    parser.add_argument('--kj', type=str, default='cat', choices=['cat', 'max'])
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    return args



def run(args):
    # Load dataset
    train_set = datasets.TSPDataset(args.data_dir / 'train.txt')
    val_set = datasets.TSPDataset(args.data_dir / 'val.txt')

    # use GPU if it is available
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print('device =', device)

    _, feat_dim = train_set[0].ndata['weight'].shape
    set_random_seed(1234)
    model = models.EdgePropertyPredictionModel(
        feat_dim,
        args.embed_dim,
        1,
        args.n_layers,
        n_heads=args.n_heads,
        embed_dim2 = args.embed_dim2,
        kj = args.kj
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    if args.target == 'regret':
        criterion = torch.nn.MSELoss()

    elif args.target == 'in_solution':
        # only works for a homogenous dataset
        y = train_set[0].ndata['in_solution']
        pos_weight = len(y) / y.sum() - 1
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                            num_workers=16, pin_memory=True)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}'
    log_dir = args.tb_dir / run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    counter = 0
    
    
    pbar = tqdm.trange(args.n_epochs)
    min_epoch_val_loss = float(1e6)
    for epoch in pbar:
        epoch_loss = train(model, train_loader, args.target, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_val_loss = test(model, val_loader, args.target, criterion, device)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
        average_corr1 = 0
        average_corr2 = 0
        num_samples = 20
        average_gap = 0
        for idx in range(num_samples):
            G = nx.read_gpickle(args.data_dir / val_set.instances[idx])
            H = val_set.get_scaled_features(G).to(device)
            x = H.ndata['weight']
            y = H.ndata['regret']
            with torch.no_grad():
                y_pred = model(H, x)
            
            regret_pred = val_set.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
            es = H.ndata['e'].cpu().numpy()
            for e, regret_pred_i in zip(es, regret_pred):
                G.edges[e]['regret_pred'] = np.maximum(regret_pred_i.item(), 0)
            G = tsp_to_atsp_instance(G)
            opt_cost = gnngls.optimal_cost(G, weight='weight')
            print(opt_cost)
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
            init_cost = gnngls.tour_cost(G, init_tour)
            average_corr1 += correlation_matrix(y_pred.cpu(),H.ndata['regret'].cpu())
            average_corr2 += cosine_similarity(y_pred.cpu().flatten(),H.ndata['regret'].cpu().flatten())
            average_gap += (init_cost / opt_cost - 1) * 100

        min_epoch_val_loss = min(min_epoch_val_loss, epoch_val_loss)

        pbar.set_postfix({
            'Train Loss': '{:.4f}'.format(epoch_loss),
            'Validation Loss': '{:.4f}'.format(epoch_val_loss),
            "correlation : ": '{:.4f}'.format(average_corr1/num_samples),
            "cosin correlation : ": '{:.4f}'.format(average_corr2/num_samples),
            "gap : ": '{:.4f}'.format(average_gap/num_samples),
        })
        
        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / checkpoint_name)

        if best_score is None or epoch_val_loss < best_score - args.min_delta:
            save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_best_val.pt')

            best_score = epoch_val_loss
            counter = 0
        else:
            counter += 1
        
        
        if counter >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    params = dict(vars(args))
    params['data_dir'] = str(params['data_dir'])
    params['tb_dir'] = str(params['tb_dir'])
    json.dump(params, open(args.tb_dir / run_name / 'params.json', 'w'))

    save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')
    return min_epoch_val_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_dir', type=pathlib.Path, help='Where to load dataset')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Where to log Tensorboard data')
    parser.add_argument('--embed_dim', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--embed_dim2', type=int, default=128, help='Maximum hidden feature dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of message passing steps')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads for GAT')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint frequency')
    parser.add_argument('--target', type=str, default='regret', choices=['regret', 'in_solution'])
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    return args
def main():
    search_space = {
        "embed_dim": [256, 128],
        "embd_dim2": [256, 128],
        "n_layers": [4],
        "lr_init": [1e-3],
        "n_heads": [32],
        "kj": ['cat']
    }

    args = parse_args()
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    combinations = list(itertools.product(*values))
    best_loss = float(1e6)
    for combination in combinations:
        param_dict = dict(zip(keys, combination))
        for key, value in param_dict.items():
            setattr(args, key, value)

        val_loss = run(args)
        if best_loss > val_loss:
            best_loss = val_loss
            saved_args = args

    print("best args : ",saved_args, flush=True)

main()