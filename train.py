#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import uuid
import itertools
import tqdm.auto as tqdm
import numpy as np
import networkx as nx
import torch
import dgl.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import gnngls
from gnngls import datasets, algorithms
from gnngls.model import get_model
from utils import *
from args import parse_args
# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def epoch_train(model, train_loader, target, criterion, optimizer, device):
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

def epoch_test(model, data_loader, target, criterion, device):
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

def train(args):
    fix_seed(args.seed)
    # Load dataset
    train_set = datasets.TSPDataset(args.data_dir / 'train.txt')
    val_set = datasets.TSPDataset(args.data_dir / 'val.txt')

    # use GPU if it is available
    device = torch.device('cuda:0' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print('device =', device)

    model = get_model(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                            num_workers=16, pin_memory=True)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{args.model}_trained_ATSP{args.tsp_size}'
    log_dir = args.tb_dir / run_name
    writer = SummaryWriter(log_dir)

    # early stopping
    best_score = None
    counter = 0
    
    
    pbar = tqdm.trange(args.n_epochs)
    min_epoch_val_loss = float(1e6)
    for epoch in pbar:
        epoch_loss = epoch_train(model, train_loader, args.target, criterion, optimizer, device)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        epoch_val_loss = epoch_test(model, val_loader, args.target, criterion, device)
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

if __name__ == "__main__":
    args = parse_args()
    train(args)