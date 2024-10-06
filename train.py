#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import os 

import tqdm.auto as tqdm
import torch
import dgl.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gnngls import datasets
from gnngls.model import get_model
from utils import *
from args import parse_args
# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def epoch_train(model, train_loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(train_loader):
        batch = batch.to(device)
        x = batch.ndata['weight']
        y = batch.ndata['regret']

        optimizer.zero_grad()
        y_pred = model(batch, x)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)

    return epoch_loss

def epoch_test(model, data_loader, criterion, device):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(device)
            x = batch.ndata['weight']
            y = batch.ndata['regret']

            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
        
        return epoch_loss

def train(args, trial_id):
    fix_seed(args.seed)
    # Load dataset
    train_set = datasets.TSPDataset(f'{args.data_dir}/train.txt')
    val_set = datasets.TSPDataset(f'{args.data_dir}/val.txt')

    # use GPU if it is available
    args.device = torch.device('cuda' if args.device  == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device =', args.device)

    model = get_model(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                            num_workers=4, pin_memory=True)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{args.model}_trained_ATSP{args.atsp_size}'
    os.makedirs(f'{args.tb_dir}/{run_name}', exist_ok=True)
    log_dir = f'{args.tb_dir}/{run_name}/trial_{str(trial_id)}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    pbar = tqdm.trange(args.n_epochs)
    
    result = dict()
    result['min_val_loss'] = torch.tensor(float(1e6))
    result['train_loss'] = torch.empty((args.n_epochs), dtype=torch.float)
    result['val_loss'] = torch.empty((args.n_epochs), dtype=torch.float)
    # early stopping
    result['best_score'] = None
    result['counter'] = 0

    for epoch in pbar:
        result['train_loss'][epoch] = epoch_train(model, train_loader, criterion, optimizer, args.device)
        result['val_loss'][epoch] = epoch_test(model, val_loader, criterion, args.device)
        
        result2 = atsp_results(model, args, val_set)
        
        for key, value in result2.items():
            writer.add_scalar(key, value, global_step=epoch) 

        result['min_val_loss'] = torch.min(result['min_val_loss'], result['val_loss'][epoch])

        formatted_result = {key: f'{(value/args.n_samples_result_train):.4f}' for key, value in result2.items()}  # Format values to 4 decimal places
        pbar.set_postfix(**formatted_result) 
                
        # Saving the model
        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/{checkpoint_name}')

        if result['best_score'] is None or result['val_loss'][epoch] < result['best_score'] - args.min_delta:
            save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/checkpoint_best_val.pt')
            result['best_score'] = result['val_loss'][epoch]
            result['counter'] = 0
        else:
            result['counter'] += 1
        
        if result['counter'] >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()

    writer.close()

    params = dict(vars(args))
    params['device'] = str(args.device)  # Convert device to a string

    json.dump(params, open(f'{log_dir}/params.json', 'w'))

    save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/checkpoint_final.pt')
    
    return result['min_val_loss']

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.tb_dir, exist_ok=True)
    for trial_id in range(args.n_trials):
        print(f"Best Validation Loss : {train(args, trial_id)}")