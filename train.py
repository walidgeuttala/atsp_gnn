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

def train(args, trial_id, run_name=None):
    args.device = 'cuda'
    print(args)
    fix_seed(args.seed)
    # Load dataset
    train_set = datasets.TSPDataset(f'{args.data_dir}/train.txt', args)
    val_set = datasets.TSPDataset(f'{args.data_dir}/val.txt', args)

    # use GPU if it is available
    args.device = torch.device('cuda' if args.device  == 'cuda' and torch.cuda.is_available() else 'cpu')
    print('device =', args.device)

    model = get_model(args).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=dgl.batch,
                            num_workers=8, pin_memory=True)

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    if run_name == None:
        run_name = f'{timestamp}_{args.model}_trained_ATSP{args.atsp_size}'
    os.makedirs(f'{args.tb_dir}/{run_name}', exist_ok=True)
    log_dir = f'{args.tb_dir}/{run_name}/trial_{str(trial_id)}'
    os.makedirs(log_dir, exist_ok=True)
    
    output_file_path = f'{log_dir}/train_logs.txt'
    max_line_length = 50

    with open(output_file_path, 'a') as file:
        file.write('args = { \n')

        for key, value in vars(args).items():
            line = f'{key}: {value}'
            while len(line) > max_line_length:
                file.write(line[:max_line_length] + '\n')
                line = line[max_line_length:]
            file.write(line + '\n')
        file.write('} \n\n')
    pbar = tqdm.trange(args.n_epochs)
    
    result = dict()
    result['train_loss'] = torch.empty((args.n_epochs), dtype=torch.float)
    result['val_loss'] = torch.empty((args.n_epochs), dtype=torch.float)
    # early stopping
    result['min_val_loss'] = None
    result['min_avg_gap'] = None
    result['counter'] = 0
    ordered_keys = ['epoch', 'train_loss', 'val_loss', 'avg_gap', 'avg_init_cost', 'avg_opt_cost', 'avg_corr', 'avg_corr_cosin']

    for epoch in pbar:
        result['train_loss'][epoch] = epoch_train(model, train_loader, criterion, optimizer, args.device)
        result['val_loss'][epoch] = epoch_test(model, val_loader, criterion, args.device)
        result2 = atsp_results(model, args, val_set)
        
        formatted_result = {key: f'{(value/args.n_samples_result_train):.4f}' for key, value in result2.items()}  # Format values to 4 decimal places
        formatted_result['train_loss'] = f"{result['train_loss'][epoch]:.4f}"
        formatted_result['val_loss'] = f"{result['val_loss'][epoch]:.4f}"
        formatted_result['epoch'] = f'{epoch:.4f}'
        pbar.set_postfix(**formatted_result) 
        # Create the formatted string in the specified order
        formatted_result_str = ' '.join([f'{key}: {formatted_result[key]}' for key in ordered_keys if key in formatted_result])
        with open(output_file_path, 'a') as file:
            file.write(formatted_result_str + '\n')
        # Saving the model
        if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
            checkpoint_name = f'checkpoint_{epoch}.pt'
            save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/{checkpoint_name}')

        if result['min_avg_gap'] is None or result2['avg_gap'] < result['min_avg_gap']:
            save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/checkpoint_best_avg_gap.pt')
            result['min_avg_gap'] = result2['avg_gap']
            best_avg_gap_result = formatted_result.copy()
        if result['min_val_loss'] is None or result['val_loss'][epoch] < result['min_val_loss'] - args.min_delta:
            save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/checkpoint_best_val.pt')
            result['min_val_loss'] = result['val_loss'][epoch]
            result['counter'] = 0
            best_val_result = formatted_result.copy()
        else:
            result['counter'] += 1

        
        if result['counter'] >= args.patience:
            pbar.close()
            break

        lr_scheduler.step()
    
    params = dict(vars(args))
    params['device'] = str(args.device)  # Convert device to a string

    json.dump(params, open(f'{log_dir}/params.json', 'w'))

    save(model, optimizer, epoch, result['train_loss'][epoch], result['val_loss'][epoch], f'{log_dir}/checkpoint_final.pt')
    
    return best_val_result, best_avg_gap_result, run_name

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.tb_dir, exist_ok=True)
    run_name = None
    list_best_val_result = []
    list_best_avg_gap_result = []
    for trial_id in range(args.n_trials):
        best_val_result, best_avg_gap_result, run_name = train(args, trial_id, run_name)
        print('Best validation result')
        for key, value in best_val_result.items():
            print(f'{key} {value}', end=' ')  
        print('\n') 
        print('Best average gap result')
        for key, value in best_avg_gap_result.items():
            print(f'{key} {value}', end=' ')  
        print('\n') 
        list_best_val_result.append(best_val_result)
        list_best_avg_gap_result.append(best_avg_gap_result)
        args.seed += 1

    if args.n_trials > 1:
        keys = list_best_val_result[0].keys()

        tensor_best_val_result = {key: torch.tensor([float(d[key]) for d in list_best_val_result], dtype=torch.float32) for key in keys}
        tensor_best_avg_gap_result = {key: torch.tensor([float(d[key]) for d in list_best_avg_gap_result], dtype=torch.float32) for key in keys}
        stats_best_val_result = calculate_statistics(tensor_best_val_result)
        stats_best_avg_gap_result = calculate_statistics(tensor_best_avg_gap_result)
        log_dir = f'{args.tb_dir}/{run_name}'
        torch.save(tensor_best_val_result, f'{log_dir}/tensor_best_val_result.pt')
        torch.save(tensor_best_avg_gap_result, f'{log_dir}/tensor_best_avg_gap_result.pt')
        print(stats_best_val_result)
        print(stats_best_avg_gap_result)