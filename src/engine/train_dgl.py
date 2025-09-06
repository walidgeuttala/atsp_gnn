import datetime
import json
import os
import time
from typing import Dict, Any, Optional

import torch
import dgl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm.auto as tqdm

from ..data.dataset_dgl import ATSPDatasetDGL
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.models import get_dgl_model  # Your model factory
from src.utils import fix_seed


class ATSPTrainerDGL:
    """Training manager for DGL-based ATSP models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.setup_directories()
    
    def setup_directories(self):
        """Setup logging and checkpoint directories."""
        timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.run_name = f'{timestamp}_{self.args.model}_ATSP{self.args.atsp_size}'
        self.log_dir = f'{self.args.tb_dir}/{self.run_name}'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_datasets(self):
        """Create train/val datasets."""
        self.train_dataset = ATSPDatasetDGL(
            dataset_dir=self.args.data_dir,
            split='train',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            device=self.device
        )
        
        self.val_dataset = ATSPDatasetDGL(
            dataset_dir=self.args.data_dir,
            split='val', 
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            device=self.device
        )
    
    def create_dataloaders(self):
        """Create data loaders."""
        self.train_loader = self.train_dataset.get_dataloader(
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = self.val_dataset.get_dataloader(
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def epoch_train(self, model, criterion, optimizer) -> float:
        """Training epoch."""
        model.train()
        epoch_loss = 0
        
        for batch_i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            x = batch.ndata['weight']
            y = batch.ndata['regret']
            
            optimizer.zero_grad()
            y_pred = model(batch, x)
            loss = criterion(y_pred, y.type_as(y_pred))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.detach().item()
        
        return epoch_loss / (batch_i + 1)
    
    def epoch_test(self, model, criterion) -> float:
        """Validation epoch."""
        model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                x = batch.ndata['weight']
                y = batch.ndata['regret']
                
                y_pred = model(batch, x)
                loss = criterion(y_pred, y.type_as(y_pred))
                epoch_loss += loss.item()
        
        return epoch_loss / (batch_i + 1)
    
    def train(self, model, trial_id: int = 0) -> Dict[str, Any]:
        """Main training loop."""
        self.create_datasets()
        self.create_dataloaders()
        
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.args.lr_decay)
        criterion = torch.nn.MSELoss()
        
        trial_dir = f'{self.log_dir}/trial_{trial_id}'
        os.makedirs(trial_dir, exist_ok=True)
        
        # Results tracking
        results = {
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'patience_counter': 0
        }
        
        pbar = tqdm.trange(self.args.n_epochs)
        
        for epoch in pbar:
            # Training
            train_loss = self.epoch_train(model, criterion, optimizer)
            val_loss = self.epoch_test(model, criterion)
            
            results['train_losses'].append(train_loss)
            results['val_losses'].append(val_loss)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            
            # Save best model
            if val_loss < results['best_val_loss'] - self.args.min_delta:
                results['best_val_loss'] = val_loss
                results['best_epoch'] = epoch
                results['patience_counter'] = 0
                self.save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                                   f'{trial_dir}/best_model.pt')
            else:
                results['patience_counter'] += 1
            
            # Early stopping
            if results['patience_counter'] >= self.args.patience:
                break
            
            scheduler.step()
        
        # Save final results
        with open(f'{trial_dir}/results.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'patience_counter'}, f, indent=2)
        
        return results
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, path):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(self.args)
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

        # also save a "latest" copy for easy resume
        latest_path = os.path.join(os.path.dirname(path), 'latest.pt')
        torch.save(state, latest_path)
