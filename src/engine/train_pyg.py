import datetime
import json
import os
from typing import Dict, Any

import torch
from torch_geometric.loader import DataLoader
import tqdm.auto as tqdm

from ..data.dataset_pyg import ATSPDatasetPyG


class ATSPTrainerPyG:
    """Training manager for PyG-based ATSP models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.setup_directories()
    
    def setup_directories(self):
        """Setup logging directories."""
        timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.run_name = f'{timestamp}_{self.args.model}_ATSP{self.args.atsp_size}'
        self.log_dir = f'{self.args.tb_dir}/{self.run_name}'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_datasets(self):
        """Create train/val datasets.""" 
        self.train_dataset = ATSPDatasetPyG(
            dataset_dir=self.args.data_dir,
            split='train',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            device=self.device,
            undirected=getattr(self.args, 'undirected', False)
        )
        
        self.val_dataset = ATSPDatasetPyG(
            dataset_dir=self.args.data_dir,
            split='val',
            atsp_size=self.args.atsp_size, 
            relation_types=tuple(self.args.relation_types),
            device=self.device,
            undirected=getattr(self.args, 'undirected', False)
        )
    
    def create_dataloaders(self):
        """Create PyG data loaders."""
        self.train_loader = self.train_dataset.get_dataloader(
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        self.val_loader = self.val_dataset.get_dataloader(
            batch_size=self.args.batch_size,
            shuffle=False
        )
    
    def epoch_train(self, model, criterion, optimizer) -> float:
        """Training epoch for PyG."""
        model.train()
        epoch_loss = 0
        
        for batch_i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            y_pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(y_pred, batch.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.detach().item()
        
        return epoch_loss / (batch_i + 1)
    
    def epoch_test(self, model, criterion) -> float:
        """Validation epoch for PyG."""
        model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                y_pred = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(y_pred, batch.y)
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
            train_loss = self.epoch_train(model, criterion, optimizer)
            val_loss = self.epoch_test(model, criterion)
            
            results['train_losses'].append(train_loss)
            results['val_losses'].append(val_loss)
            
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
        
        # Save results
        with open(f'{trial_dir}/results.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'patience_counter'}, f, indent=2)
        
        return results
    
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(self.args)
        }, path)
