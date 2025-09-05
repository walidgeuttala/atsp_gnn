import datetime
import json
import os
import torch
from typing import Dict, Any

def setup_directories(args, prefix: str = "") -> str:
    """Setup logging and checkpoint directories."""
    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{prefix}_{args.model}_ATSP{args.atsp_size}'
    log_dir = f'{args.tb_dir}/{run_name}'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path, args=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    if args:
        checkpoint['args'] = vars(args)
    
    torch.save(checkpoint, path)

def load_checkpoint(model, checkpoint_path: str, device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def save_results(results: Dict[str, Any], path: str):
    """Save training/testing results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
