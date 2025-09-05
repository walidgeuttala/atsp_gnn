import argparse
import yaml
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for both DGL and PyG."""
    parser = argparse.ArgumentParser(description='ATSP GNN Training/Testing')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--atsp_size', type=int, default=10)
    parser.add_argument('--relation_types', nargs='+', default=['ss', 'st', 'tt', 'pp'])
    parser.add_argument('--undirected', action='store_true', help='Use undirected graphs (PyG only)')
    
    # Model/training arguments
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-6)
    
    # Infrastructure
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--tb_dir', type=str, default='./runs')
    
    # Testing specific
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint for testing')
    parser.add_argument('--time_limit', type=float, default=10.0, help='GLS time limit per instance')
    parser.add_argument('--perturbation_moves', nargs='+', default=['2opt'], help='Perturbation moves')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)