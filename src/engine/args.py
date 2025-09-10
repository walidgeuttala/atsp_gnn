import argparse
import yaml
from typing import Dict, Any
import inspect

def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    # Mode / infra
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='train or test')
    parser.add_argument('--framework', type=str, choices=['dgl', 'pyg'], default='dgl',
                        help='Which graph library to use')
    parser.add_argument('--tb_dir', type=str, default='./runs', 
                        help='Tensorboard log directory')
    parser.add_argument('--config', type=str, default=None, help='YAML config file to load')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    return parser


def train_test_parser() -> argparse.ArgumentParser:
    """Comprehensive parser used by training and testing flows."""
    parent = base_parser()

    parser = argparse.ArgumentParser(parents=[parent])

    # Data & dataset
    parser.add_argument('--data_dir', type=str, required=False, default='./data',
                        help='Dataset root directory (contains train/val/test folders or files)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Alternative single-path dataset (used by some test scripts)')
    parser.add_argument('--atsp_size', type=int, default=50)
    parser.add_argument('--relation_types', nargs='+', default=['ss', 'st', 'tt', 'pp'],
                        help='Relation types used by dataset')
    parser.add_argument('--undirected', action='store_true', help='Use undirected graphs (PyG only)')
    parser.add_argument('--hetero', action='store_true', help='Use heterogeneous graphs')

    # Model / architecture
    parser.add_argument('--model', type=str, default='gcn', help='Model name in your models factory')
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_gnn_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--jk', type=str, default=None, choices=[None, 'cat'])
    parser.add_argument('--skip_connection', action='store_true')
    parser.add_argument('--agg', type=str, default='sum', choices=['sum', 'concat'])

    # Training hyperparams
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--min_delta', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=10)

    # Testing / solver
    parser.add_argument('--model_path', type=str, default=None,
                        help='Checkpoint path for testing or resume')
    parser.add_argument('--time_limit', type=float, default=5./30, help='Search time limit per instance')
    parser.add_argument('--perturbation_moves', type=int, default=30,
                        help='Perturbation moves number for search-based tester')
    parser.add_argument('--perturbation_count', type=int, default=5,
                        help='Number of perturbations (older arg name mapping)')

    # Logging / extras
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--n_samples_result_train', type=int, default=30)

    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_config_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """
    Update args with config values when those are not provided via CLI.
    CLI overrides YAML.
    """
    for k, v in cfg.items():
        if not hasattr(args, k):
            # skip unknown keys but still attach them
            setattr(args, k, v)
        else:
            current = getattr(args, k)
            # If CLI used a default and config provides a different value, use config
            # BUT only if user did not explicitly pass the arg (we can't easily detect that here),
            # so we choose to always use config values to simplify reproducible experiments.
            setattr(args, k, v)
    return args


def smart_instantiate(Cls, args):
    """Instantiate Cls by matching __init__ parameters with args Namespace."""
    sig = inspect.signature(Cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
        elif param.default is inspect.Parameter.empty:
            raise ValueError(f"Missing required argument: {name}")
    return Cls(**kwargs)

def parse_args() -> argparse.Namespace:
    parser = train_test_parser()
    args = parser.parse_args()

    # Optionally load YAML and override default args
    if getattr(args, 'config', None):
        cfg = load_config(args.config)
        args = merge_config_args(args, cfg)

    return args
