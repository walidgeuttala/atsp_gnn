import argparse
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test model')

    parser.add_argument('data_dir', type=pathlib.Path, help='Dataset directory')
    parser.add_argument('tb_dir', type=pathlib.Path, help='Tensorboard log directory')
    
    # Model parameters
    parser.add_argument('--model', type=str, default="HetroGAT", help='set the model name to use')
    parser.add_argument('--input_dim', type=int, default=128, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden feature dimension')
    parser.add_argument('--output_dim', type=int, default=64, help='Output feature dimension')
    parser.add_argument('--relation_types', type=int, default=5, help='Number of relation types')  
    parser.add_argument('--n_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--kj', type=str, default='cat', choices=['cat', 'max'])

    # Hyper-parameter about the training/testing
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Checkpoint frequency')
    
    # Flag for using GPU
    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()
    
    return args


# add the seed arg
# add the number of trials arg
# 