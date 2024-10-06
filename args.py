import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test model')
    # correct the name of the folder that have the instances insatnces-> instances
    parser.add_argument('--data_dir', type=str, default='../tsp_input/generated_insatnces_3000_size_50/', help='Dataset directory')
    parser.add_argument('--tb_dir', type=str, default='../atsp_model_train_result', help='Tensorboard log directory')

    # ATSP graph parameters
    parser.add_argument('--atsp_size', type=int, default=50, help="Size of the atsp to be solved")

    # Model parameters
    parser.add_argument('--model', type=str, default='HetroGAT', help='set the model name to use')
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden feature dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output feature dimension')
    # In the future activate this so you can choose the type of edges to include in the g2 graph, where g1 -> g2
    # But for now I take it from the dataset and assume it have 5 type edges
    parser.add_argument('--relation_types', type=str, default='ss st ts tt pp', help='Number of relation types')  
    parser.add_argument('--n_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--kj', type=str, default='cat', choices=['cat', 'max'])
    parser.add_argument('--aggregate', type=str, default='sum', choices=['cat', 'sum'])

    # Hyper-parameter about the training/testing
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1234, help='Fix the seed of exprs')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of model trials')
    parser.add_argument('--n_samples_result_train', type=int, default=20, help='Number of samples to print the average gap cost extra in each epoch training')

    # Flag for using GPU
    parser.add_argument('--device', type=str, default='cuda', help="Number of gpu to be used")

    args = parser.parse_args()
    
    return args

def parse_args_test():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--atsp_size', type=int, default=150, help="Size of the atsp to be solved")
    parser.add_argument('--data_path', type=str, default='../tsp_input/generated_insatnces_100_size_150', help='Dataset directory')
    parser.add_argument('--model_path', default='../atsp_model_train_result/Oct06_23-31-25_HetroGAT_trained_ATSP50/trial_0', type=str)
    parser.add_argument('--time_limit', type=float, default=10., help='Time limit for the 2 opt search in seconds') 
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda', help="Number of gpu to be used")

    args = parser.parse_args()

    return args

def load_params(args, params):
    # Update args with values from params
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

