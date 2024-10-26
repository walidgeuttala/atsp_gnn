import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test model')
    # correct the name of the folder that have the instances insatnces-> instances
    parser.add_argument('--data_dir', type=str, default='../tsp_input/generated_insatnces_3000_size_50/', help='Dataset directory')
    parser.add_argument('--tb_dir', type=str, default='../atsp_model_train_result', help='Tensorboard log directory')

    # ATSP graph parameters
    parser.add_argument('--atsp_size', type=int, default=50, help="Size of the atsp to be solved")
    parser.add_argument('--to_homo', type=bool, default=False, help="tranfer the pre-processed graph into homogines")
    parser.add_argument('--half_st', type=bool, default=False, help="add only source target graph half of it, ")

    # Model parameters
    parser.add_argument('--model', type=str, default='HetroGATSum', help='set the model name to use')
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden feature dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output feature dimension')
    parser.add_argument('--relation_types', type=str, default='ss tt pp', help='Number of relation types')  
    parser.add_argument('--n_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--n_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--jk', type=str, default='cat', choices=['cat'])

    # Hyper-parameter about the training/testing
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=4, help='Fix the seed of exprs')
    parser.add_argument('--n_trials', type=int, default=1, help='Number of model trials')
    parser.add_argument('--n_samples_result_train', type=int, default=30, help='Number of samples to print the average gap cost extra in each epoch training')

    # Flag for using GPU
    parser.add_argument('--device', type=str, default='cuda', help="Number of gpu to be used")

    args = parser.parse_args()
    
    return args

def parse_args_test():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--atsp_size', type=int, default=100, help="Size of the atsp to be solved")
    parser.add_argument('--data_path', type=str, default=f'../tsp_input/generated_insatnces_30_size_', help='Dataset directory')
    parser.add_argument('--model_path', default='../atsp_model_train_result/Oct17_04-09-53_HetroGATConcat_trained_ATSP50/trial_0', type=str)
    parser.add_argument('--time_limit', type=float, default=0.16, help='Time limit for the 2 opt search in seconds') 
    parser.add_argument('--perturbation_moves', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda', help="Number of gpu to be used")

    args = parser.parse_args()

    return args

def load_params(args, params):
    # Update args with values from params
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

