from train import run
import itertools
from utils import *
from args import parse_args

# The grid search space
search_space = {
        "embed_dim": [256, 128],
        "embd_dim2": [256, 128],
        "n_layers": [4],
        "lr_init": [1e-3],
        "n_heads": [32],
        "kj": ['cat']
    }

def main():
    fix_seed(1234)

    args = parse_args()
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    combinations = list(itertools.product(*values))
    best_loss = float(1e6)
    for combination in combinations:
        param_dict = dict(zip(keys, combination))
        for key, value in param_dict.items():
            setattr(args, key, value)

        val_loss = run(args)
        if best_loss > val_loss:
            best_loss = val_loss
            saved_args = args

    print("best args : ",saved_args, flush=True)