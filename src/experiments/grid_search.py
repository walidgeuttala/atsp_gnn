from train import train
import itertools
from utils import *
from args import parse_args


search_space = {
        'half_st': [True, False],
        'model': ['HetroGATConcat', 'HetroGATSum']
    }



def main():
    fix_seed(1234)

    args = parse_args()
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    combinations = list(itertools.product(*values))
    best_loss = float(1e6)
    idx = 0
    for combination in combinations:
        param_dict = dict(zip(keys, combination))
        for key, value in param_dict.items():
            setattr(args, key, value)
        val_loss = train(args, f'search_space_{idx}')
        print(f"Best Validation Loss : {val_loss}")
        if best_loss > val_loss:
            best_loss = val_loss
            saved_args = args
        idx += 1
    print("best args : ",saved_args, flush=True)

if __name__ == '__main__':
    main()