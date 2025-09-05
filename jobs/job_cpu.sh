#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

export PYTHONPATH=/project/c_gnn_001/code/tsp/atsp_gnn/:$PYTHONPATH

# python /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/dataset_generator.py 10 10 /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset --parallel
# python /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/validate_atsp.py
python -m src.data.preprocessor /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_10x10 --n_train 10 --n_val 0 --n_test 0 --atsp_size 10
