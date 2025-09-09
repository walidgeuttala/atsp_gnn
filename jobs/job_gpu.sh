#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-01:10:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1       # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=80000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# helps for the src path to be added in the code of the python code
export PYTHONPATH=/project/c_gnn_001/code/tsp/atsp_gnn/:$PYTHONPATH
# helps to fix the issue regarding the dgl for the ld library path files 
export CUDA_HOME=/opt/software/packages/cuda/12.1
export LD_LIBRARY_PATH=/project/c_gnn_001/glibc_install/glibc-2.31/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# python3 ../src/data2/dataset_generator.py 10 10 ../saved_dataset --parallel
# python3 ../src/data2/validate_atsp.py
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x150 --n_train 0 --n_val 0 --n_test 30 --atsp_size 150

# python -m src.data.preprocessor ../saved_dataset/ATSP_30x500 --n_train 0 --n_val 0 --n_test 30 --atsp_size 500

# python3 -m src.engine.run --mode train --framework dgl --data_dir ../saved_dataset/ATSP_10x10 \
#      --model HetroGAT --atsp_size 10 --batch_size 1 --n_epochs 100 --lr_init 1e-3 --tb_dir ./runs \
#       --agg concat --device cuda --undirected

python3 -m src.engine.run --mode test --framework dgl --atsp_size 150 --model HetroGAT --batch_size 1 \
    --data_path ../saved_dataset/ATSP_30x150 --tb_dir ./runs --agg concat\
    --model_path ../jobs/runs/Sep09_02-08-40_HetroGAT_ATSP10/trial_0/best_model.pt \
    --device cuda --undirected

# python3 -m src.data.graph_transforms