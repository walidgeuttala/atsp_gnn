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


# python3 /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/dataset_generator.py 10 10 /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset --parallel
# python3 /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/validate_atsp.py
# python3 -m src.data.preprocessor /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_30x100 --n_train 30 --n_val 0 --n_test 0 --atsp_size 10
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x500 --n_train 0 --n_val 0 --n_test 30 --atsp_size 500

python -m src.engine.run --mode train --framework dgl --data_dir ../saved_dataset/ATSP_3000x50 --model HetroGAT --atsp_size 50 --batch_size 1 --n_epochs 100 --lr_init 1e-3 --tb_dir ./runs --agg concat


# python3 -m src.engine.run \
#     --mode test \
#     --framework pyg \
#     --atsp_size 50 \
#     --data_path ../saved_dataset/ATSP_100x50 \
#     --model_path ../saved_model/Oct17_04-09-53_HetroGATConcat_trained_ATSP50/trial_0/checkpoint_best_val.pt \
#     --device cuda

