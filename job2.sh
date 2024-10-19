#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-02:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1       # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node


# best results model that I used in the 64 size is the sum in the paper ../model_result_try/Jul13_16-09-51_6937d04a2f2f4c90b92ad923ed0d8304/checkpoint_best_val.pt

python train.py
#python test.py
