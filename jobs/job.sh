#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-03:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

export CUDA_HOME=/opt/software/packages/cuda/12.1
export LD_LIBRARY_PATH=/project/c_gnn_001/glibc_install/glibc-2.31/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python3 -c "import torch; import dgl; import torch_geometric;print(torch.cuda.is_available());from torch_geometric.utils import scatter; import torch_sparse;"
# python -c "import torch; print('PyTorch imported successfully')"
# python3 -c "import torch; import dgl; import torch_geometric; from torch_geometric.utils import scatter; import torch_sparse;"
# bash script/build_dgl.sh -g
