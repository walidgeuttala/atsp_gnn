#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3            # Job name
#SBATCH --time=0-04:00:00          # Maximum walltime
#SBATCH --partition=gpu            # Select GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem-per-cpu=4000         # Memory per CPU core in MB
#SBATCH --nodes=1                  # Request 1 node

set -e  # exit on first error

ENV_NAME="gnn5"

# Source conda (mandatory for batch scripts)
source ~/c_gnn_001/miniconda3/etc/profile.d/conda.sh

# Create environment if it doesn't exist
if ! conda env list | grep -q $ENV_NAME; then
    echo "[INFO] Creating environment $ENV_NAME..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# Activate environment
conda activate $ENV_NAME

# Environment variables for CUDA and project
export PYTHONPATH=/project/c_gnn_001/code/tsp/atsp_gnn/:$PYTHONPATH
export CUDA_HOME=/opt/software/packages/cuda/12.1
export LD_LIBRARY_PATH=/project/c_gnn_001/glibc_install/glibc-2.31/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install packages (if needed)
pip install --upgrade pip
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install pyg_lib==0.3.1+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install "numpy<2" scikit-learn==1.5.2 pandas pyyaml tqdm matplotlib tsplib95 lkh networkx
pip install -e ~/c_gnn_001/torch_sparse_build/pytorch_sparse
pip install -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-scatter

echo "[INFO] Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"

# Run your Python script
python /project/c_gnn_001/code/tsp/atsp_gnn/src/data/graph_transforms.py
