# Graph Neural Network Guided Local Search for ATSP

This repository contains the implementation of a Graph Neural Network (GNN) guided local search for solving the Asymmetric Traveling Salesperson Problem (ATSP). It leverages Deep Learning frameworks (DGL and PyTorch Geometric) to learn heuristics for the LKH solver.

## Setup

### Environment Installation

We recommend using Conda to manage the environment. The detailed requirements are listed in `src/req.txt`.

```bash
# Create new Conda environment with Python 3.11
conda create -n graph2 python=3.11 -y
conda activate graph2

# Install System Dependencies (CUDA 12.1 example)
export CUDA_HOME=/opt/software/packages/cuda/12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install PyTorch
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install PyTorch Geometric
conda install pyg -c pyg -y

# Install DGL
pip3 install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# Install additional libraries
conda install -c conda-forge scikit-learn pandas numpy tqdm matplotlib networkx tsplib95 optuna -y
pip3 install lkh
```

### PYTHONPATH

For the scripts to import modules correctly, you must add the current directory to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Workflows

### 1. Dataset Generation

Generate synthetic ATSP datasets or process existing TSPLIB instances.

```bash
# Generate 100 instances of size 50 nodes
python src/data/dataset_generator.py 100 50 data/ --regret_mode fixed_edge_lkh --parallel
```
This creates `data/ATSP_50x100/` with `.pkl` files and a `summary.csv`.

### 2. Training (Standard)

Train a model on a generated dataset.

```bash
# Train HeteroGAT on ATSP-50
python -m src.engine.run --mode train --framework dgl --data_dir data/ATSP_50x100 \
    --model HetroGAT --atsp_size 50 --batch_size 32 --n_epochs 100 \
    --agg sum --device cuda --undirected
```

### 3. Hyperparameter Search (Advanced)

Use `src.engine.search_all_combo` to run Optuna-based hyperparameter optimization. This script assumes a strategy where configurations are pre-screened on size 500 instances for OOM errors before training on size 50.

```bash
python -m src.engine.search_all_combo --mode train --framework dgl \
    --data_dir data/ATSP_3000x50 --model HetroGAT --atsp_size 50 \
    --batch_size 32 --n_epochs 100 --lr_init 1e-3 --tb_dir ./runs \
    --device cuda --undirected --n_trials 20 --seed 42 \
    --relation_types ss st tt pp --relation_subsets st,ss_st
```

### 4. Testing

Evaluate a trained model on standard test sets.

```bash
python -m src.engine.run --mode test --framework dgl --atsp_size 100 --model HetroGAT --batch_size 1 \
    --data_path data/ATSP_30x100 --tb_dir ./runs --agg sum \
    --model_path runs/Trial_0/best_model.pt \
    --device cuda --undirected
```

### 5. Large Graph Inference

For large graphs (e.g., 1000 nodes), use `src.engine.run_large` which employs a subgraph-based approach.

```bash
python -m src.engine.run_large --mode test --framework dgl \
  --atsp_size 1000 --sub_size 250 --model HetroGAT --batch_size 1 \
  --data_path data/ATSP_30x1000 --template_path data/ATSP_30x250 \
  --tb_dir ./runs --agg sum \
  --model_path runs/Trial_0/best_model.pt \
  --device cuda --undirected
```

## Project Structure

- `src/engine/`: Core execution logic.
    - `run.py`: Main entry for standard training/testing.
    - `run_large.py`: Inference on large graphs using subgraphs.
    - `search_all_combo.py`: Optuna hyperparameter search with OOM screening.
    - `batch_run_export.py`: Multi-model export utils.
- `src/models/`: GNN model definitions (`models_dgl.py`).
- `src/data/`: Data generation (`dataset_generator.py`) and loading.
