#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp22          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=100000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

#!/bin/bash

# Create new folders
mkdir -p src/core
mkdir -p src/models
mkdir -p src/data
mkdir -p src/solvers
mkdir -p src/experiments
mkdir -p src/visualization
mkdir -p data/raw data/processed
mkdir -p jobs
mkdir -p saved_models
mkdir -p results/graphs

# --- CORE ---
mv core/args.py src/core/
mv core/utils.py src/core/

# --- MODELS ---
mv core/model.py src/models/gnn_model.py
mv gnngls/model.py src/models/hetero_gat.py
mv gnngls/model_utils.py src/models/model_utils.py
mv gnngls/datasets.py src/data/gnngls_datasets.py
mv gnngls/algorithms.py src/models/algorithms.py
mv gnngls/operators.py src/models/local_search_ops.py
mv experiments/models_test.py src/models/rgcn4.py   # keep separate test arch

# Create __init__.py for models
cat > src/models/__init__.py <<EOL
from .gnn_model import *
from .hetero_gat import *
from .rgcn4 import *
from .model_utils import *

def get_model(name, **kwargs):
    if name.lower() == "gnn":
        from .gnn_model import GNNModel
        return GNNModel(**kwargs)
    elif name.lower() == "rgcn4":
        from .rgcn4 import RGCN4
        return RGCN4(**kwargs)
    elif name.lower().startswith("hetero"):
        from .hetero_gat import get_hetero_model
        return get_hetero_model(name, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
EOL

# --- DATA ---
mv data/* src/data/ 2>/dev/null
mv src/data/data_utils.py src/data/data_utils.py
mv src/data/convert_atsp_to_tsp.py src/data/convert_atsp_to_tsp.py
mv src/data/preprocess_dataset.py src/data/preprocess_dataset.py
mv src/data/transform_graph.py src/data/transform_graph.py
mv src/data/line_graph_utils.py src/data/line_graph_utils.py
mv src/data/generate_instances.py src/data/generate_instances.py

# --- SOLVERS ---
mv solvers/* src/solvers/

# --- EXPERIMENTS ---
mv experiments/* src/experiments/

# --- VISUALIZATION ---
mv visualization/* src/visualization/

# --- JOBS ---
mv jobs/* jobs/

# --- RESULTS ---
mv result_plot*.pdf results/
mv graphs/* results/graphs/ 2>/dev/null

# --- CLEANUP ---
rm -rf gnngls
rm -rf core
rm -rf data
rm -rf experiments
rm -rf solvers
rm -rf visualization
rm -rf graphs
rm -rf __pycache__

echo "✅ Repo reorganized into new src/ structure."
