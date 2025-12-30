#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-02:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node


export PYTHONPATH=/project/c_gnn_001/code/tsp/atsp_gnn/:$PYTHONPATH
# python3 -m src.data.graph_transforms

# python3 -m src.data.preprocessor ../saved_dataset/ATSP_10x10 --n_train 0 --n_val 0 --n_test 10 --atsp_size 10
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_3000x50 --n_train 2500 --n_val 250 --n_test 250 --atsp_size 50
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x100 --n_train 0 --n_val 0 --n_test 30 --atsp_size 100
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x150 --n_train 0 --n_val 0 --n_test 30 --atsp_size 150
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x250 --n_train 0 --n_val 0 --n_test 30 --atsp_size 250
# python3 -m src.data.dataset_generator \
#   1000 30 ../saved_dataset \
#   --from_pt_file ../../matnet/data/atsp/ATSP1000.pt \
#   --regret_mode row_best --parallel
# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x1000 --n_train 0 --n_val 0 --n_test 30 --atsp_size 1000

# python3 -m src.test
# python3 -m src.engine.visualize_batch_summary \
#   --summary-csv ../jobs/search/output.csv \
#   --output-dir ../jobs/search/plots

# python3 -m src.test \
#   --hybrid-logs /project/c_gnn_001/code/tsp/regret_driven_heuristics/slurm-12237679.out \
#   --batch-summary ../jobs/search/batch_test_summary.csv \
#   --out-csv ../jobs/search/all_output.csv


# python3 -m src.test \
#   --hybrid-logs /project/c_gnn_001/code/tsp/regret_driven_heuristics/slurm-12237679.out \
#   --batch-summary ../jobs/search/batch_test_summary.csv \
#   --out-csv ../jobs/search/merged.csv \
#   --keep-all-iterations

# python3 -m src.visualization.plot_result

python3 -m src.data.batch_preprocess