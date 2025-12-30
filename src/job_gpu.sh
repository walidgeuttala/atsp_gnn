#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-05:10:00        # Maximum walltime (30 minutes)
#SBATCH --partition=ai           # Select the ai partition
#SBATCH --gres=gpu:1       # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=80000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# helps for the src path to be added in the code of the python code
export PYTHONPATH=/project/c_gnn_001/code/tsp/atsp_gnn/:$PYTHONPATH

# python3 -m src.data.preprocessor ../saved_dataset/ATSP_30x500 --n_train 0 --n_val 0 --n_test 30 --atsp_size 500

# python3 -m src.engine.run --mode train --framework dgl --data_dir ../saved_dataset/ATSP_3000x50 \
#      --model HetroGAT --atsp_size 50 --batch_size 32 --n_epochs 1 --lr_init 1e-3 --tb_dir ./runs \
#       --agg sum --device cuda --undirected

# python3 -m src.engine.run --mode test --framework dgl --atsp_size 100 --model HetroGAT --batch_size 1 \
#     --data_path ../saved_dataset/ATSP_30x100 --tb_dir ./runs --agg concat\
#     --model_path ../jobs/runs/Sep13_01-59-49_HetroGATconcat_ATSP50_Comboss_tt_pp/trial_0/best_model.pt \
#     --device cuda --undirected

# python3 -m src.engine.run --mode test --framework dgl --atsp_size 150 --model HetroGAT --batch_size 1 \
#     --data_path ../saved_dataset/ATSP_30x150 --tb_dir ./runs --agg concat\
#     --model_path ../jobs/runs/Sep13_01-59-49_HetroGATconcat_ATSP50_Comboss_tt_pp/trial_0/best_model.pt \
#     --device cuda --undirected

# python3 -m src.engine.run --mode test --framework dgl --atsp_size 250 --model HetroGAT --batch_size 1 \
#     --data_path ../saved_dataset/ATSP_30x250 --tb_dir ./runs --agg concat\
#     --model_path ../jobs/runs/Sep13_01-59-49_HetroGATconcat_ATSP50_Comboss_tt_pp/trial_0/best_model.pt \
#     --device cuda --undirected

# python3 -m src.engine.run --mode test --framework dgl --atsp_size 500 --model HetroGAT --batch_size 1 \
#     --data_path ../saved_dataset/ATSP_30x500 --tb_dir ./runs --agg concat\
#     --model_path ../jobs/runs/Sep13_01-59-49_HetroGATconcat_ATSP50_Comboss_tt_pp/trial_0/best_model.pt \
#     --device cuda --undirected

# python3 -m src.engine.run_large --mode test --framework dgl \
#   --atsp_size 1000 --sub_size 250 --model HetroGAT --batch_size 1 \
#   --data_path ../saved_dataset/ATSP_30x1000 --template_path ../saved_dataset/ATSP_30x250 \
#   --tb_dir ./runs --agg sum \
#   --model_path ../jobs/runs/Sep14_03-32-03_HetroGATsum_ATSP50_Comboss_st_tt_pp/trial_0/best_model.pt \
#   --device cuda --undirected


# python3 -m src.engine.search_all_combo --mode train --framework dgl --data_dir ../saved_dataset/ATSP_3000x50 \
#      --model HetroGAT --atsp_size 50 --batch_size 32 --n_epochs 1 --lr_init 1e-3 --tb_dir ./runs \
#       --agg concat --device cuda --undirected

# python3 -m src.engine.search_all_combo --mode train --framework dgl \
#     --data_dir ../saved_dataset/ATSP_3000x50 --model HetroGAT --atsp_size 50 \
#     --batch_size 32 --n_epochs 100 --lr_init 1e-3 --tb_dir ./runs \
#     --device cuda --undirected --n_trials 20 --seed 42 \
#     --relation_types ss st tt pp --time_limit 5.0 --perturbation_moves 30

# python3 -m src.engine.batch_test_search_models 

# python3 -m src.engine.run_export \
#   --mode test \
#   --framework dgl \
#   --model HetroGAT \
#   --model_path ../jobs/search/12201359/best_model_rel_st_sum.pt \
#   --data_path ../saved_dataset/ATSP_30x500 \
#   --relation_types st \
#   --agg sum \
#   --atsp_size 500 \
#   --device cuda \
#   --time_limit 0.1667 \
#   --perturbation_moves 30

# python3 -m src.engine.run_large --mode test --framework dgl --model HetroGAT \
#   --model_path ../jobs/search/12201357/best_model_rel_pp_ss_st_tt_attn.pt \
#   --data_path ../saved_dataset/ATSP_30x1000 \
#   --template_path ../saved_dataset/ATSP_30x250 \
#   --atsp_size 1000 \
#   --sub_size 250 \
#   --batch_size 4 \
#   --time_limit 1 \
#   --perturbation_moves 30 \
#   --relation_types pp ss st tt \
#   --agg attn \
#   --results_dir ../jobs/search/12201357/best_model_rel_pp_ss_st_tt_attn

# python3 -m src.engine.run_export --mode test --framework dgl \
#     --atsp_size 50 --model HetroGAT --agg concat \
#     --data_path ../saved_dataset/ATSP_3000x50 \
#     --model_path ../jobs/search/12201358/best_model_rel_pp_ss_st_tt_concat.pt \
#     --device cuda --undirected --relation_types ss st tt pp

# python3 -m src.data.validate_atsp --instances_dir "../jobs/search/12201358/best_model_rel_pp_ss_st_tt_concat/trial_0/test_atsp50" --atsp_size 50

# python3 -m src.engine.search_all_combo \
#     --data_dir ../saved_dataset/ATSP_3000x50 --model HetroGAT \
#     --framework dgl --device cuda --undirected \
#     --n_trials 20 --seed 42 --batch_size 32 --n_epochs 100 \
#     --relation_types ss st tt pp \
#     --relation_subsets st

# python3 -m src.engine.run \
#     --mode train --framework dgl --model HetroGAT \
#     --data_dir ../saved_dataset/ATSP_3000x50 \
#     --relation_types st \
#     --agg concat \
#     --device cuda --undirected \
#     --batch_size 32 --n_epochs 100 --lr_init 1e-3 --seed 42

# python3 -m src.engine.run_export --mode test --framework dgl --model HetroGAT \
#   --model_path /project/c_gnn_001/code/tsp/atsp_gnn/jobs/search/12201358/best_model_rel_pp_ss_st_tt_concat.pt \
#   --data_path /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_30x1000 \
#   --template_path /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_30x250 \
#   --atsp_size 1000 \
#   --time_limit 0.3 \
#   --perturbation_moves 30 \
#   --relation_types pp ss st tt \
#   --agg concat \
#   --results_dir /project/c_gnn_001/code/tsp/atsp_gnn/jobs/search/12201358/best_model_rel_pp_ss_st_tt_concat_export

# python3 -m src.engine.run_eval_custom
python3 -m src.engine.batch_run_export