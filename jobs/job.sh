#!/bin/bash
#SBATCH -A c_gnn_001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=walidgeuttala@gmail.com  # Email address for notifications

# Your job commands here
# python generate_instances.py 30 100 ../tsp_input/atsp_100_samples_30_solved_test_only ../tsp_input/generated_insatnces_30_size_100
# python generate_instances.py 30 150 ../tsp_input/atsp_150_samples_30_solved_test_only ../tsp_input/generated_insatnces_30_size_150
# python generate_instances.py 30 250 ../tsp_input/atsp_250_samples_30_solved_test_only ../tsp_input/generated_insatnces_30_size_250
# python generate_instances.py 30 500 ../tsp_input/atsp_500_samples_30_solved_test_only ../tsp_input/generated_insatnces_30_size_500
# python generate_instances.py 100 250 atsp_250_solved_test_only ../generated_insatnces_100_size_250

# python generate_instances.py 3000 50 atsp_50_solved ../generated_insatnces_3000_size_50
# python preprocess_dataset.py ../generated_insatnces_3000_size_50

# python preprocess_dataset.py ../tsp_input/generated_insatnces_30_size_100
# python preprocess_dataset.py ../tsp_input/generated_insatnces_30_size_150
# python preprocess_dataset.py ../tsp_input/generated_insatnces_30_size_250
# python preprocess_dataset.py ../tsp_input/generated_insatnces_30_size_1000


#python test.py ../generated_insatnces_1000_size_100/test.txt ../model_result_try/Jul13_16-09-51_6937d04a2f2f4c90b92ad923ed0d8304/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../output_ATSP_samples_1000_size_100

#python atsp_solve_GLOP_format.py



# My best model is ../atsp_model_train_result/Oct07_01-45-55_HetroGAT_trained_ATSP50/trial_0/tral_0/test_atsp150/results.json

#python test.py






# python /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/dataset_generator.py 10 10 /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset --parallel
# python /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/validate_atsp.py
python /project/c_gnn_001/code/tsp/atsp_gnn/src/data2/preprocessor.py /project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_10x10 --n_train 10 --n_val 0 --n_test 0 --atsp_size 10
