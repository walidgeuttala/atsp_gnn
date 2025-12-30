import subprocess
import os
import re

base_dir = "/project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/tsp_variantes"
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

def extract_atsp_size(folder_name):
    # Match the first number after an underscore (e.g., HCP_100x30 → 100)
    match = re.search(r'_(\d+)x', folder_name)
    return int(match.group(1)) if match else None

for folder in folders:
    if folder.startswith("KTSP"):
        print(f"Skipping folder (KTSP): {folder}")
        continue
    atsp_size = extract_atsp_size(folder)
    if atsp_size is None:
        print(f"Could not extract ATSP size from folder name: {folder}")
        continue
    dataset_dir = os.path.join(base_dir, folder)
    print(f"Processing {dataset_dir} with ATSP size {atsp_size} ...")
    cmd = [
        "python3", "-m", "src.data.preprocessor",
        dataset_dir,
        "--n_train", "0",
        "--n_val", "0",
        "--n_test", "30",
        "--atsp_size", str(atsp_size)
    ]
    subprocess.run(cmd, check=True)