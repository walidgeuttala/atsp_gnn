import subprocess
import os
import re

base_data_dir = "/project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/tsp_variantes"
jobs_dir = "/project/c_gnn_001/code/tsp/atsp_gnn/jobs"
model_path = f"{jobs_dir}/search/12201357/best_model_rel_pp_ss_st_tt_attn.pt"
relation_types = ["pp", "ss", "st", "tt"]
agg = "attn"
model = "HetroGAT"
framework = "dgl"
time_limit = "0.3"
perturbation_moves = "30"

folders = [f for f in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, f))]

def extract_atsp_size(folder_name):
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
    data_path = os.path.join(base_data_dir, folder)
    template_path = os.path.join(data_path, "templates", "all", f"template_{atsp_size}_pp_ss_st_tt.dgl")
    results_dir = os.path.join(jobs_dir, f"export_{folder}_attn")
    os.makedirs(results_dir, exist_ok=True)
    cmd = [
        "python3", "-m", "src.engine.run_export",
        "--mode", "test",
        "--framework", framework,
        "--model", model,
        "--model_path", model_path,
        "--data_path", data_path,
        "--template_path", template_path,
        "--atsp_size", str(atsp_size),
        "--time_limit", time_limit,
        "--perturbation_moves", perturbation_moves,
        "--relation_types", *relation_types,
        "--agg", agg,
        "--results_dir", results_dir
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)