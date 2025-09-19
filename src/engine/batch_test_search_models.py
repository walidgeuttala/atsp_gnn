"""
Batch testing for saved search models.

Scans `jobs/search/*` for checkpoints saved by `search_all_combo.py` in the
format `best_model_rel_{relations}_{agg}.pt`, and evaluates each model on
ATSP sizes 100, 150, 250, and 500 using the existing testing utilities.

For each (model, size):
- Runs the GNN to predict regrets, runs GLS, and computes gaps/costs.
- Saves per-instance outputs (weights, regret, regret_pred, costs) into
  `{model_dir}/trial_0/test_atsp{size}/instance<name>.txt` to integrate with
  downstream heuristic pipelines.
- Saves a JSON summary to `{model_dir}/test_atsp{size}/results.json`.

Additionally writes CSV summaries aggregating all runs under `jobs/search`.

This script prefers DGL (default framework in this repo). If a checkpoint was
trained with PyG (framework='pyg' in its args), it falls back to a PyG path.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import tqdm

try:
    from torch.profiler import profile, ProfilerActivity
except ImportError:  # Older torch versions
    profile = None
    ProfilerActivity = None

# Ensure both project root and src/ are on sys.path so that modules that import
# 'data.*' or 'utils.*' (absolute from src root) resolve correctly, matching
# search_all_combo.py behavior.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PROJ_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
for _p in (_PROJ_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local imports via src.* to honor package structure
from src.data.dataset_dgl import ATSPDatasetDGL
from src.models.models_dgl import get_dgl_model
from src.utils.algorithms import guided_local_search, nearest_neighbor
from src.utils.atsp_utils import tour_cost, optimal_cost


def add_diag(num_nodes: int, values_1d: torch.Tensor) -> torch.Tensor:
    """Place a length N*(N-1) vector into an NxN matrix with zero diagonal.

    Order matches dataset edge ordering: i from 0..N-1, j from 0..N-1, skip i==j.
    """
    n = int(num_nodes)
    out = torch.zeros(n, n, dtype=torch.float32)
    cnt = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out[i, j] = values_1d[cnt]
            cnt += 1
    return out


@dataclass
class EvalConfig:
    search_root: Path
    sizes: Tuple[int, ...] = (100, 150, 250, 500)
    device: str = 'cuda'
    time_limit: float = 5.0/30.0
    perturbation_moves: int = 30
    only_dirs: Optional[List[str]] = None  # e.g., ['12201357']
    limit_models: Optional[int] = None
    framework: str = 'dgl'  # 'auto' | 'dgl' | 'pyg'
    profile_flops: bool = True
    reuse_predictions: bool = False
    override_sizes: Optional[Tuple[int, ...]] = None


def discover_checkpoints(root: Path, only_dirs: Optional[List[str]] = None) -> List[Path]:
    candidates: List[Path] = []
    if not root.exists():
        return candidates
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        if only_dirs and sub.name not in set(only_dirs):
            continue
        # find all best_model_rel_*.pt in the immediate subdir
        for pt in sorted(sub.glob('best_model_rel_*_*.pt')):
            candidates.append(pt)
    return candidates


def dataset_path_for_size(project_root: Path, size: int) -> Path:
    return project_root / 'saved_dataset' / f'ATSP_30x{size}'


def load_ckpt_args(ckpt_path: Path, map_location: str) -> Dict:
    state = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    args_dict = state.get('args', {})
    return args_dict


def build_eval_args(base_args: Dict, size: int, device: torch.device, model_path: str) -> argparse.Namespace:
    # Build a light args namespace compatible with get_dgl_model and tester
    from types import SimpleNamespace
    ns = SimpleNamespace()
    # Core
    setattr(ns, 'device', device)
    setattr(ns, 'atsp_size', size)
    setattr(ns, 'relation_types', list(base_args.get('relation_types', ['ss', 'st', 'tt', 'pp'])))
    setattr(ns, 'undirected', base_args.get('undirected', False))
    setattr(ns, 'agg', base_args.get('agg', 'sum'))
    setattr(ns, 'model', base_args.get('model', 'HetroGAT'))
    setattr(ns, 'framework', base_args.get('framework', 'dgl'))
    # Solver
    setattr(ns, 'time_limit', base_args.get('time_limit', 5.0))
    setattr(ns, 'perturbation_moves', base_args.get('perturbation_moves', 30))
    # Paths
    setattr(ns, 'model_path', model_path)
    # Optionally present fields to help factories
    for k in ['input_dim', 'hidden_dim', 'output_dim', 'num_gnn_layers', 'num_heads', 'jk', 'skip_connection']:
        if k in base_args:
            setattr(ns, k, base_args[k])
    return ns


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def result_json_path(ckpt_path: Path, size: int) -> Path:
    """Return the expected summary JSON path for a checkpoint/size pair."""
    model_dir = ckpt_path.parent
    ckpt_stem = ckpt_path.stem.replace(os.sep, '_')
    return model_dir / ckpt_stem / f"test_atsp{size}" / "results.json"


def extract_model_metadata(base_args: Dict) -> Dict:
    return {
        'hidden_dim': base_args.get('hidden_dim'),
        'num_heads': base_args.get('num_heads'),
        'num_gnn_layers': base_args.get('num_gnn_layers') or base_args.get('num_layers'),
    }


def compute_model_param_count(
    ckpt_path: Path,
    base_args: Dict,
    fallback_size: int,
) -> Optional[int]:
    try:
        size = int(base_args.get('atsp_size', fallback_size))
    except Exception:
        size = fallback_size
    try:
        meta_device = torch.device('cpu')
        args = build_eval_args(base_args, size=size, device=meta_device, model_path=str(ckpt_path))
        model = get_dgl_model(args)
        count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
        return count
    except Exception as exc:
        print(f"Failed to compute param count for {ckpt_path}: {exc}")
        return None


def measure_forward_flops(
    model,
    graph,
    features,
    device: torch.device,
    enabled: bool,
) -> Optional[float]:
    if not enabled:
        return None
    if profile is None or ProfilerActivity is None:
        return None
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda' and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
            with torch.no_grad():
                model(graph, features)
        total_flops = 0.0
        for evt in prof.key_averages():
            flops = getattr(evt, 'flops', None)
            if flops is not None:
                total_flops += flops
        return float(total_flops) if total_flops > 0 else None
    except Exception as exc:
        print(f"FLOP profiling failed: {exc}")
        return None


def refresh_flops_for_size(
    ckpt_path: Path,
    base_args: Dict,
    size: int,
    project_root: Path,
    enabled: bool,
) -> Tuple[Optional[float], Optional[float]]:
    if not enabled:
        return None, None
    if profile is None or ProfilerActivity is None:
        return None, None

    data_dir = dataset_path_for_size(project_root, size)
    if not data_dir.exists():
        return None, None

    eval_device = torch.device('cpu')
    model = None
    try:
        args = build_eval_args(base_args, size=size, device=eval_device, model_path=str(ckpt_path))
        model = get_dgl_model(args)
        model.eval()

        dataset = ATSPDatasetDGL(
            data_dir=data_dir,
            split='test',
            atsp_size=size,
            relation_types=tuple(args.relation_types),
            undirected=args.undirected,
            device=eval_device,
            load_once=False,
        )

        if len(dataset) == 0:
            return None, None

        H_sample, _ = dataset[0]
        flops_forward = measure_forward_flops(
            model,
            H_sample,
            H_sample.ndata['weight'],
            eval_device,
            enabled=True,
        )
        if flops_forward is None:
            return None, None
        total_flops = float(flops_forward * len(dataset))
        return float(flops_forward), total_flops
    except Exception as exc:
        print(f"Failed to refresh FLOPs for {ckpt_path} size {size}: {exc}")
        return None, None
    finally:
        if model is not None:
            del model
def build_summary_row(
    ckpt: Path,
    relations,
    agg: str,
    framework: str,
    model_name: str,
    size: int,
    results: Dict,
    metadata: Dict,
) -> Dict:
    param_count = results.get('model_param_count')
    if param_count is None:
        param_count = metadata.get('model_param_count')
    hidden_dim = results.get('hidden_dim', metadata.get('hidden_dim'))
    num_heads = results.get('num_heads', metadata.get('num_heads'))
    num_layers = results.get('num_gnn_layers', metadata.get('num_gnn_layers'))
    flops_forward = results.get('flops_per_forward')
    flops_total = results.get('estimated_total_flops')
    return {
        'slurm_dir': ckpt.parent.name,
        'model_file': ckpt.name,
        'relations': '_'.join(relations) if isinstance(relations, (list, tuple)) else relations,
        'agg': agg,
        'framework': framework,
        'model': model_name,
        'atsp_size': size,
        'avg_init_gap': results.get('avg_init_gaps'),
        'avg_final_gap': results.get('avg_final_gaps'),
        'avg_init_cost': results.get('avg_init_costs'),
        'avg_final_cost': results.get('avg_final_costs'),
        'avg_opt_cost': results.get('avg_opt_costs'),
        'total_model_time': results.get('total_model_time'),
        'total_gls_time': results.get('total_gls_time'),
        'model_param_count': param_count,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'num_gnn_layers': num_layers,
        'flops_per_forward': flops_forward,
        'estimated_total_flops': flops_total,
    }


# Predefined total GLS budgets (seconds) per ATSP size.
_TOTAL_GLS_TIME_BY_SIZE = {
    100: 2.5,
    150: 3.5,
    250: 5.0,
    500: 7.0,
}


def load_saved_regret_pred_matrix(
    base_out_dir: Path,
    size: int,
    instance_name: str,
) -> Tuple[Optional[np.ndarray], Path]:
    safe_name = str(instance_name).replace(os.sep, '_')
    path = base_out_dir / "trial_0" / f"test_atsp{size}" / f"instance{safe_name}.txt"
    if not path.exists():
        return None, path

    matrix: List[List[float]] = []
    capture = False
    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('regret_pred:'):
                capture = True
                matrix = []
                continue
            if capture:
                if not stripped:
                    break
                row = [float(x) for x in stripped.split()]
                matrix.append(row)
    if not matrix:
        return None, path
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (size, size):
        return None, path
    return arr, path


def write_instance_artifacts(
    out_dir: Path,
    instance_name: str,
    edge_weight: np.ndarray,
    regret_mat: np.ndarray,
    regret_pred_mat: np.ndarray,
    opt_cost: float,
    num_iterations: int,
    init_cost: float,
    best_cost: float,
) -> None:
    ensure_dir(out_dir)
    # sanitize instance name to avoid path separators
    safe_name = str(instance_name).replace(os.sep, '_')
    with open(out_dir / f"instance{safe_name}.txt", "w") as f:
        f.write("edge_weight:\n")
        np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
        f.write("\n")

        f.write("regret:\n")
        np.savetxt(f, regret_mat, fmt="%.8f", delimiter=" ")
        f.write("\n")

        f.write("regret_pred:\n")
        np.savetxt(f, regret_pred_mat, fmt="%.8f", delimiter=" ")
        f.write("\n")

        f.write(f"opt_cost: {opt_cost}\n")
        f.write(f"num_iterations: {num_iterations}\n")
        f.write(f"init_cost: {init_cost}\n")
        f.write(f"best_cost: {best_cost}\n")


def evaluate_model_dgl(
    ckpt_path: Path,
    size: int,
    data_dir: Path,
    device: torch.device,
    time_limit: float,
    perturbation_moves: int,
    profile_flops: bool,
    reuse_predictions: bool,
) -> Dict:
    """Evaluate a DGL model checkpoint on the given ATSP size and dataset.

    Returns a result dict with aggregates and writes per-instance artifacts.
    """
    base_args = load_ckpt_args(ckpt_path, map_location=device)
    args = build_eval_args(base_args, size=size, device=device, model_path=str(ckpt_path))
    # Use the model factory which also loads weights when model_path is set
    model = get_dgl_model(args)
    model.eval()
    model_param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    results_metadata = extract_model_metadata(base_args)
    results_metadata['model_param_count'] = model_param_count

    # Dataset in lazy mode to retrieve NX graphs
    test_dataset = ATSPDatasetDGL(
        data_dir=data_dir,
        split='test',
        atsp_size=size,
        relation_types=tuple(args.relation_types),
        undirected=args.undirected,
        device=device,
        load_once=False,
    )

    flops_per_forward = None
    estimated_total_flops = None
    if profile_flops and not reuse_predictions and len(test_dataset) > 0:
        H_sample, sample_graph = test_dataset[0]
        sample_features = H_sample.ndata['weight']
        flops_per_forward = measure_forward_flops(
            model,
            H_sample,
            sample_features,
            device,
            enabled=True,
        )
        if flops_per_forward is not None:
            estimated_total_flops = float(flops_per_forward * len(test_dataset))
        del H_sample, sample_features, sample_graph

    # Output dirs (model-specific to avoid collisions within same SLURM folder)
    model_dir = ckpt_path.parent
    ckpt_stem = ckpt_path.stem.replace(os.sep, '_')
    base_out_dir = model_dir / ckpt_stem
    out_dir_json = base_out_dir / f"test_atsp{size}"
    out_dir_txt = base_out_dir / "trial_0" / f"test_atsp{size}"
    ensure_dir(out_dir_json)
    ensure_dir(out_dir_txt)

    results = {
        'init_gaps': [],
        'final_gaps': [],
        'init_costs': [],
        'final_costs': [],
        'opt_costs': [],
        'avg_cnt_search': [],
        'total_model_time': 0.0,
        'total_gls_time': 0.0,
        'model_times': [],
        'gls_times': [],
        'num_nodes': size,
        'instances': [],
        'model_param_count': model_param_count,
        'hidden_dim': results_metadata.get('hidden_dim'),
        'num_heads': results_metadata.get('num_heads'),
        'num_gnn_layers': results_metadata.get('num_gnn_layers'),
        'flops_per_forward': flops_per_forward,
        'estimated_total_flops': estimated_total_flops,
    }

    pbar = tqdm.tqdm(range(len(test_dataset)), desc=f"{ckpt_path.name} @ {size}")
    for idx in pbar:
        # Load graph and build scaled features
        H, G = test_dataset[idx]
        x = H.ndata['weight']
        y_pred = None

        if not reuse_predictions:
            # warmup model
            if idx == 0:
                with torch.no_grad():
                    _ = model(H, x)
            # Predict regrets with timing
            t0 = time.time()
            with torch.no_grad():
                y_pred = model(H, x)
            model_time = time.time() - t0
            results['total_model_time'] += model_time
            results['model_times'].append(float(model_time))

            regret_pred_vec = test_dataset.scalers.inverse_transform(
                y_pred.detach().cpu().flatten(), 'regret'
            )

            # Write predicted regrets back to graph
            edge_idx = 0
            for i in range(size):
                for j in range(size):
                    if i == j:
                        continue
                    if (i, j) in G.edges:
                        G.edges[(i, j)]['regret_pred'] = max(float(regret_pred_vec[edge_idx]), 0.0)
                    edge_idx += 1

            regret_pred_mat_np = add_diag(size, y_pred.detach().cpu().flatten()).numpy()
        else:
            saved_mat, saved_path = load_saved_regret_pred_matrix(
                base_out_dir, size, test_dataset.instances[idx]
            )
            if saved_mat is None:
                raise FileNotFoundError(
                    f"Missing saved predictions for instance {test_dataset.instances[idx]} (size {size}) at {saved_path}"
                )
            for i in range(size):
                for j in range(size):
                    if i == j:
                        continue
                    if (i, j) in G.edges:
                        G.edges[(i, j)]['regret_pred'] = max(float(saved_mat[i, j]), 0.0)
            results['model_times'].append(0.0)
            regret_pred_mat_np = saved_mat

        # Initial tour by predicted regret
        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour)
        # GLS with timing
        t_g0 = time.time()
        best_tour, best_cost, search_progress, cnt_iters = guided_local_search(
            G, init_tour, init_cost, t_lim=time.time() + float(time_limit),
            weight='weight', guides=['regret_pred'],
            perturbation_moves=perturbation_moves, first_improvement=False
        )
        gls_time = time.time() - t_g0
        results['total_gls_time'] += gls_time
        results['gls_times'].append(float(gls_time))

        opt_cost = optimal_cost(G, weight='weight')
        init_gap = (init_cost / opt_cost - 1) * 100.0
        final_gap = (best_cost / opt_cost - 1) * 100.0

        # Save per-instance artifacts
        edge_weight = np.asarray(
            [[G.edges[(i, j)]['weight'] if i != j else 0.0 for j in range(size)] for i in range(size)]
        )
        regret_mat = add_diag(size, H.ndata['regret'].detach().cpu().flatten()).numpy()

        write_instance_artifacts(
            out_dir=out_dir_txt,
            instance_name=str(test_dataset.instances[idx]),
            edge_weight=edge_weight,
            regret_mat=regret_mat,
            regret_pred_mat=regret_pred_mat_np,
            opt_cost=float(opt_cost),
            num_iterations=int(cnt_iters),
            init_cost=float(init_cost),
            best_cost=float(best_cost),
        )

        # Aggregate
        results['instances'].append(str(test_dataset.instances[idx]))
        results['init_costs'].append(float(init_cost))
        results['final_costs'].append(float(best_cost))
        results['opt_costs'].append(float(opt_cost))
        results['init_gaps'].append(float(init_gap))
        results['final_gaps'].append(float(final_gap))
        results['avg_cnt_search'].append(int(cnt_iters))

        # Free
        del H, x
        if y_pred is not None:
            del y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.set_postfix({
            'Avg_Gap_init': f"{np.mean(results['init_gaps']):.4f}",
            'Avg_Gap_best': f"{np.mean(results['final_gaps']):.4f}",
        })

    # Add average values for numeric lists only (skip 'instances')
    for key, val in list(results.items()):
        if isinstance(val, list) and len(val) > 0:
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in val):
                results[f'avg_{key}'] = float(np.mean(val))

    # Save summary JSON
    with open(out_dir_json / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def summarize_to_csv(rows: List[Dict], out_csv: Path) -> None:
    if not rows:
        return
    ensure_dir(out_csv.parent)
    # Unified columns
    cols = [
        'slurm_dir', 'model_file', 'relations', 'agg', 'framework', 'model',
        'atsp_size', 'avg_init_gap', 'avg_final_gap',
        'avg_init_cost', 'avg_final_cost', 'avg_opt_cost',
        'total_model_time', 'total_gls_time',
        'model_param_count', 'hidden_dim', 'num_heads', 'num_gnn_layers',
        'flops_per_forward', 'estimated_total_flops'
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main():
    parser = argparse.ArgumentParser(description='Batch test saved search models')
    parser.add_argument('--search_root', type=str, default=str(Path(__file__).resolve().parents[2] / 'jobs' / 'search'))
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 150, 250, 500])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--time_limit', type=float, default=5.0/30.0)
    parser.add_argument('--perturbation_moves', type=int, default=30)
    parser.add_argument('--only_dirs', type=str, nargs='*', default=None)
    parser.add_argument('--limit_models', type=int, default=None)
    parser.add_argument(
        '--profile_flops',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Profile a sample forward pass to estimate FLOPs for each model (default: enabled).',
    )
    parser.add_argument(
        '--reuse_predictions',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Reuse saved regret predictions instead of running the model.',
    )
    parser.add_argument(
        '--override_sizes',
        type=int,
        nargs='*',
        default=None,
        help='ATSP sizes to force re-evaluation (e.g., 100 150).',
    )
    args = parser.parse_args()

    cfg = EvalConfig(
        search_root=Path(args.search_root),
        sizes=tuple(args.sizes),
        device=args.device,
        time_limit=args.time_limit,
        perturbation_moves=args.perturbation_moves,
        only_dirs=args.only_dirs,
        limit_models=args.limit_models,
        profile_flops=args.profile_flops,
        reuse_predictions=args.reuse_predictions,
        override_sizes=tuple(args.override_sizes) if args.override_sizes else None,
    )

    project_root = Path(__file__).resolve().parents[2]
    device = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    checkpoints = discover_checkpoints(cfg.search_root, only_dirs=cfg.only_dirs)
    if cfg.limit_models is not None:
        checkpoints = checkpoints[: cfg.limit_models]
    if not checkpoints:
        print(f"No checkpoints found under: {cfg.search_root}")
        return

    all_rows: List[Dict] = []
    override_set = set(cfg.override_sizes) if cfg.override_sizes is not None else None
    if cfg.reuse_predictions and override_set is None:
        print('Warning: --reuse-predictions has no effect without --override_sizes; skipping reuse.')

    for ckpt in checkpoints:
        try:
            base_args = load_ckpt_args(ckpt, map_location=str(device))
            relations = base_args.get('relation_types', [])
            agg = base_args.get('agg', 'sum')
            framework = base_args.get('framework', 'dgl')
            model_name = base_args.get('model', 'HetroGAT')
            metadata = extract_model_metadata(base_args)
            metadata['model_param_count'] = None
            fallback_size = cfg.sizes[0] if cfg.sizes else base_args.get('atsp_size', 100)
            if framework == 'dgl':
                metadata['model_param_count'] = compute_model_param_count(
                    ckpt_path=ckpt,
                    base_args=base_args,
                    fallback_size=int(base_args.get('atsp_size', fallback_size)),
                )

            for size in cfg.sizes:
                data_dir = dataset_path_for_size(project_root, size)
                if not data_dir.exists():
                    print(f"Skip size {size}: dataset folder not found: {data_dir}")
                    continue

                summary_path = result_json_path(ckpt, size)
                summary_exists = summary_path.exists()
                force_override = override_set is not None and size in override_set
                apply_reuse = cfg.reuse_predictions and force_override

                if summary_exists and not force_override:
                    print(
                        f"Skip {ckpt.name} size {size}: found existing results at {summary_path}"
                    )
                    try:
                        with open(summary_path, 'r') as f:
                            existing_results = json.load(f)
                        if framework == 'dgl' and (
                            existing_results.get('flops_per_forward') is None
                            or existing_results.get('estimated_total_flops') is None
                        ) and cfg.profile_flops:
                            flops_forward, flops_total = refresh_flops_for_size(
                                ckpt_path=ckpt,
                                base_args=base_args,
                                size=size,
                                project_root=project_root,
                                enabled=cfg.profile_flops,
                            )
                            if flops_forward is not None:
                                existing_results['flops_per_forward'] = flops_forward
                                existing_results['estimated_total_flops'] = flops_total
                        all_rows.append(
                            build_summary_row(
                                ckpt,
                                relations,
                                agg,
                                framework,
                                model_name,
                                size,
                                existing_results,
                                metadata,
                            )
                        )
                    except Exception as exc:
                        print(f"Failed to load existing summary {summary_path}: {exc}")
                    continue

                if framework != 'dgl':
                    print(f"Skipping non-DGL checkpoint for now: {ckpt} (framework={framework})")
                    continue

                if summary_exists and force_override:
                    print(
                        f"Override {ckpt.name} size {size}: recomputing GLS"
                        f" (reuse_predictions={'yes' if apply_reuse else 'no'})"
                    )

                total_time_budget = _TOTAL_GLS_TIME_BY_SIZE.get(size)
                if total_time_budget is not None:
                    per_instance_time_limit = total_time_budget / 30.0
                else:
                    per_instance_time_limit = cfg.time_limit

                res = evaluate_model_dgl(
                    ckpt_path=ckpt,
                    size=size,
                    data_dir=data_dir,
                    device=device,
                    time_limit=per_instance_time_limit,
                    perturbation_moves=cfg.perturbation_moves,
                    profile_flops=cfg.profile_flops,
                    reuse_predictions=apply_reuse,
                )
                if framework == 'dgl' and (
                    res.get('flops_per_forward') is None
                    or res.get('estimated_total_flops') is None
                ) and cfg.profile_flops:
                    flops_forward, flops_total = refresh_flops_for_size(
                        ckpt_path=ckpt,
                        base_args=base_args,
                        size=size,
                        project_root=project_root,
                        enabled=cfg.profile_flops,
                    )
                    if flops_forward is not None:
                        res['flops_per_forward'] = flops_forward
                        res['estimated_total_flops'] = flops_total
                for key in (
                    'model_param_count',
                    'hidden_dim',
                    'num_heads',
                    'num_gnn_layers',
                ):
                    if metadata.get(key) is None and res.get(key) is not None:
                        metadata[key] = res.get(key)

                # Add a summary row
                all_rows.append(
                    build_summary_row(
                        ckpt,
                        relations,
                        agg,
                        framework,
                        model_name,
                        size,
                        res,
                        metadata,
                    )
                )
        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}")
            continue

    # Write global summary CSV
    out_csv = cfg.search_root / 'batch_test_summary.csv'
    summarize_to_csv(all_rows, out_csv)
    print(f"Wrote summary CSV: {out_csv}")

    # Also write per-slurm directory summaries
    rows_by_dir: Dict[str, List[Dict]] = {}
    for r in all_rows:
        rows_by_dir.setdefault(r['slurm_dir'], []).append(r)
    for slurm_id, rows in rows_by_dir.items():
        per_csv = cfg.search_root / slurm_id / 'summary.csv'
        summarize_to_csv(rows, per_csv)
        print(f"Wrote {per_csv}")


if __name__ == '__main__':
    main()
