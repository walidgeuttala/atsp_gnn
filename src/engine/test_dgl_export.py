import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from .test_dgl import ATSPTesterDGL
from src.utils.atsp_utils import tour_cost, optimal_cost
from src.utils.algorithms import guided_local_search, nearest_neighbor
from data.dataset_dgl import ATSPDatasetDGL


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


class ATSPTesterDGLExport(ATSPTesterDGL):
    """ATSP tester that also exports per-instance artifacts in a fixed text format.

    Folder layout (given args.model_path):
      base = dirname(model_path) if model_path endswith .pt else model_path
      ckpt = stem(model_path)
      - {base}/{ckpt}/test_atsp{size}/results.json
      - {base}/{ckpt}/trial_0/test_atsp{size}/instance<name>.txt
    """

    def _resolve_output_dirs(self) -> Dict[str, Path]:
        model_path = getattr(self.args, 'model_path', '') or ''
        base_dir = Path(model_path)
        if base_dir.suffix == '.pt':
            base_dir = base_dir.parent
        ckpt_stem = Path(model_path).stem if model_path else 'model'
        root = base_dir / ckpt_stem
        out_json = root / f"test_atsp{self.args.atsp_size}"
        out_txt = root / "trial_0" / f"test_atsp{self.args.atsp_size}"
        ensure_dir(out_json)
        ensure_dir(out_txt)
        return {"json": out_json, "txt": out_txt}

    def create_test_dataset(self):
        """Create test dataset with merged splits and single-load caching."""
        return ATSPDatasetDGL(
            data_dir=self.args.data_path,
            split="all",  # use train + valid + test together
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            undirected=self.args.undirected,
            device=self.device,
            load_once=True  # force full load once
        )

    def test_instance(self, model, test_dataset, instance_idx: int) -> Dict[str, Any]:
        sample = test_dataset[instance_idx]
        # Handle both lazy (returns (H, G)) and preloaded (returns H only) modes
        if isinstance(sample, tuple) and len(sample) == 2:
            H, G = sample
        else:
            H = sample
            # Load original NetworkX graph directly from dataset
            try:
                G = test_dataset._load_instance_graph(test_dataset.instances[instance_idx])
            except Exception:
                # Fallback to pickle load
                with open(Path(test_dataset.data_dir) / test_dataset.instances[instance_idx], 'rb') as f:
                    import pickle as _p
                    G = _p.load(f)
        n = self.args.atsp_size

        # Predict regrets
        x = H.ndata['weight']
        t0 = time.time()
        with torch.no_grad():
            y_pred = model(H, x)
        model_time = time.time() - t0

        regret_pred_vec = test_dataset.scalers.inverse_transform(
            y_pred.detach().cpu().flatten(), 'regret'
        )

        # Assign predicted regrets back to G
        edge_idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) in G.edges:
                    G.edges[(i, j)]['regret_pred'] = max(float(regret_pred_vec[edge_idx]), 0.0)
                edge_idx += 1

        # Initial tour and GLS
        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour, weight='weight')
        gls_start = time.time()
        time_limit = gls_start + self.args.time_limit
        best_tour, best_cost, search_progress, num_iterations = guided_local_search(
            G, init_tour, init_cost, time_limit,
            weight='weight', guides=['regret_pred'],
            perturbation_moves=self.args.perturbation_moves, first_improvement=False
        )
        gls_time = time.time() - gls_start

        # Optimal cost (from labels)
        opt_cost = optimal_cost(G, weight='weight')
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100

        # Export artifacts
        out_dirs = self._resolve_output_dirs()
        # Edge weight matrix
        edge_weight = np.array(
            [[G.edges[(i, j)]['weight'] if i != j else 0.0 for j in range(n)] for i in range(n)],
            dtype=np.float32
        )
        regret_mat = self._add_diag(n, H.ndata['regret'].detach().cpu().flatten()).numpy()
        regret_pred_mat = self._add_diag(n, y_pred.detach().cpu().flatten()).numpy()
        # Unscaled true regrets and in_solution mask
        regret_true_vec = test_dataset.scalers.inverse_transform(
            H.ndata['regret'].detach().cpu().flatten(), 'regret'
        )
        regret_true_mat = self._add_diag(n, regret_true_vec).numpy()
        in_solution_mat = self._add_diag(n, H.ndata['in_solution'].detach().cpu().flatten()).numpy()

        # Write standard three blocks first
        write_instance_artifacts(
            out_dir=out_dirs['txt'],
            instance_name=str(test_dataset.instances[instance_idx]),
            edge_weight=edge_weight,
            regret_mat=regret_mat,
            regret_pred_mat=regret_pred_mat,
            opt_cost=float(opt_cost),
            num_iterations=int(num_iterations),
            init_cost=float(init_cost),
            best_cost=float(best_cost),
        )
        # Append regret_true and in_solution blocks for validation
        fp = out_dirs['txt'] / f"instance{str(test_dataset.instances[instance_idx]).replace(os.sep, '_')}.txt"
        with open(fp, 'a') as f:
            f.write("\nregret_true:\n")
            np.savetxt(f, regret_true_mat, fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write("in_solution:\n")
            np.savetxt(f, in_solution_mat, fmt="%.0f", delimiter=" ")
            f.write("\n")

        # Cleanup
        del H, x, y_pred, regret_pred_vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'opt_cost': float(opt_cost),
            'init_cost': float(init_cost),
            'final_cost': float(best_cost),
            'init_gap': float(init_gap),
            'final_gap': float(final_gap),
            'num_iteration': int(num_iterations),
            'model_time': float(model_time),
            'gls_time': float(gls_time),
            'instance': test_dataset.instances[instance_idx],
        }

    @staticmethod
    def _add_diag(num_nodes: int, values_1d: torch.Tensor) -> torch.Tensor:
        """Place a length N*(N-1) vector into an NxN matrix with zero diagonal."""
        n = int(num_nodes)
        t2 = torch.zeros(n, n, dtype=torch.float32)
        cnt = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t2[i, j] = values_1d[cnt]
                cnt += 1
        return t2

    def run_test(self, model) -> Dict[str, Any]:
        test_dataset = self.create_test_dataset()
        results = self.test_all(model, test_dataset)

        out_dirs = self._resolve_output_dirs()
        with open(out_dirs['json'] / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results
