import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from .test_dgl_export import ensure_dir, write_instance_artifacts
from .test_dgl_large import ATSPTesterDGLLarge
from src.utils.atsp_utils import tour_cost, optimal_cost, is_valid_tour
from src.utils.algorithms import guided_local_search, nearest_neighbor
from data.dataset_dgl_large import ATSPDatasetDGL


class ATSPTesterDGLLargeExport(ATSPTesterDGLLarge):
    """Large-graph tester that also exports per-instance artifacts."""

    def _resolve_output_dirs(self) -> Dict[str, Path]:
        model_path = getattr(self.args, 'model_path', '') or ''
        base_dir = Path(getattr(self.args, 'results_dir', '') or model_path)
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
        return ATSPDatasetDGL(
            data_dir=self.args.data_path,
            split='test',
            atsp_size=self.args.atsp_size,
            relation_types=tuple(self.args.relation_types),
            undirected=self.args.undirected,
            device=self.device,
            load_once=False,
            sub_size=getattr(self.args, 'sub_size', None),
            template_dir=getattr(self.args, 'template_path', None),
        )

    def test_instance(self, model, test_dataset, instance_idx: int) -> Dict[str, Any]:
        instance_name = test_dataset.instances[instance_idx]
        G = test_dataset._load_instance_graph(instance_name)
        n = self.args.atsp_size

        # Prepare full-graph features for ground-truth exports
        H_full = test_dataset.get_scaled_features(G)

        start = time.time()
        self._predict_regrets_large(model, test_dataset, G)
        model_time = time.time() - start

        init_tour = nearest_neighbor(G, start=0, weight='regret_pred')
        if not is_valid_tour(G, init_tour):
            raise RuntimeError(f"Init tour invalid (len={len(init_tour)}) for instance {instance_name}")
        init_cost = tour_cost(G, init_tour)

        gls_start = time.time()
        time_limit = gls_start + self.args.time_limit
        best_tour, final_cost, _, num_iterations = guided_local_search(
            G,
            init_tour,
            init_cost,
            time_limit,
            weight='weight',
            guides=['regret_pred'],
            perturbation_moves=self.args.perturbation_moves,
            first_improvement=False,
        )
        gls_time = time.time() - gls_start

        opt_cost = optimal_cost(G, weight='weight')
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (final_cost / opt_cost - 1) * 100

        # Export artifacts
        out_dirs = self._resolve_output_dirs()
        edge_weight = np.array(
            [[G.edges[(i, j)]['weight'] if i != j else 0.0 for j in range(n)] for i in range(n)],
            dtype=np.float32,
        )

        regret_true_vec = test_dataset.scalers.inverse_transform(
            H_full.ndata['regret'].detach().cpu().flatten(), 'regret'
        )
        regret_true_tensor = torch.as_tensor(np.asarray(regret_true_vec, dtype=np.float32))
        regret_true_mat = self._add_diag(n, regret_true_tensor).detach().cpu().numpy()

        regret_pred_vec = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                regret_pred_vec.append(G.edges[(i, j)].get('regret_pred', 0.0))
        regret_pred_tensor = torch.as_tensor(np.asarray(regret_pred_vec, dtype=np.float32))
        regret_pred_mat = self._add_diag(n, regret_pred_tensor).detach().cpu().numpy()

        write_instance_artifacts(
            out_dir=out_dirs['txt'],
            instance_name=str(instance_name),
            edge_weight=edge_weight,
            regret_mat=regret_true_mat,
            regret_pred_mat=regret_pred_mat,
            opt_cost=float(opt_cost),
            num_iterations=int(num_iterations),
            init_cost=float(init_cost),
            best_cost=float(final_cost),
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'opt_cost': float(opt_cost),
            'init_cost': float(init_cost),
            'final_cost': float(final_cost),
            'init_gap': float(init_gap),
            'final_gap': float(final_gap),
            'num_iteration': int(num_iterations),
            'model_time': float(model_time),
            'gls_time': float(gls_time),
            'instance': instance_name,
        }

    @staticmethod
    def _add_diag(num_nodes: int, values_1d: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        idx = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                t[i, j] = values_1d[idx]
                idx += 1
        return t

    def run_test(self, model) -> Dict[str, Any]:
        test_dataset = self.create_test_dataset()
        results = self.test_all(model, test_dataset)

        out_dirs = self._resolve_output_dirs()
        with open(out_dirs['json'] / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results
