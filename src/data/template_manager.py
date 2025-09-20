import pathlib
from typing import Tuple, Optional
import dgl
import torch
from torch_geometric.data import Data

class TemplateManager:
    """Manages template path resolution based on relation types."""
    
    @staticmethod
    def get_template_path(
        dataset_dir: pathlib.Path,
        atsp_size: int,
        relation_types: Tuple[str, ...]
    ) -> pathlib.Path:
        """Get template path based on relation types."""
        templates_dir = dataset_dir / "templates"
        type_str = "_".join(relation_types)

        num_types = len(relation_types)
        if num_types == 1:
            subdir = "single"
        elif num_types == 2:
            subdir = "pairs"
        elif num_types == 3:
            subdir = "triplets"
        elif num_types == 4:
            subdir = "all"
        else:
            subdir = ""

        candidates = []
        if subdir:
            candidates.append(templates_dir / subdir / f"template_{atsp_size}_{type_str}.dgl")

        candidates.append(templates_dir / f"template_{atsp_size}.dgl")
        candidates.append(templates_dir / f"full_template_{atsp_size}.dgl")

        for path in candidates:
            if path.exists():
                return path

        return candidates[0]
