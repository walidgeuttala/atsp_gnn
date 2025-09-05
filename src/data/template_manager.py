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
        
        # Sort relation types for consistent naming
        sorted_types = tuple(sorted(relation_types))
        type_str = "_".join(sorted_types)
        
        # Determine subdirectory
        num_types = len(sorted_types)
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
        
        # Try specific path first, then fallback to base
        if subdir:
            template_path = templates_dir / subdir / f"template_{atsp_size}_{type_str}.dgl"
            if template_path.exists():
                return template_path
        
        # Fallback to base template
        return templates_dir / f"template_{atsp_size}.dgl"
