import pathlib
import pickle
from typing import Tuple, List
import networkx as nx
import numpy as np
import torch
import dgl


class ATSPTorchDataset:
    """
    Optimized ATSP dataset for PyTorch/DGL models.
    Uses pre-computed line graph templates and fitted scalers.
    """
    
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        split: str,  # 'train', 'val', 'test'
        atsp_size: int,
        template_path: pathlib.Path,
        scalers_path: pathlib.Path,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        device: str = "cpu"
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.atsp_size = atsp_size
        self.device = device
        self.relation_types = relation_types
        
        # Load instance list
        split_file = dataset_dir / f"{split}.txt"
        self.instances = self._load_instance_list(split_file)
        
        # Load template and scalers
        self.template = self._load_template(template_path)
        self.scalers = self._load_scalers(scalers_path)
        
        # Pre-compute dimensions
        self.num_edges = atsp_size * (atsp_size - 1)
    
    def _load_instance_list(self, split_file: pathlib.Path) -> List[str]:
        """Load list of instance filenames for this split."""
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _load_template(self, template_path: pathlib.Path) -> dgl.DGLGraph:
        """Load pre-computed line graph template."""
        graphs, _ = dgl.load_graphs(str(template_path))
        template = graphs[0].to(self.device)
        
        # Remove any existing features
        for key in list(template.ndata.keys()):
            if key != 'edge_mapping':
                del template.ndata[key]
        
        return template
    
    def _load_scalers(self, scalers_path: pathlib.Path):
        """Load fitted feature scalers."""
        from .scalers import FeatureScaler
        return FeatureScaler.load(scalers_path)
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Get a single processed graph instance."""
        # Load NetworkX graph
        instance_path = self.dataset_dir / self.instances[idx]
        G = nx.read_gpickle(instance_path)
        
        # Convert to feature tensors
        return self._process_graph(G)
    
    def _process_graph(self, G: nx.DiGraph) -> dgl.DGLGraph:
        """Convert NetworkX graph to DGL graph with features."""
        # Extract edge features in consistent order
        weights = np.zeros(self.num_edges, dtype=np.float32)
        regrets = np.zeros(self.num_edges, dtype=np.float32)
        in_solution = np.zeros(self.num_edges, dtype=np.float32)
        
        edge_idx = 0
        for i in range(self.atsp_size):
            for j in range(self.atsp_size):
                if i == j:
                    continue
                
                edge = (i, j)
                if edge in G.edges:
                    weights[edge_idx] = G.edges[edge].get('weight', 0.0)
                    regrets[edge_idx] = G.edges[edge].get('regret', 0.0)
                    in_solution[edge_idx] = 1.0 if G.edges[edge].get('in_solution', False) else 0.0
                
                edge_idx += 1
        
        # Apply scaling
        scaled_weights = self.scalers.transform(weights, 'weight')
        scaled_regrets = self.scalers.transform(regrets, 'regret')
        
        # Clone template and add features
        graph = self.template.clone()
        graph.ndata['weight'] = torch.from_numpy(scaled_weights).unsqueeze(1).to(self.device)
        graph.ndata['regret'] = torch.from_numpy(scaled_regrets).unsqueeze(1).to(self.device)
        graph.ndata['in_solution'] = torch.from_numpy(in_solution).unsqueeze(1).to(self.device)
        
        # Add graph-level info if needed
        if 'tour' in G.graph:
            graph.graph_attr = {
                'tour': G.graph['tour'],
                'cost': G.graph['cost']
            }
        
        return graph
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = None, **kwargs):
        """Get PyTorch DataLoader for this dataset."""
        from torch.utils.data import DataLoader
        
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    def _collate_fn(self, batch: List[dgl.DGLGraph]) -> dgl.DGLGraph:
        """Collate function for batching graphs."""
        return dgl.batch(batch)