import pathlib
import pickle
from typing import Tuple, List, Optional
import networkx as nx
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader

from .scalers import FeatureScaler
from .template_manager import TemplateManager


class ATSPDatasetDGL:
    """ATSP dataset optimized for DGL models."""
    
    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        atsp_size: int,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        device: str = "cpu",
        undirected=True,
        load_once=True,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.atsp_size = atsp_size
        self.device = device
        self.relation_types = sorted(relation_types)
        self.undirected = undirected
        self.load_once = load_once

        # Load instance filenames, template, and scalers
        self.instances = self._load_instance_list()
        # self.template = self._load_template()
        self.scalers = self._load_scalers()

        # Pre-compute edge count
        self.num_edges = atsp_size * (atsp_size - 1)

        # Preload only if required
        if self.load_once:
            self.template = self._load_template()
            self.graphs = [
                self._process_graph(self._load_instance_graph(f))
                for f in self.instances
            ]
        else:
            self.graphs = None  # lazy loading mode

    def _load_instance_list(self) -> List[str]:
        """Load instance filenames for this split."""
        split_file = self.data_dir / f"{self.split}.txt"
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _load_template(self) -> dgl.DGLGraph:
        template_path = TemplateManager.get_template_path(
            self.data_dir, self.atsp_size, self.relation_types
        )

        template_path = TemplateManager.get_template_path(
        self.data_dir, self.atsp_size, self.relation_types
        )

        graphs, _ = dgl.load_graphs(str(template_path))
        template = graphs[0]
        if self.undirected:
            template = dgl.add_reverse_edges(template)
        # Move to device
        template = template.to(self.device)

        return template
    
    def _load_scalers(self) -> FeatureScaler:
        """Load fitted scalers."""
        scalers_path = self.data_dir / 'scalers.pkl'
        return FeatureScaler.load(scalers_path)
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Get processed graph instance."""
        if self.load_once:
            return self.graphs[idx]
        else:
            # Load one at a time
            graph = self._load_instance_graph(self.instances[idx])
            H = self._process_graph(graph)
            import gc; gc.collect()

            return H, graph
    
    def _load_instance_graph(self, instance_file):
        with open(self.data_dir / instance_file, 'rb') as f:
            return pickle.load(f)

    def _process_graph(self, G: nx.DiGraph) -> dgl.DGLGraph:
        """Convert NetworkX graph to DGL with features."""
        # Extract features in consistent order
        weights, regrets, in_solution = self._extract_features(G)
        
        # Apply scaling
        scaled_weights = self.scalers.transform(weights, 'weight')
        scaled_regrets = self.scalers.transform(regrets, 'regret')
        
        # Create graph with features
        graph = self.template.clone() if self.load_once else self._load_template()
        
        graph.ndata['weight'] = torch.from_numpy(scaled_weights).unsqueeze(1).to(self.device).to(torch.float32)
        graph.ndata['regret'] = torch.from_numpy(scaled_regrets).unsqueeze(1).to(self.device).to(torch.float32)
        graph.ndata['in_solution'] = torch.from_numpy(in_solution).unsqueeze(1).to(self.device).to(torch.float32)
        
        # Add metadata
        if hasattr(G, 'graph'):
            graph.graph_attr = dict(G.graph)
        
        return graph
    
    def _extract_features(self, G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features in consistent edge order."""
        weights = torch.empty(self.num_edges, dtype=torch.float32)
        regrets = torch.empty(self.num_edges, dtype=torch.float32)
        in_solution = torch.empty(self.num_edges, dtype=torch.float32)
        
        edge_idx = 0
        for i in range(self.atsp_size):
            for j in range(self.atsp_size):
                if i == j:
                    continue
                
                edge = (i, j)
                weights[edge_idx] = G.edges[(idx_i, idx_j)]['weight']
                regrets[edge_idx] = G.edges[(idx_i, idx_j)]['regret']
                in_solution[edge_idx] = G.edges[(idx_i, idx_j)]['in_solution']
                
                edge_idx += 1
        
        return weights, regrets, in_solution
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = None, **kwargs):
        """Get DataLoader with DGL batching."""
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dgl.batch,
            **kwargs
        )
    
    def get_scaled_features(self, G: nx.DiGraph) -> dgl.DGLGraph:
        """Get scaled features for single graph (for testing)."""
        return self._process_graph(G)
