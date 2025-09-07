import pathlib
import pickle
from typing import Tuple, List, Optional
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected

from .scalers import FeatureScaler
from .template_manager import TemplateManager


class ATSPDatasetPyG:
    """ATSP dataset optimized for PyG models."""
    
    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        atsp_size: int,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        device: str = "cpu",
        undirected: bool = False
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.atsp_size = atsp_size
        self.device = device
        self.relation_types = relation_types
        self.undirected = undirected
        
        # Load components
        self.instances = self._load_instance_list()
        self.edge_index_template = self._create_edge_template()
        self.scalers = self._load_scalers()
        
        # Pre-compute dimensions
        self.num_edges = atsp_size * (atsp_size - 1)
    
    def _load_instance_list(self) -> List[str]:
        """Load instance filenames for this split."""
        split_file = self.data_dir / f"{self.split}.txt"
        with open(split_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _create_edge_template(self) -> torch.Tensor:
        """Create edge connectivity template from DGL template."""
        template_path = TemplateManager.get_template_path(
            self.data_dir, self.atsp_size, self.relation_types
        )
        
        import dgl
        graphs, _ = dgl.load_graphs(str(template_path))
        dgl_graph = graphs[0]
        
        # Extract edge connectivity
        src, dst = dgl_graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        
        if self.undirected:
            edge_index = to_undirected(edge_index)
        
        return edge_index.to(self.device)
    
    def _load_scalers(self) -> FeatureScaler:
        """Load fitted scalers."""
        scalers_path = self.data_dir / 'scalers.pkl'
        return FeatureScaler.load(scalers_path)
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Data:
        """Get processed PyG Data instance."""
        instance_path = self.data_dir / self.instances[idx]
        with open(instance_path, 'rb') as f:
            G = pickle.load(f)
        return self._process_graph(G)
    
    def _process_graph(self, G: nx.DiGraph) -> Data:
        """Convert NetworkX graph to PyG Data with features."""
        # Extract features
        weights, regrets, in_solution = self._extract_features(G)
        
        # Apply scaling
        scaled_weights = self.scalers.transform(weights, 'weight')
        scaled_regrets = self.scalers.transform(regrets, 'regret')
        
        # Create node features (each node represents an edge in original graph)
        x = torch.stack([
            torch.from_numpy(scaled_weights),
            torch.from_numpy(scaled_regrets),
            torch.from_numpy(in_solution)
        ], dim=1).to(self.device)
        
        # Create PyG Data
        data = Data(
            x=x,
            edge_index=self.edge_index_template,
            num_nodes=self.num_edges
        )
        
        # Add targets
        data.y = torch.from_numpy(scaled_regrets).unsqueeze(1).to(self.device)
        
        # Add metadata
        if hasattr(G, 'graph'):
            for key, value in G.graph.items():
                setattr(data, key, value)
        
        return data
    
    def _extract_features(self, G: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features in consistent edge order."""
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
        
        return weights, regrets, in_solution
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = None, **kwargs):
        """Get PyG DataLoader."""
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    
    def get_scaled_features(self, G: nx.DiGraph) -> Data:
        """Get scaled features for single graph (for testing)."""
        return self._process_graph(G)
