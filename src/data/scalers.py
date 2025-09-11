import torch
import pickle
from typing import List, Dict, Any
import pathlib

class FeatureScaler:
    """MinMax scaling for graph features using PyTorch (float32)."""
    
    def __init__(self):
        self.mins: Dict[str, torch.Tensor] = {}
        self.maxs: Dict[str, torch.Tensor] = {}
    
    def fit_from_dataset(self, dataset_dir: pathlib.Path, train_files: List[str], feature_names: List[str]):
        """Compute min and max for each feature from training dataset."""
        import networkx as nx
        import tqdm
        
        for feature in feature_names:
            all_vals = []
            for filename in tqdm.tqdm(train_files, desc=f"Fitting {feature}"):
                filepath = dataset_dir / filename
                with open(filepath, 'rb') as f:
                    G = pickle.load(f)
                for u, v in G.edges():
                    if feature in G.edges[u, v]:
                        all_vals.append(float(G.edges[u, v][feature]))
            if all_vals:
                vals = torch.tensor(all_vals, dtype=torch.float32)
                self.mins[feature] = vals.min()
                self.maxs[feature] = vals.max()
            else:
                self.mins[feature] = torch.tensor(0.0)
                self.maxs[feature] = torch.tensor(1.0)
    
    def transform(self, features: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Scale features to [0,1]."""
        if feature_name not in self.mins or feature_name not in self.maxs:
            raise ValueError(f"Feature {feature_name} not fitted")
        min_val = self.mins[feature_name]
        max_val = self.maxs[feature_name]
        return (features - min_val) / (max_val - min_val + 1e-8)
    
    def inverse_transform(self, features: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Inverse MinMax transform."""
        if feature_name not in self.mins or feature_name not in self.maxs:
            raise ValueError(f"Feature {feature_name} not fitted")
        min_val = self.mins[feature_name]
        max_val = self.maxs[feature_name]
        return features * (max_val - min_val + 1e-8) + min_val
    
    def save(self, filepath: pathlib.Path):
        """Save the min/max values."""
        torch.save({'mins': self.mins, 'maxs': self.maxs}, filepath)
    
    @classmethod
    def load(cls, filepath: pathlib.Path) -> 'MinMaxFeatureScaler':
        """Load saved min/max values."""
        data = torch.load(filepath)
        instance = cls()
        instance.mins = data['mins']
        instance.maxs = data['maxs']
        return instance
