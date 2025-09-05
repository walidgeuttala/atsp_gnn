import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Any, List
import pathlib

class FeatureScaler:
    """Handles feature scaling for graph datasets."""
    
    def __init__(self, scaler_type: str = "minmax"):
        self.scaler_type = scaler_type
        self.scalers: Dict[str, Any] = {}
        self._create_scalers()
    
    def _create_scalers(self):
        """Initialize scalers based on type."""
        scaler_class = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler, 
            "robust": RobustScaler
        }.get(self.scaler_type, MinMaxScaler)
        
        self.scalers = {
            'weight': scaler_class(),
            'regret': scaler_class()
        }
    
    def fit_from_dataset(self, dataset_dir: pathlib.Path, train_files: List[str]):
        """Fit scalers on training data."""
        import networkx as nx
        import tqdm
        
        # Collect all features for fitting
        all_features = {key: [] for key in self.scalers.keys()}
        
        for filename in tqdm.tqdm(train_files, desc="Fitting scalers"):
            filepath = dataset_dir / filename
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
            for edge in G.edges():
                for feature_name in self.scalers.keys():
                    if feature_name in G.edges[edge]:
                        all_features[feature_name].append(G.edges[edge][feature_name])
        
        # Fit each scaler
        for feature_name, scaler in self.scalers.items():
            if all_features[feature_name]:
                feature_array = np.array(all_features[feature_name]).reshape(-1, 1)
                scaler.fit(feature_array)
    
    def transform(self, features: np.ndarray, feature_name: str) -> np.ndarray:
        """Transform features using fitted scaler."""
        if feature_name not in self.scalers:
            raise ValueError(f"No scaler found for feature: {feature_name}")
        return self.scalers[feature_name].transform(features.reshape(-1, 1)).flatten()
    
    def inverse_transform(self, features: np.ndarray, feature_name: str) -> np.ndarray:
        """Inverse transform features."""
        if feature_name not in self.scalers:
            raise ValueError(f"No scaler found for feature: {feature_name}")
        return self.scalers[feature_name].inverse_transform(features.reshape(-1, 1)).flatten()
    
    def save(self, filepath: pathlib.Path):
        """Save fitted scalers."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'scaler_type': self.scaler_type
            }, f)
    
    @classmethod
    def load(cls, filepath: pathlib.Path) -> 'FeatureScaler':
        """Load fitted scalers."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(scaler_type=data['scaler_type'])
        instance.scalers = data['scalers']
        return instance
