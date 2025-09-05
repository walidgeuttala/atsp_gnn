import argparse
import pathlib
import random
from typing import List, Tuple
from itertools import combinations
import dgl
import torch
from torch_geometric.data import Data
from .scalers import FeatureScaler
from .graph_transforms import LineGraphTransform


class DatasetPreprocessor:
    """Handles dataset preprocessing: splitting, scaling, template creation."""
    
    def __init__(self, dataset_dir: pathlib.Path):
        self.dataset_dir = dataset_dir
    
    def create_splits(
        self, 
        n_train: int, 
        n_val: int, 
        n_test: int,
        seed: int = 42
    ) -> Tuple[List[str], List[str], List[str]]:
        """Create train/val/test splits and save to files."""
        # Get all pickle files
        instances = [f.name for f in self.dataset_dir.glob('*.pkl') 
                    if f.name != 'scalers.pkl']
        
        if len(instances) < (n_train + n_val + n_test):
            raise ValueError(f"Not enough instances. Found {len(instances)}, need {n_train + n_val + n_test}")
        
        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(instances)
        
        # Create splits
        train_set = instances[:n_train]
        val_set = instances[n_train:n_train + n_val]
        test_set = instances[n_train + n_val:n_train + n_val + n_test]
        
        # Save split files
        splits = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
        
        for split_name, file_list in splits.items():
            with open(self.dataset_dir / f"{split_name}.txt", 'w') as f:
                for filename in file_list:
                    f.write(f"{filename}\n")
            print(f"{split_name}: {len(file_list)} instances")
        
        return train_set, val_set, test_set
    
    def fit_scalers(self, train_files: List[str], scaler_type: str = "minmax"):
        """Fit and save feature scalers on training data."""
        scaler = FeatureScaler(scaler_type=scaler_type)
        scaler.fit_from_dataset(self.dataset_dir, train_files)
        
        scalers_path = self.dataset_dir / 'scalers.pkl'
        scaler.save(scalers_path)
        print(f"Scalers saved to {scalers_path}")
        
        return scaler
    
    def create_templates(
        self,
        atsp_sizes: List[int],
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        half_st: bool = True,
        directed: bool = True,
        add_reverse_edges: bool = True,
        save_pyg: bool = True
    ):
        """Create and save line graph templates for different sizes and relation type combinations."""
        transform = LineGraphTransform(
            relation_types=relation_types,
            half_st=half_st,
            directed=directed,
            add_reverse_edges=add_reverse_edges
        )
        
        # Create base templates directory
        templates_dir = self.dataset_dir / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        # Define subdirectories for different combination sizes
        subdirs = {
            1: templates_dir / 'single',
            2: templates_dir / 'pairs',
            3: templates_dir / 'triplets',
            len(relation_types): templates_dir / 'all'
        }
        
        # Create subdirectories
        for subdir in subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Generate all combinations of relation types (1, 2, 3, and all)
        for size in atsp_sizes:
            for r in range(1, len(relation_types) + 1):
                for rel_combo in combinations(relation_types, r):
                    # Create a new transform instance with the current combination
                    combo_transform = LineGraphTransform(
                        relation_types=rel_combo,
                        half_st=half_st,
                        directed=directed,
                        add_reverse_edges=add_reverse_edges
                    )
                    
                    # Create filename with relation types
                    rel_str = "_".join(rel_combo)
                    subdir = subdirs[r] if r < len(relation_types) else subdirs[len(relation_types)]
                    dgl_template_path = subdir / f"template_{size}_{rel_str}.dgl"
                    pyg_template_path = subdir / f"template_{size}_{rel_str}.pt"
                    
                    # Save DGL template
                    try:
                        template = combo_transform.save_template(size, dgl_template_path)
                        print(f"DGL template for size {size}, relations {rel_str} saved to {dgl_template_path}")
                        print(f"  Nodes: {template.number_of_nodes()}")
                        print(f"  Edges: {template.number_of_edges()}")
                        print(f"  Relation types: {list(template.etypes)}")
                    except Exception as e:
                        print(f"Failed to save DGL template for size {size}, relations {rel_str}: {e}")
                        continue
                    
                    # Save PyG template (optional)
                    if save_pyg:
                        try:
                            # Convert DGL graph to PyG Data object
                            pyg_data = self._dgl_to_pyg(template)
                            torch.save(pyg_data, pyg_template_path)
                            print(f"PyG template for size {size}, relations {rel_str} saved to {pyg_template_path}")
                        except Exception as e:
                            print(f"Failed to save PyG template for size {size}, relations {rel_str}: {e}")
    
    def _dgl_to_pyg(self, dgl_graph: dgl.DGLGraph) -> 'Data':
        """Convert a DGL graph to a PyG Data object."""
        from torch_geometric.data import Data
        
        # Extract edge indices and edge types
        edge_index = torch.stack([dgl_graph.edges()[0], dgl_graph.edges()[1]], dim=0)
        
        # Extract node features if available
        x = dgl_graph.ndata.get('feat', None)
        if x is None:
            x = torch.ones((dgl_graph.number_of_nodes(), 1))  # Placeholder feature
        
        # Extract edge features if available
        edge_attr = dgl_graph.edata.get('feat', None)
        if edge_attr is None:
            edge_attr = torch.ones((dgl_graph.number_of_edges(), 1))  # Placeholder feature
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # Add edge types if the graph is heterogeneous
        if dgl_graph.is_homogeneous:
            data.edge_type = torch.zeros(dgl_graph.number_of_edges(), dtype=torch.long)
        else:
            edge_types = []
            for etype in dgl_graph.etypes:
                mask = dgl_graph.edges(etype=etype)[0]  # Get edges for this etype
                edge_types.extend([dgl_graph.get_etype_id(etype)] * mask.size(0))
            data.edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return data


def main():
    """CLI for dataset preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess ATSP dataset.')
    parser.add_argument('dataset_dir', type=pathlib.Path, help='Dataset directory')
    parser.add_argument('--n_train', type=int, default=800, help='Training instances')
    parser.add_argument('--n_val', type=int, default=100, help='Validation instances')
    parser.add_argument('--n_test', type=int, default=100, help='Test instances')
    parser.add_argument('--atsp_size', type=int, default=10, help='ATSP problem size')
    parser.add_argument('--scaler_type', type=str, default='minmax', 
                       choices=['minmax', 'standard', 'robust'])
    parser.add_argument('--relation_types', nargs='+', default=['ss', 'st', 'tt', 'pp'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_pyg', action='store_true', help='Save templates in PyG format')
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(args.dataset_dir)
    
    # Create splits
    train_files, val_files, test_files = preprocessor.create_splits(
        args.n_train, args.n_val, args.n_test, seed=args.seed
    )
    
    # Fit scalers
    preprocessor.fit_scalers(train_files, scaler_type=args.scaler_type)
    
    # Create templates
    preprocessor.create_templates(
        atsp_sizes=[args.atsp_size],
        relation_types=tuple(args.relation_types),
        save_pyg=args.save_pyg
    )
    
    print("Preprocessing complete!")


if __name__ == '__main__':
    main()