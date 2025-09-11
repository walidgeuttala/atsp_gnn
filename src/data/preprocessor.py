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
        """Create or load train/val/test splits from existing files if available."""
        def read_split(file_path: pathlib.Path):
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            return None

        # Check for existing split files
        train_file = self.dataset_dir / "train.txt"
        val_file = self.dataset_dir / "val.txt"
        test_file = self.dataset_dir / "test.txt"

        train_set = read_split(train_file)
        val_set = read_split(val_file)
        test_set = read_split(test_file)

        # If all exist, just return them
        if train_set is not None and val_set is not None and test_set is not None:
            print("Found existing split files. Using them directly.")
            return train_set, val_set, test_set

        # Otherwise create splits as before
        instances = [f.name for f in self.dataset_dir.glob('*.pkl') if f.name != 'scalers.pkl']
        
        if len(instances) < (n_train + n_val + n_test):
            raise ValueError(f"Not enough instances. Found {len(instances)}, need {n_train + n_val + n_test}")
        
        random.seed(seed)
        random.shuffle(instances)
        
        train_set = instances[:n_train]
        val_set = instances[n_train:n_train + n_val]
        test_set = instances[n_train + n_val:n_train + n_val + n_test]
        
        splits = {'train': train_set, 'val': val_set, 'test': test_set}
        
        for split_name, file_list in splits.items():
            with open(self.dataset_dir / f"{split_name}.txt", 'w') as f:
                for filename in file_list:
                    f.write(f"{filename}\n")
            print(f"{split_name}: {len(file_list)} instances")
        
        return train_set, val_set, test_set
    
    def fit_scalers(self, files: List[str], scaler_type: str = "minmax"):
        """Fit and save feature scalers on provided data."""
        scaler = FeatureScaler(scaler_type=scaler_type)
        scaler.fit_from_dataset(self.dataset_dir, files)
        
        scalers_path = self.dataset_dir / 'scalers.pkl'
        scaler.save(scalers_path)
        print(f"Scalers saved to {scalers_path}")
        return scaler
    
    def load_scalers(self, scaler_path: pathlib.Path):
        """Load existing feature scalers from file."""
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        scaler = FeatureScaler.load(scaler_path)
        print(f"Loaded scalers from {scaler_path}")
        return scaler

    def get_or_create_scalers(
        self, train_files: List[str], val_files: List[str], test_files: List[str], scaler_type: str
    ):
        """Load existing scalers or fit new scalers based on priority: train > test > val."""
        scaler_path = self.dataset_dir / 'scalers.pkl'
        if scaler_path.exists():
            return self.load_scalers(scaler_path)

        if len(train_files) > 0:
            print("Fitting scalers on training set.")
            return self.fit_scalers(train_files, scaler_type)
        elif len(test_files) > 0:
            print("Fitting scalers on test set (no training set available).")
            return self.fit_scalers(test_files, scaler_type)
        elif len(val_files) > 0:
            print("Fitting scalers on validation set (no training or test sets available).")
            return self.fit_scalers(val_files, scaler_type)
        else:
            raise ValueError("No data available to fit scalers.")
    
    def create_templates(
        self,
        atsp_sizes: List[int],
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        half_st: bool = False,
        directed: bool = False,
        save_pyg: bool = True
    ):
        """Create and save line graph templates for different sizes and relation type combinations."""
        templates_dir = self.dataset_dir / 'templates'
        templates_dir.mkdir(exist_ok=True)
        
        subdirs = {
            1: templates_dir / 'single',
            2: templates_dir / 'pairs',
            3: templates_dir / 'triplets',
            len(relation_types): templates_dir / 'all'
        }
        for subdir in subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        for size in atsp_sizes:
            # Build one full template for all relation types
            full_transform = LineGraphTransform(
                relation_types=relation_types,
                half_st=half_st,
                directed=directed
            )
            full_template = full_transform.save_template(size, templates_dir / f"full_template_{size}.dgl")
            full_template = self.remove_duplicate_edges(full_template)
            for r in range(1, len(relation_types) + 1):
                for rel_combo in combinations(relation_types, r):
                    rel_combo = sorted(rel_combo)
                    rel_str = "_".join(rel_combo)
                    subdir = subdirs[r] if r < len(relation_types) else subdirs[len(relation_types)]
                    dgl_template_path = subdir / f"template_{size}_{rel_str}.dgl"
                    pyg_template_path = subdir / f"template_{size}_{rel_str}.pt"
                    
                    try:
                        template = combo_transform.save_template(size, dgl_template_path)
                        print(f"DGL template {size}, {rel_str} saved to {dgl_template_path}")
                    except Exception as e:
                        print(f"Failed DGL template {size}, {rel_str}: {e}")
                        continue
                    
                    if save_pyg:
                        try:
                            pyg_data = self._dgl_to_pyg(template)
                            torch.save(pyg_data, pyg_template_path)
                            print(f"PyG template {size}, {rel_str} saved to {pyg_template_path}")
                        except Exception as e:
                            print(f"Failed PyG template {size}, {rel_str}: {e}")
    

    def remove_duplicate_edges(full_template: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Remove duplicate edges from a DGL heterograph.
        Works for each relation type independently.
        """
        node_type = full_template.ntypes[0]  # assuming single node type
        new_edge_dict = {}

        for etype in full_template.etypes:
            src, dst = full_template.edges(etype=etype)
            # Stack src and dst, then use torch.unique to remove duplicates
            edges = torch.stack([src, dst], dim=1)
            edges = torch.unique(edges, dim=0)
            new_edge_dict[(node_type, etype, node_type)] = (edges[:, 0], edges[:, 1])

        num_nodes = full_template.num_nodes(node_type)
        clean_graph = dgl.heterograph(new_edge_dict, num_nodes_dict={node_type: num_nodes})

        # Copy node data
        for key, val in full_template.ndata.items():
            clean_graph.ndata[key] = val.clone()

        return clean_graph

    def _dgl_to_pyg(self, dgl_graph: dgl.DGLGraph):
        """Convert a DGL graph to a PyG Data/HeteroData object."""
        from torch_geometric.utils import from_dgl
        return from_dgl(dgl_graph)


def main():
    parser = argparse.ArgumentParser(description='Preprocess ATSP dataset.')
    parser.add_argument('dataset_dir', type=pathlib.Path, help='Dataset directory')
    parser.add_argument('--n_train', type=int, default=800)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--atsp_size', type=int, default=10)
    parser.add_argument('--scaler_type', type=str, default='minmax', choices=['minmax', 'standard', 'robust'])
    parser.add_argument('--relation_types', nargs='+', default=['ss', 'st', 'tt', 'pp'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_pyg', action='store_true')

    args = parser.parse_args()
    preprocessor = DatasetPreprocessor(args.dataset_dir)

    train_files, val_files, test_files = preprocessor.create_splits(
        args.n_train, args.n_val, args.n_test, seed=args.seed
    )

    # Load existing scalers or fit new scalers with priority
    preprocessor.get_or_create_scalers(train_files, val_files, test_files, args.scaler_type)

    # Create templates
    preprocessor.create_templates(
        atsp_sizes=[args.atsp_size],
        relation_types=tuple(args.relation_types),
        save_pyg=args.save_pyg
    )

    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
