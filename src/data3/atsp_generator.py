import argparse
import multiprocessing as mp
import pathlib
import uuid
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from . import utils

class ATSPDatasetGenerator:
    def __init__(self, n_nodes, n_instances, output_dir, weight_min=100, weight_max=1000):
        """Initialize the ATSP dataset generator with parameters."""
        self.n_nodes = n_nodes
        self.n_instances = n_instances
        self.output_dir = output_dir
        self.weight_min = weight_min
        self.weight_max = weight_max

    def _generate_instance(self):
        """Generate a single ATSP instance with weights satisfying triangle inequality."""
        # Random adjacency matrix (no self-loops)
        weights = np.random.randint(self.weight_min, self.weight_max + 1, 
                                    size=(self.n_nodes, self.n_nodes)).astype(float)
        np.fill_diagonal(weights, 0.0)

        # Enforce triangle inequality via Floyd–Warshall
        dist = floyd_warshall(weights, directed=True)

        # Build NetworkX graph
        G = nx.DiGraph()
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        # Solve the ATSP instance (your gnngls solver)
        opt_solution = gnngls.optimal_tour(G)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        return G

    def get_instance_generator(self):
        """Yield a generator for ATSP instances."""
        for _ in range(self.n_instances):
            yield self._generate_instance()

    def save_dataset(self):
        """Generate and save ATSP instances to the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with mp.Pool() as pool:
            try:
                for G in pool.imap_unordered(utils.prepare_instance, self.get_instance_generator()):
                    output_file = self.output_dir / f'{uuid.uuid4().hex}.pkl'
                    nx.write_gpickle(G, output_file)
            except Exception as e:
                print(f"Error during processing: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an ATSP dataset.')
    parser.add_argument('n_samples', type=int, help='Number of instances to generate')
    parser.add_argument('n_nodes', type=int, help='Number of nodes per instance')
    parser.add_argument('dir', type=pathlib.Path, help='Output directory')
    parser.add_argument('--weight_min', type=int, default=100, help='Minimum weight for edges')
    parser.add_argument('--weight_max', type=int, default=1000, help='Maximum weight for edges')
    args = parser.parse_args()

    generator = ATSPDatasetGenerator(
        n_nodes=args.n_nodes,
        n_instances=args.n_samples,
        output_dir=args.dir,
        weight_min=args.weight_min,
        weight_max=args.weight_max
    )
    generator.save_dataset()
