import networkx as nx
import torch
import dgl
from typing import Tuple
import pathlib


class LineGraphTransform:
    """Handles line graph transformations for complete directed graphs."""
    
    def __init__(
        self,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        half_st: bool = False,
        directed: bool = True,
    ):
        self.relation_types = relation_types
        self.half_st = half_st
        self.directed = directed
    
    def create_line_graph_template(self, n_nodes: int) -> dgl.DGLGraph:
        """Create optimized line graph template for complete graph with n nodes."""
        # Create complete directed graph
        g = nx.complete_graph(n_nodes, create_using=nx.DiGraph())
        return self._optimized_line_graph(g)
    
    def _optimized_line_graph(self, g: nx.DiGraph) -> dgl.DGLGraph:
        """Optimized line graph transformation."""
        n = g.number_of_nodes()
        m1 = n * (n - 1) * (n - 2) // 2
        m2 = n * (n - 1) // 2
        
        # Pre-allocate tensors based on relation types
        edge_tensors = {}
        
        if 'ss' in self.relation_types:
            edge_tensors['ss'] = torch.empty((m1, 2), dtype=torch.int64)
        if 'st' in self.relation_types:
            size = m1 if self.half_st else m1 * 2
            edge_tensors['st'] = torch.empty((size, 2), dtype=torch.int64)
        if 'tt' in self.relation_types:
            edge_tensors['tt'] = torch.empty((m1, 2), dtype=torch.int64)
        if 'pp' in self.relation_types:
            edge_tensors['pp'] = torch.empty((m2, 2), dtype=torch.int64)
        
        edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
        idx = 0
        idx2 = 0
        
        # Build edge relationships
        for x in range(n):
            for y in range(n - 1):
                if x != y:
                    for z in range(y + 1, n):
                        if x != z:
                            if 'ss' in self.relation_types:
                                edge_tensors['ss'][idx] = torch.tensor(
                                    [edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int64
                                )
                            if 'st' in self.relation_types:
                                if self.half_st:
                                    edge_tensors['st'][idx] = torch.tensor(
                                        [edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int64
                                    )
                                else:
                                    edge_tensors['st'][idx * 2] = torch.tensor(
                                        [edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int64
                                    )
                                    edge_tensors['st'][idx * 2 + 1] = torch.tensor(
                                        [edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int64
                                    )
                            if 'tt' in self.relation_types:
                                edge_tensors['tt'][idx] = torch.tensor(
                                    [edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int64
                                )
                            idx += 1
            
            if 'pp' in self.relation_types:
                for y in range(x + 1, n):
                    edge_tensors['pp'][idx2] = torch.tensor(
                        [edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int64
                    )
                    idx2 += 1
        
        # Create heterograph structure
        edge_types = {}
        node_type = 'node'
        
        for rel_type in self.relation_types:
            if rel_type in edge_tensors:
                edge_types[(node_type, rel_type, node_type)] = (
                    edge_tensors[rel_type][:, 0].to(torch.int64), 
                    edge_tensors[rel_type][:, 1].to(torch.int64)
                )
        
        num_nodes = n * (n - 1)
        g2 = dgl.heterograph(edge_types,num_nodes_dict={node_type: num_nodes})
        
        if not self.directed:
            g2 = dgl.add_reverse_edges(g2)
        
        # Store original edge mapping
        # g2.ndata['edge_mapping'] = torch.tensor(list(edge_id.keys()))
        
        return g2
    
    def save_template(self, n_nodes: int, save_path: pathlib.Path):
        """Create and save line graph template."""
        template = self.create_line_graph_template(n_nodes)
        dgl.save_graphs(str(save_path), [template])
        return template

import pathlib

def main():
    # Define parameters
    n_nodes = 4  # small number to debug
    relation_types = ("ss", "st", "tt", "pp")
    save_path = pathlib.Path("test_template.dgl")

    # Initialize transform
    transform = LineGraphTransform(
        relation_types=relation_types,
        half_st=False,
        directed=True
    )

    # Create line graph template
    template = transform.create_line_graph_template(n_nodes)
    print(f"Created template with {template.num_nodes()} nodes and {template.num_edges()} edges.")

    # Save template
    transform.save_template(n_nodes, save_path)
    print(f"Template saved to {save_path}")

    # # Check edge types and mapping
    # print("Edge types:", template.etypes)
    # print("Edge mapping:", template.ndata['edge_mapping'])

if __name__ == "__main__":
    main()