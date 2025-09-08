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
        directed: bool = False,
        add_reverse_edges: bool = True
    ):
        self.relation_types = relation_types
        self.half_st = half_st
        self.directed = directed
        self.add_reverse_edges = add_reverse_edges
    
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
            edge_tensors['ss'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'st' in self.relation_types:
            size = m1 if self.half_st else m1 * 2
            edge_tensors['st'] = torch.empty((size, 2), dtype=torch.int32)
        if 'tt' in self.relation_types:
            edge_tensors['tt'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'pp' in self.relation_types:
            edge_tensors['pp'] = torch.empty((m2, 2), dtype=torch.int32)
        
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
                                    [edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32
                                )
                            if 'st' in self.relation_types:
                                if self.half_st:
                                    edge_tensors['st'][idx] = torch.tensor(
                                        [edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32
                                    )
                                else:
                                    edge_tensors['st'][idx * 2] = torch.tensor(
                                        [edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32
                                    )
                                    edge_tensors['st'][idx * 2 + 1] = torch.tensor(
                                        [edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32
                                    )
                            if 'tt' in self.relation_types:
                                edge_tensors['tt'][idx] = torch.tensor(
                                    [edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32
                                )
                            idx += 1
            
            if 'pp' in self.relation_types:
                for y in range(x + 1, n):
                    edge_tensors['pp'][idx2] = torch.tensor(
                        [edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32
                    )
                    idx2 += 1
        
        # Create heterograph structure
        edge_types = {}
        node_type = 'node'
        
        for rel_type in self.relation_types:
            if rel_type in edge_tensors:
                edge_types[(node_type, rel_type, node_type)] = (
                    edge_tensors[rel_type][:, 0], 
                    edge_tensors[rel_type][:, 1]
                )
        
        g2 = dgl.heterograph(edge_types)
        
        if self.add_reverse_edges:
            g2 = dgl.add_reverse_edges(g2)
        
        # Store original edge mapping
        g2.ndata['edge_mapping'] = torch.tensor(list(edge_id.keys()))
        
        return g2
    
    def save_template(self, n_nodes: int, save_path: pathlib.Path):
        """Create and save line graph template."""
        template = self.create_line_graph_template(n_nodes)
        dgl.save_graphs(str(save_path), [template])
        return template