import networkx as nx
import torch
import dgl
from typing import Tuple
import pathlib
import matplotlib.pyplot as plt


class LineGraphTransform:
    """Handles line graph transformations for complete directed graphs."""
    
    def __init__(
        self,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        directed: bool = False,
    ):
        self.relation_types = sorted(relation_types)
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
            edge_tensors['ss'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'st' in self.relation_types:
            edge_tensors['st'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'tt' in self.relation_types:
            edge_tensors['tt'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'pp' in self.relation_types:
            edge_tensors['pp'] = torch.empty((m2, 2), dtype=torch.int32)
        
        edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
        idx = 0
        idx2 = 0
        
        # Build edge relationships
        for x in range(n):
            for y in range(x+1, n):
                for z in range(y + 1, n):
                    if 'ss' in self.relation_types:
                        edge_tensors['ss'][idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                    if 'st' in self.relation_types:
                        edge_tensors['st'][idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                    if 'tt' in self.relation_types:
                        edge_tensors['tt'][idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                    idx += 1
            
            if 'pp' in self.relation_types:
                for y in range(x + 1, n):
                    edge_tensors['pp'][idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
                    idx2 += 1
        
        # Create heterograph structure
        edge_types = {}
        node_type = 'node'
        
        for rel_type in self.relation_types:
            if rel_type in edge_tensors:
                edge_types[(node_type, rel_type, node_type)] = (
                torch.as_tensor(edge_tensors[rel_type][:, 0], dtype=torch.int32),
                torch.as_tensor(edge_tensors[rel_type][:, 1], dtype=torch.int32)
            )
                    
        num_nodes = n * (n - 1)
        g2 = dgl.heterograph(edge_types,num_nodes_dict={node_type: num_nodes})
        
        if not self.directed:
            g2 = dgl.add_reverse_edges(g2)
        
        # Store original edge mapping
        g2.ndata['edge_mapping'] = torch.tensor(list(edge_id.keys()))
        
        return g2
    
    def plot_full_graph(
        self, 
        full_graph: dgl.DGLGraph, 
        output_dir: str | pathlib.Path, 
        relation_types: Tuple[str, ...] = None
    ):
        """
        Plot and save the full heterograph and its relation-type subgraphs.

        Args:
            full_graph: DGL heterograph (full template).
            output_dir: path to save PNG plots.
            relation_types: tuple of relation types to plot (defaults to full_graph.etypes).
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if relation_types is None:
            relation_types = tuple(full_graph.etypes)

        # Convert heterograph to networkx with relation labels
        g2 = nx.Graph()
        for rel in relation_types:
            src, dst = full_graph.edges(etype=rel)
            edges = list(zip(src.tolist(), dst.tolist()))
            g2.add_edges_from([(u, v, {"relation": rel}) for u, v in edges])

        # Define edge colors
        edge_colors = {
            "ss": "red",
            "st": "green",
            "tt": "blue",
            "pp": "orange"
        }

        # Shared layout (consistent positioning)
        pos = nx.spring_layout(g2, seed=42)

        # Plot per-relation graphs
        for relation in relation_types:
            plt.figure(figsize=(8, 8))
            nx.draw_networkx_nodes(g2, pos, node_color="lightgreen", node_size=500)

            edges = [(u, v) for u, v, d in g2.edges(data=True) if d["relation"] == relation]
            nx.draw_networkx_edges(
                g2, pos, edgelist=edges,
                edge_color=edge_colors.get(relation, "black"),
                label=relation
            )
            nx.draw_networkx_labels(g2, pos, font_size=10)

            plt.title(f"Line Graph - Relation: {relation}")
            plt.legend()
            plt.savefig(output_dir / f"line_graph_{relation}.png")
            plt.close()

        # Plot all relations together
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(g2, pos, node_color="lightgreen", node_size=500)
        for relation in relation_types:
            edges = [(u, v) for u, v, d in g2.edges(data=True) if d["relation"] == relation]
            nx.draw_networkx_edges(
                g2, pos, edgelist=edges,
                edge_color=edge_colors.get(relation, "black"),
                label=relation
            )
        nx.draw_networkx_labels(g2, pos, font_size=10)
        plt.title("Line Graph - All Relations")
        plt.legend()
        plt.savefig(output_dir / "line_graph_all.png")
        plt.close()

    def extract_subgraph(self, full_template: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Extract a subgraph from a full line graph template based on self.relation_types.
        Assumes the full_template already has duplicates removed.
        """
        node_type = 'node'
        edge_types = {}

        for etype in self.relation_types:
            if etype in full_template.etypes:
                src, dst = full_template.edges(etype=etype)
                edge_types[(node_type, etype, node_type)] = (src, dst)

        num_nodes = full_template.num_nodes(node_type)
        subgraph = dgl.heterograph(edge_types, num_nodes_dict={node_type: num_nodes})

        # Copy node data if exists
        for key, val in full_template.ndata.items():
            subgraph.ndata[key] = val.clone()

        return subgraph



    def save_template(self, n_nodes: int, save_path: pathlib.Path):
        """Create and save line graph template."""
        template = self.create_line_graph_template(n_nodes)
        dgl.save_graphs(str(save_path), [template])
        return template



if __name__ == '__main__':
    transform = LineGraphTransform()
    full_g = transform.create_line_graph_template(n_nodes=3)  # example
    transform.plot_full_graph(full_g, output_dir="../jobs/runs/plots")
