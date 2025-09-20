import networkx as nx
import torch
import dgl
from typing import Tuple
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import defaultdict

class LineGraphTransform:
    """Handles line graph transformations for complete directed graphs."""
    
    def __init__(
        self,
        relation_types: Tuple[str, ...] = ("ss", "st", "tt", "pp"),
        directed: bool = False,
        half_st = False
    ):
        self.relation_types = sorted(relation_types)
        self.directed = directed
        self.half_st = half_st

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
            edge_tensors['st'] = torch.empty((m1 if self.half_st else self.m1*2, 2), dtype=torch.int32)
        if 'tt' in self.relation_types:
            edge_tensors['tt'] = torch.empty((m1, 2), dtype=torch.int32)
        if 'pp' in self.relation_types:
            edge_tensors['pp'] = torch.empty((m2, 2), dtype=torch.int32)
        
        edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
        idx = 0
        idx2 = 0
        
        # Build edge relationships
        for x in range(n):
            for y in range(n-1):
                if x != y:
                    for z in range(y + 1, n):
                        if x != z:
                            if 'ss' in self.relation_types:
                                edge_tensors['ss'][idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                            if 'st' in self.relation_types:
                                if self.half_st:
                                    edge_tensors['st'][idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                                else:
                                    edge_tensors['st'][idx*2] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                                    edge_tensors['st'][idx*2+1] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32)
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
        
        g2 = dgl.add_reverse_edges(g2)
        
        # Store original edge mapping
        g2.ndata['edge_mapping'] = torch.tensor(list(edge_id.keys()), dtype=torch.float32)
        
        return g2
    
    def plot_full_graph(self, full_graph, output_dir, relation_types=None):
        """
        Plot and save the original graph, per-relation line graphs, and combined line graph.

        Args:
            full_graph: DGL heterograph (full template).
            output_dir: Directory to save PNG plots.
            relation_types: Relation types to plot. Defaults to full_graph.etypes.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if relation_types is None:
            relation_types = tuple(full_graph.etypes)

        # Convert heterograph to networkx
        g2 = nx.Graph()
        for rel in relation_types:
            src, dst = full_graph.edges(etype=rel)
            edges = list(zip(src.tolist(), dst.tolist()))
            g2.add_edges_from([(u, v, {"relation": rel}) for u, v in edges])

        # Create original graph from edge_mapping if available, otherwise reconstruct
        if 'edge_mapping' in full_graph.ndata:
            # Use the stored edge mapping to create the original graph
            edge_mapping = full_graph.ndata['edge_mapping'].numpy()
            original_edges = [(int(edge_mapping[i, 0]), int(edge_mapping[i, 1])) 
                            for i in range(len(edge_mapping))]
            edge_id = {edge: i for i, edge in enumerate(original_edges)}
            g = nx.Graph()
            g.add_edges_from(original_edges)
        else:
            # Fallback: create edge_id consistent with g.edges()
            g = nx.Graph()
            unique_edges = set()
            for rel in relation_types:
                src, dst = full_graph.edges(etype=rel)
                edges = list(zip(src.tolist(), dst.tolist()))
                unique_edges.update(edges)
            
            g.add_edges_from(unique_edges)
            # Create edge_id mapping that matches g.edges() order
            edge_id = {edge: i for i, edge in enumerate(g.edges())}

        # Shared edge colors
        edge_colors = {
            'ss': 'red',
            'st': 'green',
            'tt': 'blue',
            'pp': 'orange'
        }

        # Helper to ensure edge_color is always a list (avoids np.alltrue issue)
        def safe_color(color):
            return [color] if isinstance(color, str) else color

        # 1. Original graph with edge IDs
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, with_labels=False, node_color='skyblue', node_size=500, edge_color='gray')

        # Edge labels with slight random offsets to prevent overlap
        edge_labels = {edge: str(edge_id[edge]) for edge in g.edges()}
        for edge in g.edges():
            x_offset = np.random.uniform(-0.1, 0.1)
            y_offset = np.random.uniform(-0.1, 0.1)
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2 + x_offset
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2 + y_offset
            plt.text(x, y, edge_labels[edge], fontsize=12, ha='center')

        plt.title("Original Graph")
        plt.savefig(output_dir / "original_graph.png")
        plt.close()

        # 2. Per-relation line graphs
        pos2 = nx.spring_layout(g2, seed=42)
        for relation in relation_types:
            plt.figure(figsize=(8, 8))
            nx.draw_networkx_nodes(g2, pos2, node_color='lightgreen', node_size=500)

            edges = [(u, v) for u, v, d in g2.edges(data=True) if d['relation'] == relation]
            nx.draw_networkx_edges(g2, pos2, edgelist=edges,
                                edge_color=safe_color(edge_colors.get(relation, 'black')),
                                label=relation)

            nx.draw_networkx_labels(g2, pos2, font_size=15)
            plt.title(f"Line Graph - Relation Type: {relation}")
            plt.legend()
            plt.savefig(output_dir / f"line_graph_{relation}.png")
            plt.close()

        # 3. Combined line graph with all relations
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(g2, pos2, node_color='lightgreen', node_size=500)
        for relation in relation_types:
            edges = [(u, v) for u, v, d in g2.edges(data=True) if d['relation'] == relation]
            nx.draw_networkx_edges(g2, pos2, edgelist=edges,
                                edge_color=safe_color(edge_colors.get(relation, 'black')),
                                label=relation)

        nx.draw_networkx_labels(g2, pos2, font_size=15)
        plt.title("Line Graph - All Relation Types")
        plt.legend()
        plt.savefig(output_dir / "line_graph_all.png")
        plt.close()


    def extract_subgraph(self, full_template: dgl.DGLGraph, relation_types: Tuple[str, ...]) -> dgl.DGLGraph:
        """
        Extract a subgraph from a full line graph template based on the given relation_types.
        Assumes the full_template already has duplicates removed.

        Args:
            full_template: The original DGL heterograph containing all relations.
            relation_types: Tuple of relation types to include in the subgraph.

        Returns:
            DGL heterograph containing only the specified relation types.
        """
        node_type = full_template.ntypes[0]  # Assume single node type
        edge_types = {}

        # Select only edges for the requested relation types
        for etype in relation_types:
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

    def count_connected_components_per_relation(self, full_template: dgl.DGLGraph) -> Dict[str, int]:
        """
        Count the number of connected components for each individual relation type subgraph.
        
        Args:
            full_template: The full DGL heterograph containing all relations.
        
        Returns:
            Dictionary mapping each relation type to its connected component count.
        """
        component_counts = {}
        
        # Iterate through each relation type individually
        for relation_type in full_template.etypes:
            # Extract subgraph for this single relation type
            subgraph = self.extract_subgraph(full_template, (relation_type,))
            
            # Convert to NetworkX undirected graph for connected components analysis
            nx_graph = nx.Graph()
            
            # Add all nodes from the line graph
            num_nodes = subgraph.num_nodes()
            nx_graph.add_nodes_from(range(num_nodes))
            
            # Add edges for this relation type
            if relation_type in subgraph.etypes:
                src, dst = subgraph.edges(etype=relation_type)
                edges = list(zip(src.tolist(), dst.tolist()))
                nx_graph.add_edges_from(edges)
            
            # Count connected components
            num_components = nx.number_connected_components(nx_graph)
            component_counts[relation_type] = num_components
        
        return component_counts


    def analyze_individual_connectivity(self, n_nodes: int, output_results: bool = True) -> Dict:
        """
        Analyze connectivity for each individual relation type subgraph.
        
        Args:
            n_nodes: Number of nodes in the original complete graph.
            output_results: Whether to print results.
        
        Returns:
            Dictionary containing connectivity analysis for each relation type.
        """
        # Create full template
        full_template = self.create_line_graph_template(n_nodes)
        
        # Get component counts for each relation type
        component_counts = self.count_connected_components_per_relation(full_template)
        
        # Organize results
        results = {
            'n_nodes_original': n_nodes,
            'n_nodes_line_graph': full_template.num_nodes(),
            'relation_component_counts': component_counts
        }
        
        if output_results:
            print(f"\n=== Connected Components Analysis ===")
            print(f"Original graph nodes: {n_nodes}")
            print(f"Line graph nodes: {full_template.num_nodes()}")
            print(f"\nConnected Components per Relation Type:")
            print("-" * 40)
            
            for relation_type, count in component_counts.items():
                print(f"{relation_type:>4}: {count:>3} components")
            
            print("-" * 40)
        
        return results
    

    # def analyze_component_structures(self, full_template: dgl.DGLGraph, detailed_output: bool = True) -> Dict:
    #     """
    #     Analyze the structure and characteristics of connected components for each relation type.
        
    #     Args:
    #         full_template: The full DGL heterograph containing all relations
    #         detailed_output: Whether to print detailed analysis
        
    #     Returns:
    #         Dictionary containing detailed component analysis
    #     """
    #     results = {
    #         'relation_analysis': {},
    #         'best_components_summary': {},
    #         'component_details': {}
    #     }
        
    #     for relation_type in full_template.etypes:
    #         # Extract subgraph for this relation type
    #         subgraph = self.extract_subgraph(full_template, (relation_type,))
            
    #         # Convert to NetworkX for analysis
    #         nx_graph = nx.Graph()
    #         num_nodes = subgraph.num_nodes()
    #         nx_graph.add_nodes_from(range(num_nodes))
            
    #         if relation_type in subgraph.etypes:
    #             src, dst = subgraph.edges(etype=relation_type)
    #             edges = list(zip(src.tolist(), dst.tolist()))
    #             nx_graph.add_edges_from(edges)
            
    #         # Get connected components
    #         components = list(nx.connected_components(nx_graph))
            
    #         # Analyze each component
    #         component_info = []
    #         for i, component in enumerate(components):
    #             component_size = len(component)
    #             component_subgraph = nx_graph.subgraph(component)
                
    #             # Calculate graph metrics
    #             num_edges = component_subgraph.number_of_edges()
    #             density = nx.density(component_subgraph) if component_size > 1 else 0
                
    #             # Check if it's a simple structure
    #             is_star = False
    #             is_path = False
    #             is_clique = False
    #             is_isolated = component_size == 1
                
    #             if component_size > 1:
    #                 degrees = [component_subgraph.degree(node) for node in component]
    #                 max_degree = max(degrees)
    #                 min_degree = min(degrees)
                    
    #                 # Star: one node with high degree, others with degree 1
    #                 if max_degree == component_size - 1 and min_degree == 1:
    #                     is_star = True
                    
    #                 # Path: two nodes with degree 1, others with degree 2
    #                 elif sorted(degrees) == [1, 1] + [2] * (component_size - 2):
    #                     is_path = True
                    
    #                 # Clique: all nodes connected to all others
    #                 elif density == 1.0:
    #                     is_clique = True
                
    #             component_info.append({
    #                 'component_id': i,
    #                 'nodes': component,
    #                 'size': component_size,
    #                 'edges': num_edges,
    #                 'density': density,
    #                 'is_star': is_star,
    #                 'is_path': is_path,
    #                 'is_clique': is_clique,
    #                 'is_isolated': is_isolated,
    #                 'structure_type': (
    #                     'isolated' if is_isolated else
    #                     'star' if is_star else
    #                     'path' if is_path else
    #                     'clique' if is_clique else
    #                     'complex'
    #                 )
    #             })
            
    #         # Summarize for this relation
    #         total_components = len(components)
    #         component_sizes = [info['size'] for info in component_info]
    #         avg_component_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0
    #         max_component_size = max(component_sizes) if component_sizes else 0
            
    #         # Count structure types
    #         structure_counts = defaultdict(int)
    #         for info in component_info:
    #             structure_counts[info['structure_type']] += 1
            
    #         results['relation_analysis'][relation_type] = {
    #             'total_components': total_components,
    #             'avg_component_size': avg_component_size,
    #             'max_component_size': max_component_size,
    #             'component_sizes': component_sizes,
    #             'structure_counts': dict(structure_counts),
    #             'components': component_info
    #         }
            
    #         # Identify "best" components (largest simple structures)
    #         simple_components = [info for info in component_info 
    #                         if info['structure_type'] in ['star', 'path', 'clique']]
            
    #         if simple_components:
    #             best_component = max(simple_components, key=lambda x: x['size'])
    #             results['best_components_summary'][relation_type] = {
    #                 'best_structure': best_component['structure_type'],
    #                 'best_size': best_component['size'],
    #                 'best_component_id': best_component['component_id']
    #             }
        
    #     # Print detailed analysis
    #     if detailed_output:
    #         print("=== Component Structure Analysis ===\n")
            
    #         for relation_type in full_template.etypes:
    #             analysis = results['relation_analysis'][relation_type]
    #             print(f"{relation_type.upper()} Relation:")
    #             print(f"  Total components: {analysis['total_components']}")
    #             print(f"  Average component size: {analysis['avg_component_size']:.2f}")
    #             print(f"  Max component size: {analysis['max_component_size']}")
    #             print(f"  Component sizes: {analysis['component_sizes']}")
    #             print(f"  Structure types: {analysis['structure_counts']}")
                
    #             if relation_type in results['best_components_summary']:
    #                 best = results['best_components_summary'][relation_type]
    #                 print(f"  Best simple component: {best['best_structure']} (size {best['best_size']})")
                
    #             print()
        
    #     return results

    # def find_optimal_relation_for_simplicity(self, full_template: dgl.DGLGraph) -> Tuple[str, Dict]:
    #     """
    #     Find the relation type that produces the highest number of simple, large components.
        
    #     Args:
    #         full_template: The full DGL heterograph containing all relations
        
    #     Returns:
    #         Tuple of (best_relation_type, analysis_details)
    #     """
    #     analysis = self.analyze_component_structures(full_template, detailed_output=False)
        
    #     # Score each relation type based on simplicity and size
    #     relation_scores = {}
        
    #     for relation_type, data in analysis['relation_analysis'].items():
    #         score = 0
            
    #         # Bonus for having simple structures
    #         simple_structures = ['star', 'path', 'clique']
    #         for component in data['components']:
    #             if component['structure_type'] in simple_structures:
    #                 # Score = size * structure_bonus
    #                 structure_bonus = {
    #                     'star': 3,    # Stars are very simple
    #                     'path': 2,    # Paths are simple
    #                     'clique': 1,  # Cliques are dense but still simple
    #                 }
    #                 score += component['size'] * structure_bonus[component['structure_type']]
            
    #         # Penalty for complex structures
    #         for component in data['components']:
    #             if component['structure_type'] == 'complex':
    #                 score -= component['size'] * 0.5
            
    #         # Bonus for balanced component sizes (not one huge component)
    #         sizes = data['component_sizes']
    #         if sizes:
    #             size_variance = sum((s - data['avg_component_size'])**2 for s in sizes) / len(sizes)
    #             score += max(0, 100 - size_variance)  # Lower variance = higher score
            
    #         relation_scores[relation_type] = score
        
    #     # Find the best relation type
    #     best_relation = max(relation_scores, key=relation_scores.get)
        
    #     print("=== Optimal Relation for Simple Components ===")
    #     print(f"Relation scores:")
    #     for relation, score in sorted(relation_scores.items(), key=lambda x: x[1], reverse=True):
    #         print(f"  {relation.upper()}: {score:.2f}")
        
    #     print(f"\nBest relation type: {best_relation.upper()}")
    #     print(f"Score: {relation_scores[best_relation]:.2f}")
        
    #     return best_relation, analysis['relation_analysis'][best_relation]


    # def get_component_node_mapping(self, full_template: dgl.DGLGraph, relation_type: str) -> Dict[int, List[Tuple]]:
    #     """
    #     Get the actual original graph edges that belong to each component.
        
    #     Args:
    #         full_template: The full DGL heterograph
    #         relation_type: The relation type to analyze
        
    #     Returns:
    #         Dictionary mapping component_id -> list of original graph edges
    #     """
    #     # Get edge mapping from line graph nodes to original edges
    #     edge_mapping = full_template.ndata['edge_mapping'].numpy()
        
    #     # Extract subgraph for this relation
    #     subgraph = self.extract_subgraph(full_template, (relation_type,))
        
    #     # Convert to NetworkX
    #     nx_graph = nx.Graph()
    #     num_nodes = subgraph.num_nodes()
    #     nx_graph.add_nodes_from(range(num_nodes))
        
    #     if relation_type in subgraph.etypes:
    #         src, dst = subgraph.edges(etype=relation_type)
    #         edges = list(zip(src.tolist(), dst.tolist()))
    #         nx_graph.add_edges_from(edges)
        
    #     # Get connected components
    #     components = list(nx.connected_components(nx_graph))
        
    #     # Map back to original edges
    #     component_edges = {}
    #     for comp_id, component in enumerate(components):
    #         original_edges = []
    #         for line_node in component:
    #             if line_node < len(edge_mapping):
    #                 # Convert back to tuple format
    #                 edge = tuple(edge_mapping[line_node].astype(int))
    #                 original_edges.append(edge)
    #         component_edges[comp_id] = original_edges
        
    #     return component_edges


    # def comprehensive_component_analysis(self, n_nodes: int):
    #     """
    #     Run comprehensive analysis to find the best components and their characteristics.
    #     """
    #     print(f"=== Comprehensive Component Analysis (n={n_nodes}) ===\n")
        
    #     # Create line graph template
    #     full_template = self.create_line_graph_template(n_nodes)
        
    #     # Analyze component structures
    #     analysis = self.analyze_component_structures(full_template)
        
    #     # Find optimal relation
    #     best_relation, best_analysis = self.find_optimal_relation_for_simplicity(full_template)
        
    #     # Show the actual edges in the best components
    #     print(f"\n=== Best Components for {best_relation.upper()} ===")
    #     component_edges = self.get_component_node_mapping(full_template, best_relation)
        
    #     for comp_id, edges in component_edges.items():
    #         component_info = best_analysis['components'][comp_id]
    #         print(f"Component {comp_id} ({component_info['structure_type']}, size {component_info['size']}):")
    #         print(f"  Original edges: {edges}")
        
    #    return analysis, best_relation

if __name__ == '__main__':
    transform = LineGraphTransform(half_st=True)
    n_nodes = 4
    full_g = transform.create_line_graph_template(n_nodes=n_nodes)  # example
    transform.plot_full_graph(full_g, output_dir="../jobs/runs/plots")
    # Just get the counts
    # well the CC number is equal to n for the case of ss and tt, while st have 1 in other hand the pp have n*(n-1)/2
    # counts = transform.count_connected_components_per_relation(full_g)
    # print(counts)  # {'ss': 2, 'st': 1, 'tt': 3, 'pp': 6}

    # # Or get full analysis with printed output
    # results = transform.analyze_individual_connectivity(n_nodes=n_nodes)
    # print(results)

    # analysis, best_relation = transform.comprehensive_component_analysis(n_nodes=n_nodes)
    # print(f"Best relation type for simple components: {best_relation}")
    # print(analysis)
