import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
import dgl
def optimized_line_graph(num_nodes, relation_types, output_dir="graphs"):
    # Create a random graph with num_nodes
    g = nx.complete_graph(num_nodes, create_using=nx.DiGraph())

    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
    half_st = False
    if 'ss' in relation_types:
        ss = torch.empty((m1, 2), dtype=torch.int32)
    if 'st' in relation_types:
        if half_st:
            st = torch.empty((m1, 2), dtype=torch.int32)
        else:
            st = torch.empty((m1*2, 2), dtype=torch.int32)
    if 'tt' in relation_types:
        tt = torch.empty((m1, 2), dtype=torch.int32)
    if 'pp' in relation_types:
        pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in relation_types:
                            ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'st' in relation_types:
                            if half_st:
                                st[idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                            else:
                                st[idx*2] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                                st[idx*2+1] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'tt' in relation_types:
                            tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        idx += 1
        if 'pp' in relation_types:
            for y in range(x+1, n):
                pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
                idx2 += 1
    edge_types = {}

    if 'ss' in relation_types:
        edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    if 'st' in relation_types:
        edge_types[('node1', 'st', 'node1')] = (st[:, 0], st[:, 1])
    if 'tt' in relation_types:
        edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    if 'pp' in relation_types:
        edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

    g2 = dgl.heterograph(edge_types)

    g2 = dgl.add_reverse_edges(g2)

    g2.ndata['e'] = torch.tensor(list(edge_id.keys()))
    
    # g2 = nx.Graph()

    # # Adding edges from different relation types
    # if 'ss' in relation_types:
    #     for i in range(ss.size(0)):
    #         g2.add_edge(ss[i, 0].item(), ss[i, 1].item(), relation='ss')
    # if 'st' in relation_types:
    #     for i in range(st.size(0)):
    #         g2.add_edge(st[i, 0].item(), st[i, 1].item(), relation='st')
    # if 'ts' in relation_types:
    #     for i in range(ts.size(0)):
    #         g2.add_edge(ts[i, 0].item(), ts[i, 1].item(), relation='ts')
    # if 'tt' in relation_types:
    #     for i in range(tt.size(0)):
    #         g2.add_edge(tt[i, 0].item(), tt[i, 1].item(), relation='tt')
    # if 'pp' in relation_types:
    #     for i in range(pp.size(0)):
    #         g2.add_edge(pp[i, 0].item(), pp[i, 1].item(), relation='pp')
  
    

    # # Visualize and save the original graph
    # plt.figure(figsize=(8, 8))
    # pos = nx.spring_layout(g)
    # nx.draw(g, pos, with_labels=False, node_color='skyblue', node_size=500, edge_color='gray', font_size=15)

    # # Draw edge labels with offsets to prevent overlap
    # edge_labels = {edge: str(edge_id[edge]) for edge in g.edges()}
    # for edge in g.edges():
    #     # Calculate the label position
    #     x_offset = np.random.uniform(-0.1, 0.1)
    #     y_offset = np.random.uniform(-0.1, 0.1)
    #     x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2 + x_offset
    #     y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2 + y_offset
    #     plt.text(x, y, edge_labels[edge], fontsize=12, ha='center')

    # plt.title("Original Graph")
    # plt.savefig(f"{output_dir}/original_graph.png")  # Save the original graph
    # plt.close()

    # # Define edge colors for different relation types
    # edge_colors = {
    #     'ss': 'red',
    #     'st': 'green',
    #     'tt': 'blue',
    #     'pp': 'orange'
    # }

    # # Visualize and save the line graphs for each type
    # for relation in relation_types:
    #     plt.figure(figsize=(8, 8))
    #     pos2 = nx.spring_layout(g2)
        
    #     # Draw the nodes
    #     nx.draw_networkx_nodes(g2, pos2, node_color='lightgreen', node_size=500)
        
    #     # Draw edges based on relation type
    #     edges = [(u, v) for u, v, d in g2.edges(data=True) if d['relation'] == relation]
    #     nx.draw_networkx_edges(g2, pos2, edgelist=edges, edge_color=edge_colors.get(relation, 'black'), label=relation)
        
    #     nx.draw_networkx_labels(g2, pos2, font_size=15)
    #     plt.title(f"Line Graph - Relation Type: {relation}")
    #     plt.legend()
    #     plt.savefig(f"{output_dir}/line_graph_{relation}.png")  # Save the line graph for this relation
    #     plt.close()

    # # Visualize and save all types of edges in the same figure
    # plt.figure(figsize=(8, 8))
    # pos2 = nx.spring_layout(g2)
    # nx.draw_networkx_nodes(g2, pos2, node_color='lightgreen', node_size=500)
    
    # # Draw all edges with their respective colors
    # for relation in relation_types:
    #     edges = [(u, v) for u, v, d in g2.edges(data=True) if d['relation'] == relation]
    #     nx.draw_networkx_edges(g2, pos2, edgelist=edges, edge_color=edge_colors.get(relation, 'black'), label=relation)

    # nx.draw_networkx_labels(g2, pos2, font_size=15)
    # plt.title("Line Graph - All Relation Types")
    # plt.legend()
    # plt.savefig(f"{output_dir}/line_graph_all.png")  # Save the combined line graph
    # plt.close()

    return g2, edge_id



def optimized_line_graph2(g, relation_types):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
    if 'ss' in relation_types:
        ss = torch.empty((m1, 2), dtype=torch.int32)
    if 'st' in relation_types:
        st = torch.empty((m1, 2), dtype=torch.int32)
    if 'ts' in relation_types:
        ts = torch.empty((m1, 2), dtype=torch.int32)
    if 'tt' in relation_types:
        tt = torch.empty((m1, 2), dtype=torch.int32)
    if 'pp' in relation_types:
        pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in relation_types:
                            ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'st' in relation_types:
                            st[idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                            ts[idx] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'tt' in relation_types:
                            tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        idx += 1
        if 'pp' in relation_types:
            for y in range(x+1, n):
                pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
                idx2 += 1
    edge_types = {}

    if 'ss' in relation_types:
        edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    if 'st' in relation_types:
        edge_types[('node1', 'st', 'node1')] = (st[:, 0], st[:, 1])
    if 'ts' in relation_types:
        edge_types[('node1', 'ts', 'node1')] = (ts[:, 0], ts[:, 1])
    if 'tt' in relation_types:
        edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    if 'pp' in relation_types:
        edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])
    # g2 = dgl.heterograph(edge_types)
    # # Adding self loops
    # # for rel in g2.etypes:
    # #     nodes = torch.arange(g2.num_nodes('node1'))
        
    # #     # Add self-loops for the current relation type
    # #     g2 = dgl.add_edges(g2, nodes.to(torch.int32), nodes.to(torch.int32), etype=rel)
    # g2 = dgl.add_reverse_edges(g2)

    # g2.ndata['e'] = torch.tensor(list(edge_id.keys()))

    # return g2, edge_id

import networkx as nx

import networkx as nx

def print_connected_components_info(graph):
    # Identify all unique relation types in the graph
    relation_types = set(data['relation'] for _, _, data in graph.edges(data=True))

    print(f"Number of unique relation types: {len(relation_types)}")
    
    for relation in relation_types:
        # Create a subgraph for each relation type
        relation_subgraph = graph.edge_subgraph(
            [(u, v) for u, v, d in graph.edges(data=True) if d['relation'] == relation]
        )

        # Get all connected components for the current relation subgraph
        connected_components = list(nx.connected_components(relation_subgraph.to_undirected()))
        print(f"\nRelation '{relation}':")
        print(f"  Number of connected components: {len(connected_components)}")

        # Iterate through each connected component
        for idx, component in enumerate(connected_components):
            subgraph = relation_subgraph.subgraph(component)
            edge_count = len(subgraph.edges())
            print(f"  Connected Component {idx + 1}: {edge_count} edges")

# Example usage
# Assuming 'g' is your betwokrX graph
# print_connected_components_info(g)



# Example usage:
num_nodes = 100  # Number of nodes in the graph
relation_types = ['ss', 'st', 'tt']  # Define the types of relations to create
g = nx.complete_graph(num_nodes, create_using=nx.DiGraph())
g, _ = optimized_line_graph(num_nodes, relation_types)
# print_connected_components_info(g)
file_path = '../tsp_input/graph_atsp_1000_none_st.dgl'

# Save the graph
dgl.save_graphs(file_path, [g])