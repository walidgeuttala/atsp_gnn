import pathlib
import pickle
import psutil 
import os

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data

from . import tour_cost, fixed_edge_tour, optimal_cost as get_optimal_cost

def set_features(G):
    for e in G.edges:
        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)

def set_labels(G):
    optimal_cost = get_optimal_cost(G)
    if optimal_cost == 0:
        optimal_cost = 1e-6
    if optimal_cost < 0:
        value = -1.
    else:
        value = 1.
    for e in G.edges:
        regret = 0.

        if not G.edges[e]['in_solution']: 

            tour = fixed_edge_tour(G, e)
            cost = tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost * value
            
        G.edges[e]['regret'] = regret

def set_labels2(G):
    for e in G.edges:
        regret = 0.
        G.edges[e]['regret'] = regret

def log_memory_usage2(step_description):
    process = psutil.Process(os.getpid())
    print(f"Step: {step_description}", flush=True)
    print(f"Memory usage RAM: {process.memory_info().rss / 1024 ** 2:.2f} MB", flush=True)
    print("=" * 40, flush=True)
    
def log_memory_usage(step_description):
    print(f"Step: {step_description}", flush=True)
    print(f"Allocated memory GPU: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", flush=True)
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", flush=True)
    print("=" * 40, flush=True)

def optimized_line_graph_partition(g, args):
    n = g.number_of_nodes()
    m1 = (n-1)*(n-2)//2
    m2 = n*(n-1)//2
    if 'ss' in args.relation_types:
        ss = torch.empty((n, m1, 2), dtype=torch.int32)
    if 'tt' in args.relation_types:
        tt = torch.empty((n, m1, 2), dtype=torch.int32)
    # if 'pp' in args.relation_types:
    #     pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    
    idx2 = 0
    for x in range(0, n):
        idx = 0
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in args.relation_types:
                            ss[x][idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'tt' in args.relation_types:
                            tt[x][idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        idx += 1
        # if 'pp' in args.relation_types:
        #     for y in range(x+1, n):
        #         pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
        #         idx2 += 1
    edge_types = {}

    if 'ss' in args.relation_types:
        edge_types[('node1', 'ss', 'node1')] = (ss[0][:, 0], ss[0][:, 1])
    if 'tt' in args.relation_types:
        edge_types[('node1', 'tt', 'node1')] = (tt[0][:, 0], tt[0][:, 1])
    # if 'pp' in args.relation_types:
    #     edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

  
    g2 = dgl.heterograph(edge_types)

    g2 = dgl.add_reverse_edges(g2)

    g2.ndata['e'] = torch.tensor(list(edge_id.keys()))

    return g2, edge_id    


def optimized_line_graph(g, args):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
    if 'ss' in args.relation_types:
        ss = torch.empty((m1, 2), dtype=torch.int32)
    if 'st' in args.relation_types:
        if args.half_st:
            st = torch.empty((m1, 2), dtype=torch.int32)
        else:
            st = torch.empty((m1*2, 2), dtype=torch.int32)
    if 'tt' in args.relation_types:
        tt = torch.empty((m1, 2), dtype=torch.int32)
    if 'pp' in args.relation_types:
        pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in args.relation_types:
                            ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'st' in args.relation_types:
                            if args.half_st:
                                st[idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                            else:
                                st[idx*2] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                                st[idx*2+1] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'tt' in args.relation_types:
                            tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        idx += 1
        if 'pp' in args.relation_types:
            for y in range(x+1, n):
                pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
                idx2 += 1
    edge_types = {}

    if 'ss' in args.relation_types:
        edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    if 'st' in args.relation_types:
        edge_types[('node1', 'st', 'node1')] = (st[:, 0], st[:, 1])
    if 'tt' in args.relation_types:
        edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    if 'pp' in args.relation_types:
        edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

  
    g2 = dgl.heterograph(edge_types)

    g2 = dgl.add_reverse_edges(g2)

    g2.ndata['e'] = torch.tensor(list(edge_id.keys()))

    return g2, edge_id    

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, args, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent

        self.instances = [line.strip() for line in open(instances_file)]

        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        if 'edges' in scalers: # for backward compatability
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers

        G = nx.read_gpickle(self.root_dir / self.instances[0])
        self.G, self.edge_id = optimized_line_graph(G, args)

        # Transfer the Hetro to Homo
        if args.to_homo:
            self.G = dgl.to_homogeneous(self.G, ndata=['e'])

        self.etypes = self.G.etypes

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        G = nx.read_gpickle(self.root_dir / self.instances[i])
        H = self.get_scaled_features(G)
        return H

    def get_scaled_features(self, G):
        
        features = []
        regret = []
        in_solution = []
        for e, _ in self.edge_id.items():
            features.append(G.edges[e]['weight'])
            regret.append(G.edges[e]['regret'])
            in_solution.append(G.edges[e]['in_solution'])

        features = np.vstack(features)
        features_transformed = self.scalers['weight'].transform(features)
        regret = np.vstack(regret)
        regret_transformed = self.scalers['regret'].transform(regret)
        in_solution = np.vstack(in_solution)
        
        H = self.G
        H.ndata['weight'] = torch.tensor(features_transformed, dtype=torch.float32)
        H.ndata['regret'] = torch.tensor(regret_transformed, dtype=torch.float32)
        H.ndata['in_solution'] = torch.tensor(in_solution, dtype=torch.float32)
        H.ndata['e'] = self.G.ndata['e'].clone()
        return H



# import networkx as nx
# import matplotlib.pyplot as plt

# # Define the number of nodes
# n = 500

# # Create a complete directed graph
# complete_graph = nx.complete_graph(n, create_using=nx.DiGraph)

# # Optionally, you can visualize the graph (be cautious with large graphs)
# # For a complete graph of size 250, visualization may not be clear or useful
# # You can comment the visualization part if needed
# # pos = nx.spring_layout(complete_graph)  # positions for all nodes
# # nx.draw(complete_graph, pos, node_size=50, with_labels=False)
# # plt.show()

# # To verify the number of edges in the complete directed graph
# print(f"Number of nodes: {complete_graph.number_of_nodes()}")
# print(f"Number of edges: {complete_graph.number_of_edges()}")


# _ = optimized_line_graph(complete_graph)
