import numpy as np
import networkx as nx
import os 
import torch 
import dgl
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

# This is generating the tt ss and pp graphs 
def optimized_line_graph2(g):
    memory_before = get_memory_usage()
    print(f"Memory before the function: {memory_before:.2f} MB")

    n = g.number_of_nodes()

    m1 = n * (n - 1) * (n - 2)
    m2 = n * (n - 1)
    m3 = m1 // 2
    m4 = m2 // 2

    # Pre-allocate tensors
    ss = torch.empty((m1, 2), dtype=torch.int32)
    tt = torch.empty((m1, 2), dtype=torch.int32)
    pp = torch.empty((m2, 2), dtype=torch.int32)

    # Create the edge id map
    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0

    for x in range(n):
        for y in range(n):
            if x == y:
                continue
            for z in range(y + 1, n):
                if x == z:
                    continue

                # Directly assign values to pre-allocated tensor slices
                ss[idx, :] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                ss[idx + m3, :] = torch.tensor([edge_id[(x, z)], edge_id[(x, y)]], dtype=torch.int32)
                tt[idx, :] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                tt[idx + m3, :] = torch.tensor([edge_id[(z, x)], edge_id[(y, x)]], dtype=torch.int32)

                idx += 1

        for y in range(x + 1, n):
            # Directly assign values to pre-allocated tensor slices
            pp[idx2, :] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
            pp[idx2 + m4, :] = torch.tensor([edge_id[(y, x)], edge_id[(x, y)]], dtype=torch.int32)
            idx2 += 1

    # Define edge types
    edge_types = {
        ('node1', 'ss', 'node1'): (ss[:, 0], ss[:, 1]),
        ('node1', 'tt', 'node1'): (tt[:, 0], tt[:, 1]),
        ('node1', 'pp', 'node1'): (pp[:, 0], pp[:, 1])
    }

    # Create the heterogeneous graph
    g2 = dgl.heterograph(edge_types)

    # Add edge information to the nodes (reusing edge_id for efficiency)
    g2.ndata['e'] = torch.tensor(list(edge_id.keys()), dtype=torch.int32)

    memory_peak = get_memory_usage()
    print(f"Memory peak during function: {memory_peak:.2f} MB")

    return g2


def optimized_line_graph(g, relation_types= 'tt ss pp'):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)
    m2 = n*(n-1)
    m3 = m1 // 2
    m4 = m2 // 2
    ss = torch.empty((m1, 2), dtype=torch.int32)
    tt = torch.empty((m1, 2), dtype=torch.int32)
    pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:    
                        ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        ss[idx+m3] = torch.tensor([edge_id[(x, z)], edge_id[(x, y)]], dtype=torch.int32)
                        tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        tt[idx+m3] = torch.tensor([edge_id[(z, x)], edge_id[(y, x)]], dtype=torch.int32)

                        idx += 1
        for y in range(x+1, n):
            pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
            pp[idx2+m4] = torch.tensor([edge_id[(y, x)], edge_id[(x, y)]], dtype=torch.int32)
            idx2 += 1
    edge_types = {}

    edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

  
    g2 = dgl.heterograph(edge_types)

    g2.ndata['e'] = torch.tensor(list(edge_id.keys()))
   
    return g2    

# Define the number of nodes
n = 1000
import time
complete_graph = nx.complete_graph(n, create_using=nx.DiGraph)
start_time = time.time()
g = optimized_line_graph2(complete_graph)
end_time = time.time()
print(f"Time taken for n={n}: {end_time - start_time} seconds")

dgl.save_graphs(f"../tsp_input/hetero_graph_{n}.dgl", [g])
memory_after = get_memory_usage()
print(f"Memory after the function: {memory_after:.2f} MB")