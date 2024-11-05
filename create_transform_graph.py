import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
import dgl

def optimized_line_graph(g, relation_types = 'ss tt pp', half_st = True):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
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

    return g2    



for num_nodes in [1000]:
    g = nx.complete_graph(num_nodes, create_using=nx.DiGraph())
    g = optimized_line_graph(g)

    dgl.save_graphs(f"../tsp_input/graph_{num_nodes}_half_st.dgl", [g])

