import numpy as np
import networkx as nx
import os 


dir_path = '../../cleaned_data_n5900/'
output_path = '../../tsp_n5900/'
files = [dir_path+file for file in os.listdir(dir_path)]


edges_to_remove1 = [(node_i, node_j) for node_i in range(64) for node_j in range(64)]
edges_to_remove2 = [(node_i, node_j) for node_i in range(64, 128) for node_j in range(64, 128)]
idx = 2000

for file in files:
    G = nx.read_gpickle(file) 
    G.remove_edges_from(edges_to_remove1)
    #G.remove_edges_from(edges_to_remove2)
    nx.write_gpickle(G, output_path+f'instance{idx}.pkl')
    idx += 1