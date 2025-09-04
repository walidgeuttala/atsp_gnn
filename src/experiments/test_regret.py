import networkx as nx
import dgl
import gnngls 
import torch
from gnngls import datasets
import numpy as np
test_set = datasets.TSPDataset("../generatedn2000/test.txt")
print('hi')
def check_rows_for_zeros(arr):
    """
    Check if any row in the given NumPy array contains only zeros.

    Parameters:
    arr (numpy.ndarray): Input NumPy array of shape (n, n).

    Returns:
    str: "Yes" if any row contains only zeros, "No" otherwise.
    """
    # Check if any row contains only zeros
    if np.any(np.all(arr == 0, axis=1)):
        return 1
    else:
        return 0

import numpy as np
import networkx as nx
from pathlib import Path

def read_instance_file(filepath):
    # Initialize variables to store data
    edge_weight = None
    regret = None
    regret_pred = None

    with open(filepath, 'r') as f:
        # Read lines from the file
        lines = f.readlines()

        # Parse the lines
        for i, line in enumerate(lines):
            if line.startswith("edge_weight:"):
                # Read edge_weight array (128 lines)
                edge_weight_lines = lines[i+1:i+129]
                edge_weight = np.loadtxt(edge_weight_lines, delimiter=" ")
            elif line.startswith("regret:"):
                # Read regret array (128 lines)
                regret_lines = lines[i+1:i+129]
                regret = np.loadtxt(regret_lines, delimiter=" ")
            elif line.startswith("regret_pred:"):
                # Read regret_pred array (128 lines)
                regret_pred_lines = lines[i+1:i+129]
                regret_pred = np.loadtxt(regret_pred_lines, delimiter=" ")

    return edge_weight, regret, regret_pred


def create_graph(edge_weight, regret):
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with regret as a node attribute
    for i in range(len(regret)):
        G.add_node(i, regret=regret[i])

    # Add edges to the graph
    for i in range(len(edge_weight)):
        for j in range(len(edge_weight[i])):
            if i != j:  # Avoid self-loops
                G.add_edge(i, j, weight=edge_weight[i][j])

    return G

def count_zeros_in_rows(arr):
    """
    Count the number of zeros in each row of the given NumPy array.

    Parameters:
    arr (numpy.ndarray): Input NumPy array of shape (n, n).

    Returns:
    list: List containing the number of zeros in each row.
    """
    # Count the number of zeros in each row
    zeros_count_per_row = np.sum(arr == 0, axis=1)
    return zeros_count_per_row.flatten().tolist()

def find_min_max_2d(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return min_val, max_val

ans = 0
cnt = 0
ans_min, ans_max = 1e10, 0
output_path = Path("../atspv2n1290")
cnt = 1  # Example instance number
# Read instance file
instance_file = output_path / f"instance{cnt}.txt"
edge_weight, regret, _ = read_instance_file(instance_file)
import linecache
for i in range(1):
        line = linecache.getline("../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt", i+1).strip()
        G = nx.Graph()
        adj, opt_solution, cost = line.split(',')
        adj = adj.split(' ')
        print(len(adj))
        array_float32 = np.array(adj[:-1], dtype=np.float32).reshape((128, 128))
        break
        if np.array_equal(edge_weight, array_float32):
            print('yes')

def save_lists_to_file(file_path, list_of_lists):
    """
    Save lists into a text file where each list is represented in a separate line.

    Args:
    - file_path (str): Path to the file where lists will be saved.
    - list_of_lists (list): List containing lists to be saved.

    Returns:
    - None
    """

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each list to the file
        for sublist in list_of_lists:
            # Convert the list elements to strings and join them with spaces
            line = " ".join(map(str, sublist)) + "\n"
            # Write the line to the file
            file.write(line)
tours = []
for instance in test_set.instances:
    G = nx.read_gpickle("../generatedn2000/"+ instance)
    weight, _ = nx.attr_matrix(G, 'weight')
    optimal_cost = gnngls.optimal_cost(G)
    #print(nx.attr_matrix(G, 'weight'))
    cntt = 0
    if np.array_equal(array_float32, weight):
        print(instance)
        regret, _ = nx.attr_matrix(G, 'regret')
        for e in G.edges:
            a, b = e
            if regret[a, b] == 0. and not G.edges[e]['in_solution']:
                print("edge: ",e, end = ' ')
                tour = gnngls.fixed_edge_tour(G, e)
                cost = gnngls.tour_cost(G, tour)
                if cntt == 0:
                    print(tour)
                tour = [x for i, x in enumerate(tour) if i % 2 == 0]
                if cntt == 0:
                    print(tour)
                tours.append(tour)
                print("cost: ",cost, end = ' ')
                cntt += 1
        print('optimal cost : ', optimal_cost, end = ' ')
        print('dodo')
        save_lists_to_file('../counter_example.txt', tours)
        break
    # H = test_set.get_scaled_features(G)
    # y = H.ndata['regret']
    #regret = np.abs(test_set.scalers['regret'].inverse_transform(y.cpu().numpy()))
    
    



# Example usage

