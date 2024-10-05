import json
import numpy as np
import tsplib95
import lkh
import os
import gnngls

def create_tsplib95_string(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    result = f'''NAME: ATSP
COMMENT: {n}-city problem
TYPE: ATSP
DIMENSION: {n}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
'''
    for i in range(n):
        for j in range(n):
            result += str(adjacency_matrix[i][j]) + " "
        result += "\n"
    return result.strip()

def fixed_edge_tour(string, lkh_path='../LKH-3.0.9/LKH'):
    problem = tsplib95.parse(string)
    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [solution[0][0] - 1]  # Ensure tour is a cycle
    return tour

def compute_tour_cost(tour, adjacency_matrix):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += adjacency_matrix[tour[i], tour[i + 1]]
    return cost

def parse_atsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    dimension = 0
    edge_weight_section_start = False
    weights = []

    for line in lines:
        if line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            edge_weight_section_start = True
        elif edge_weight_section_start:
            if line.strip() == "EOF":
                break
            weights.extend(map(float, line.split()))

    adjacency_matrix = np.zeros((dimension, dimension))
    idx = 0
    for i in range(dimension):
        for j in range(dimension):
            adjacency_matrix[i][j] = weights[idx]
            idx += 1

    return adjacency_matrix

def list_files_in_directory(directory_path):
    """
    Returns a list of file names in the specified directory path.
    
    Parameters:
    - directory_path (str): Path to the directory
    
    Returns:
    - list: List of file names in the directory
    """
    file_list = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            file_list.append(file)
    return file_list




all_instances_lower_triangle_tour_cost = 'adj,tour,cost\n'
input_dir = "tslib_atsp"
files = list_files_in_directory("../"+input_dir+"/")


number_instances = len(files)
output_dir = f"../atsplib95_{number_instances}_instances_public_data"
os.makedirs(output_dir, exist_ok=True)



for i in range(number_instances):
    file_path = f'../{input_dir}/{files[i]}'
    adjacency_matrix = parse_atsp_file(file_path)
    #adjacency_matrix = gnngls.as_symmetric(adjacency_matrix)
    string_problem = create_tsplib95_string(adjacency_matrix)
    tour = fixed_edge_tour(string_problem)
    cost = compute_tour_cost(tour, adjacency_matrix)
    print(cost)
    all_instances_lower_triangle_tour_cost += gnngls.convert_adj_string(adjacency_matrix) + ',' + " ".join(map(str, tour)) + ',' + str(cost) + '\n'
    
# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write the accumulated string to the output file
output_file_path = os.path.join(output_dir, 'all_instances_adj_tour_cost.txt')
with open(output_file_path, 'w') as file:
    file.write(all_instances_lower_triangle_tour_cost)



