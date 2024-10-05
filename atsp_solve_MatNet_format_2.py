import json
import numpy as np
import tsplib95
import lkh
import os
import time

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


# Path variables
input_dir = "ATSP/n100"
output_dir = "atsp_100_solved"
files = list_files_in_directory(input_dir+"/")

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Solve for 3000 instances
samples_to_solve = 10000
total_files = len(files)
if samples_to_solve > total_files:
    samples_to_solve = total_files

# Process each file and solve
for i in range(3000, samples_to_solve):
    file_path = f'{input_dir}/problem_100_0_1000000_{i}.atsp'
    adjacency_matrix = parse_atsp_file(file_path)
    string_problem = create_tsplib95_string(adjacency_matrix)
    # Measure time for solving the first instance for estimation
    if i == 0:
        start_time = time.time()

    # Solve the problem using LKH
    tour = fixed_edge_tour(string_problem)
    cost = compute_tour_cost(tour, adjacency_matrix)

    if i == 0:
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * samples_to_solve
        print(f"Time for first sample: {elapsed_time:.2f} seconds", flush=True)
        print(f"Estimated time for {samples_to_solve} samples: {estimated_total_time / 60:.2f} minutes", flush=True)

    # Save the result in the output directory
    output_file = os.path.join(output_dir, f"solved_instance_{i}.json")
    with open(output_file, 'w') as file:
        result = {
            "adjacency_matrix": adjacency_matrix.tolist(),
            "tour": tour,
            "cost": cost
        }
        json.dump(result, file)

    print(f"Instance {i+1}/{samples_to_solve} solved. Cost: {cost}", flush=True)
