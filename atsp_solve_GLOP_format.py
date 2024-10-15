import json
import numpy as np
import tsplib95
import lkh
import os
import time
import torch 

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
    solution = lkh.solve(lkh_path, problem=problem, max_trials=10000, population_size=100, runs=10, precision=1)
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



def main(n , samples):

    # Path variables
    output_dir = f"../tsp_input/atsp_{n}_samples_{samples}_solved_test_only"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    dataset = torch.load(f'../GLOP/data/atsp/ATSP{n}.pt')
    #samples = dataset.shape[0]
    tour_simple = list(range(n)) + [0]

    # Process each file and solve
    for i in range(samples):
        # Load the dataset
    
        adjacency_matrix = dataset[i]
        string_problem = create_tsplib95_string(adjacency_matrix.numpy()*1e6)
        # Measure time for solving the first instance for estimation
        if i == 0:
            start_time = time.time()

        # Solve the problem using LKH
        tour = fixed_edge_tour(string_problem)
        cost = compute_tour_cost(tour, adjacency_matrix)
        cost_simple = compute_tour_cost(tour_simple, adjacency_matrix)
        print(f'tour : {len(tour)}',flush=True)
        if i == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * samples
            print(f"Time for first sample: {elapsed_time:.2f} seconds", flush=True)
            print(f"Estimated time for {samples} samples: {estimated_total_time / 60:.2f} minutes", flush=True)

        # Save the result in the output directory
        output_file = os.path.join(output_dir, f"solved_instance_{i}.json")
        
        with open(output_file, 'w') as file:
            result = {
                "adjacency_matrix": adjacency_matrix.tolist(),
                "tour": tour,
                "cost": cost.item()
            }
            json.dump(result, file)

        print(f"Instance {i+1}/{samples} solved. Cost: {cost}, Cost Simple : {cost_simple}", flush=True)

if __name__ == '__main__':
    atsp_sizes = [100, 150, 250, 500, 1000]
    samples = 30
    for n in atsp_sizes:
        main(n, samples)