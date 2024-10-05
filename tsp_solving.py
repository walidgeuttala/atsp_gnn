import json
import numpy as np
import tsplib95
import lkh
import os
import gnngls



def create_tsplib95_string(adjacency_matrix):
    # Get the shape of the matrix
    n = adjacency_matrix.shape[0]

    # Initialize an empty string to store the result
    result = f'''NAME: TSP
    COMMENT: {n}-city problem
    TYPE: TSP
    DIMENSION: {n}
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION: 
    '''
    for i in range(n):
        # Iterate over columns up to the diagonal
        for j in range(n):
            result += str(adjacency_matrix[i][j]) + " "
        result += "\n"  # Add a newline after each row

    return result.strip()  # Remove trailing newline

def fixed_edge_tour(string, lkh_path='../LKH-3.0.9/LKH'):

    problem = tsplib95.loaders.parse(string)
    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour

def compute_tour_cost(tour, adjacency_matrix):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += adjacency_matrix[tour[i], tour[i + 1]]  # Subtract 1 to convert 1-indexed to 0-indexed
    return cost

all_instances_lower_triangle_tour_cost = 'adj,tour,cost\n'
number_instances = 2000
output_dir = f"../tsplib95_{number_instances}_instances_2000_node"
os.mkdir(output_dir)
input_dir = "generated_tsp_tasks_64_v2"

for i in range(number_instances):
  # Assuming your JSON file is named 'data.json'
  file_path = f'../{input_dir}/instance{i}.json'

  # Read the text file
  with open(file_path, 'r') as file:
      text_string = file.read()

  rows = text_string.split('\n')

  # Split each row into elements and convert them to integers
  adjacency_matrix = [[int(x) for x in row.split()] for row in rows]

  # Convert the list of lists into a NumPy array
  adjacency_matrix = np.array(adjacency_matrix)
  adjacency_matrix = gnngls.as_symmetric(adjacency_matrix)
  # Convert the ADJ into tsplib file problem
  string_problem = create_tsplib95_string(adjacency_matrix)
  tour = fixed_edge_tour(string_problem)
  cost = compute_tour_cost(tour, adjacency_matrix)
#   # Saving tsplib file
#   # Open the file in write mode
#   with open(f'{output_dir}/instance{i}.txt', 'w') as file:
#       # Write the string into the file
#       file.write(string_problem)

  all_instances_lower_triangle_tour_cost += gnngls.convert_adj_string(adjacency_matrix)+','+" ".join(map(str, tour))+','+str(cost)+'\n'
 
with open(f'{output_dir}/all_instances_adj_tour_cost.txt', 'w') as file:
      # Write the string into the file
      file.write(all_instances_lower_triangle_tour_cost)
