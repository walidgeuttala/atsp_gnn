import json
import numpy as np
import tsplib95
import lkh
adj = [
    [0, 1, 0],   # Row 1
    [5, 0, 100],   # Row 2
    [1, 0, 0]    # Row 3
]


def create_tsplib95_string(adj):
    result = '''NAME: ATSP
    COMMENT: 64-city problem
    TYPE: ATSP
    DIMENSION: 3
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION: 
    '''
    for i in range(3):
        # Iterate over columns up to the diagonal
        for j in range(3):
            result += str(adj[i][j]) + " "
        result += "\n"  # Add a newline after each row
    #result += 'EOF'
    print(result)
    result = result.strip()
    problem = tsplib95.loaders.parse(result)
    print(problem.as_name_dict())
    solution = lkh.solve("../LKH-3.0.9/LKH", problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]

    print(tour)

create_tsplib95_string(adj)