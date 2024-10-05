#import concorde.tsp as concorde
import lkh
import networkx as nx
import numpy as np
import tsplib95
from matplotlib import colors
import linecache

def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c

def tour_cost2(tour, weight):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += weight[e]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)
        if n == 0:
            if c != 2:
                return False
        elif c != 1:
            return False
    return True


# def optimal_tour(G, scale=1e3):
#     coords = scale * np.vstack([G.nodes[n]['pos'] for n in sorted(G.nodes)])
#     solver = concorde.TSPSolver.from_data(coords[:, 0], coords[:, 1], norm='EUC_2D')
#     solution = solver.solve()
#     tour = solution.tour.tolist() + [0]
#     return tour


def tranfer_tour(tour, x):
    result_list = []
    for num in tour:
        result_list.append(num)
        result_list.append(num + x)
    return result_list[:-1]

def as_symmetric(matrix, INF = 1e6):
    shape = len(matrix)
    mat = np.identity(shape) * - INF + matrix

    new_shape = shape * 2
    new_matrix = np.ones((new_shape, new_shape)) * INF
    np.fill_diagonal(new_matrix, 0)

    # insert new matrices
    new_matrix[shape:new_shape, :shape] = mat
    new_matrix[:shape, shape:new_shape] = mat.T
    # new cost matrix after transformation

    return new_matrix

def convert_adj_string(adjacency_matrix):
  ans = ''
  n = adjacency_matrix.shape[0]
  for i in range(n):
    # Iterate over columns up to the diagonal
      for j in range(n):
        ans += str(adjacency_matrix[i][j]) + " "
  return ans




def tranfer_tour(tour, x):
        result_list = []
        for num in tour:
            result_list.append(num)
            result_list.append(num + x)
        return result_list[:-1]

def append_text_to_file(filename, text):
    with open(filename, 'a') as file: file.write(text + '\n')


def atsp_to_tsp():
    value = 64e6
    for i in range(10):
        line = linecache.getline('../tsplib95_10000_instances_64_node/all_instances_adj_tour_cost.txt', i+2).strip()
        adj, opt_solution, cost = line.split(',')
        cost = float(cost)
        cost -= value
        adj = adj.split(' ')[:-1]
        opt_solution = [int(x) for x in opt_solution.split()]
        adj = np.array(adj, dtype=np.int32).reshape(64, 64)
        adj = gnngls.as_symmetric(adj)
        opt_solution = tranfer_tour(opt_solution, 64)
        instance_adj_tour_cost = gnngls.convert_adj_string(adj)+','+" ".join(map(str, opt_solution))+','+str(cost)
        append_text_to_file('../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt', instance_adj_tour_cost)

def adjacency_matrix_to_networkx(adj_matrix):
    return nx.Graph(np.triu(adj_matrix))

def optimal_cost(G, weight='weight'):
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


def get_adj_matrix_string(G):
    # Get the lower triangular adjacency matrix with diagonal
    adj_matrix = nx.to_numpy_array(G).astype(int)
    n = adj_matrix.shape[0]
    ans = f'''NAME: TSP
    COMMENT: 64-city problem
    TYPE: ATSP
    DIMENSION: {n}
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION: 
    '''
    for i in range(n):
        # Iterate over columns up to the diagonal
        for j in range(n):
            ans += str(adj_matrix[i][j]) + " "
        ans += "\n"
    # Add EOF
    # adj_matrix_string += "EOF"
    
    return ans.strip()


def fixed_edge_tour(G, e, lkh_path='../LKH-3.0.9/LKH'):
    string = get_adj_matrix_string(G)
    problem = tsplib95.loaders.parse(string)
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def plot_edge_attribute(G, attr, ax, **kwargs):
    cmap_colors = np.zeros((100, 4))
    cmap_colors[:, 0] = 1
    cmap_colors[:, 3] = np.linspace(0, 1, 100)
    cmap = colors.ListedColormap(cmap_colors)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, edge_color=attr.values(), edge_cmap=cmap, ax=ax, **kwargs)
