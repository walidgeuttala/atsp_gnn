import linecache
import networkx as nx
import numpy as np
import tsplib95
from matplotlib import colors
import lkh

def tour_to_edge_attribute(G, tour):
    """
    Convert tour to edge attributes indicating if in tour.
    
    Args:
        G (nx.Graph): The graph.
        tour (list): The tour.
    
    Returns:
        dict: Edge attributes dictionary.
    """
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    """
    Calculate the cost of a tour in the graph.
    
    Args:
        G (nx.Graph): The graph.
        tour (list): The tour.
        weight (str): Weight attribute.
    
    Returns:
        float: Total tour cost.
    """
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c


def tour_cost2(tour, weight):
    """
    Calculate tour cost from a weight dictionary.
    
    Args:
        tour (list): The tour.
        weight (dict): Weight dictionary.
    
    Returns:
        float: Total cost.
    """
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += weight[e]
    return c


def is_equivalent_tour(tour_a, tour_b):
    """
    Check if two tours are equivalent (same or reversed).
    
    Args:
        tour_a (list): First tour.
        tour_b (list): Second tour.
    
    Returns:
        bool: True if equivalent.
    """
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    """
    Validate if the tour is a valid Hamiltonian cycle.
    
    Args:
        G (nx.Graph): The graph.
        tour (list): The tour.
    
    Returns:
        bool: True if valid.
    """
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


def tranfer_tour(tour, x):
    """
    Transform tour by appending offset values.
    
    Args:
        tour (list): Original tour.
        x (int): Offset.
    
    Returns:
        list: Transformed tour.
    """
    result_list = []
    for num in tour:
        result_list.append(num)
        result_list.append(num + x)
    return result_list[:-1]


def as_symmetric(matrix, INF=1e6):
    """
    Convert matrix to symmetric for ATSP to TSP transformation.
    
    Args:
        matrix (np.ndarray): Original matrix.
        INF (float): Infinity value.
    
    Returns:
        np.ndarray: Symmetric matrix.
    """
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
    """
    Convert adjacency matrix to string representation.
    
    Args:
        adjacency_matrix (np.ndarray): The matrix.
    
    Returns:
        str: String representation.
    """
    ans = ''
    n = adjacency_matrix.shape[0]
    for i in range(n):
        # Iterate over columns up to the diagonal
        for j in range(n):
            ans += str(adjacency_matrix[i][j]) + " "
    return ans


def append_text_to_file(filename, text):
    """
    Append text to a file.
    
    Args:
        filename (str): File path.
        text (str): Text to append.
    """
    with open(filename, 'a') as file: file.write(text + '\n')


def atsp_to_tsp():
    """
    Convert ATSP instances to TSP format and append to file.
    """
    value = 64e6
    for i in range(10):
        line = linecache.getline('../tsplib95_10000_instances_64_node/all_instances_adj_tour_cost.txt', i+2).strip()
        adj, opt_solution, cost = line.split(',')
        cost = float(cost)
        cost -= value
        adj = adj.split(' ')[:-1]
        opt_solution = [int(x) for x in opt_solution.split()]
        adj = np.array(adj, dtype=np.int32).reshape(64, 64)
        adj = as_symmetric(adj)
        opt_solution = tranfer_tour(opt_solution, 64)
        instance_adj_tour_cost = convert_adj_string(adj)+','+" ".join(map(str, opt_solution))+','+str(cost)
        append_text_to_file('../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt', instance_adj_tour_cost)


def adjacency_matrix_to_networkx(adj_matrix):
    """
    Convert adjacency matrix to NetworkX graph.
    
    Args:
        adj_matrix (np.ndarray): The matrix.
    
    Returns:
        nx.Graph: The graph.
    """
    return nx.Graph(np.triu(adj_matrix))


def optimal_cost(G, weight='weight'):
    """
    Calculate optimal cost from 'in_solution' edges.
    
    Args:
        G (nx.Graph): The graph.
        weight (str): Weight attribute.
    
    Returns:
        float: Optimal cost.
    """
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


def get_adj_matrix_string(G, weight: str = 'weight', scale: float = None):
    """Generate a TSPLIB ATSP string with the graph's edge weights.

    - Extracts the full NxN weight matrix from edge attribute `weight`.
    - Scales to integers (TSPLIB requirement) using `scale` if provided, else auto-scale.

    Args:
        G (nx.DiGraph): The graph.
        weight (str): Edge attribute name for weights.
        scale (float, optional): Multiply weights by this factor before rounding to int.

    Returns:
        str: TSPLIB-formatted string with FULL_MATRIX.
    """
    W = nx.to_numpy_array(G, weight=weight)
    n = W.shape[0]

    if scale is None:
        # Auto-scale floats to preserve up to 6 decimals
        # If all entries are integers already, scale=1.
        if np.issubdtype(W.dtype, np.floating):
            scale = 1e6
        else:
            scale = 1.0
    M = np.rint(W * scale).astype(int)

    ans = (
        f"NAME: ATSP_{n}\n"
        f"COMMENT: Generated from networkx with attribute '{weight}'\n"
        f"TYPE: ATSP\n"
        f"DIMENSION: {n}\n"
        f"EDGE_WEIGHT_TYPE: EXPLICIT\n"
        f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"
        f"EDGE_WEIGHT_SECTION:\n"
    )

    for i in range(n):
        ans += " ".join(str(M[i, j]) for j in range(n)) + "\n"

    return ans.strip()


def fixed_edge_tour(G, e, lkh_path='../LKH-3.0.9/LKH'):
    """
    Solve TSP with fixed edge using LKH.
    
    Args:
        G (nx.Graph): The graph.
        e (list): Fixed edge.
        lkh_path (str): Path to LKH solver.
    
    Returns:
        list: The tour.
    """
    string = get_adj_matrix_string(G, weight='weight')
    problem = tsplib95.loaders.parse(string)
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def plot_edge_attribute(G, attr, ax, **kwargs):
    """
    Plot graph with edge colors based on attribute.
    
    Args:
        G (nx.Graph): The graph.
        attr (dict): Edge attribute dictionary.
        ax (matplotlib.axes.Axes): Axes to plot on.
        **kwargs: Additional draw kwargs.
    """
    cmap_colors = np.zeros((100, 4))
    cmap_colors[:, 0] = 1
    cmap_colors[:, 3] = np.linspace(0, 1, 100)
    cmap = colors.ListedColormap(cmap_colors)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, edge_color=attr.values(), edge_cmap=cmap, ax=ax, **kwargs)


def compute_tour_cost(tour, adjacency_matrix):
    """
    Compute tour cost from adjacency matrix.
    
    Args:
        tour (list): The tour.
        adjacency_matrix (np.ndarray): The matrix.
    
    Returns:
        float: Tour cost.
    """
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += adjacency_matrix[tour[i], tour[i + 1]]  # Subtract 1 to convert 1-indexed to 0-indexed
    return cost
