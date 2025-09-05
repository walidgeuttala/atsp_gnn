import pickle
import itertools
import numpy as np

def compute_tour_cost(tour, adj_matrix):
    """Compute total cost of a tour."""
    return sum(adj_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))

def brute_force_optimal_tour(adj_matrix):
    """Return the optimal tour and its cost using brute-force (suitable for n <= 10)."""
    n = adj_matrix.shape[0]
    best_cost = float('inf')
    best_tour = None
    for perm in itertools.permutations(range(n)):
        perm = list(perm) + [perm[0]]  # make it a cycle
        cost = compute_tour_cost(perm, adj_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = perm
    return best_tour, best_cost

def compute_edge_regret(adj_matrix, optimal_cost, edge):
    """
    Compute regret of an edge by forcing it into the tour.
    Returns (min_cost_with_edge - optimal_cost) / optimal_cost
    """
    n = adj_matrix.shape[0]
    u, v = edge
    best_cost = float('inf')
    
    for perm in itertools.permutations([i for i in range(n) if i != u and i != v]):
        # Insert edge (u->v) somewhere in the cycle
        tour = [u, v] + list(perm) + [u]  # make it a cycle
        cost = compute_tour_cost(tour, adj_matrix)
        if cost < best_cost:
            best_cost = cost

    regret = (best_cost - optimal_cost) / optimal_cost
    return regret

def validate_instance_with_regret(instance_path):
    """Validate one ATSP instance with brute-force optimal tour and edge regret."""
    with open(instance_path, 'rb') as f:
        G = pickle.load(f)
    
    n = G.number_of_nodes()
    adj_matrix = np.zeros((n, n))
    for i, j in G.edges:
        adj_matrix[i, j] = G.edges[i, j]['weight']

    # Compute optimal tour
    optimal_tour, optimal_cost = brute_force_optimal_tour(adj_matrix)
    print(f"Optimal tour cost: {optimal_cost}")
    print(f"Optimal tour: {optimal_tour}")

    # Check regret values
    for u, v in G.edges:
        in_opt = ((u, v) in zip(optimal_tour, optimal_tour[1:]))
        if not in_opt:
            computed_regret = compute_edge_regret(adj_matrix, optimal_cost, (u, v))
            stored_regret = G.edges[u, v].get('regret', None)
            print(f"Edge ({u}->{v}): stored regret={stored_regret}, computed regret={computed_regret:.4f}")
        else:
            print(f"Edge ({u}->{v}) is in optimal tour, regret=0")

# Example usage:
validate_instance_with_regret("/project/c_gnn_001/code/tsp/atsp_gnn/saved_dataset/ATSP_10x10/d5a047d84e354625af95627072021ece.pkl")
