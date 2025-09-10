import time

import networkx as nx
import numpy as np

from .operators import two_opt_a2a, relocate_a2a, two_opt_o2a, relocate_o2a
from .atsp_utils import tour_cost, tour_cost2

def nearest_neighbor(G, start, weight='weight'):
    """
    Generate a tour using the nearest neighbor heuristic.
    
    Args:
        G (nx.Graph): The graph.
        start (int): Starting node.
        weight (str): Edge weight attribute.
    
    Returns:
        list: The generated tour.
    """
    tour = [start]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(start)
    return tour


def probabilistic_nearest_neighbour(G, start, guide='weight', invert=True):
    """
    Probabilistic nearest neighbor tour construction.
    
    Args:
        G (nx.Graph): The graph.
        start (int): Starting node.
        guide (str): Guide attribute for probabilities.
        invert (bool): Invert the probabilities if True.
    
    Returns:
        list: The generated tour.
    """
    tour = [start]

    while len(tour) < len(G.nodes):
        i = tour[-1]

        neighbours = [(j, G.edges[(i, j)][guide]) for j in G.neighbors(i) if j not in tour]

        nodes, p = zip(*neighbours)

        p = np.array(p)

        # if there are any infinite values, make these 1 and others 0
        is_inf = np.isinf(p)
        if is_inf.any():
            p = is_inf

        # if there are all 0s, make everything 1
        if np.sum(p) == 0:
            p[:] = 1.

        # if the guide should be inverted, for example, edge weight
        if invert:
            p = 1 / p

        j = np.random.choice(nodes, p=p / np.sum(p))
        tour.append(j)

    tour.append(start)
    return tour


def best_probabilistic_nearest_neighbour(G, start, n_iters, guide='weight', weight='weight'):
    """
    Find the best tour from multiple probabilistic nearest neighbor runs.
    
    Args:
        G (nx.Graph): The graph.
        start (int): Starting node.
        n_iters (int): Number of iterations.
        guide (str): Guide attribute.
        weight (str): Weight attribute for cost.
    
    Returns:
        list: Best tour found.
    """
    best_tour = None
    best_cost = 0

    for _ in range(n_iters):
        new_tour = probabilistic_nearest_neighbour(G, start, guide)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour


def cheapest_insertion(G, sub_tour, n, weight='weight'):
    """
    Insert node n into sub_tour at the cheapest position.
    
    Args:
        G (nx.Graph): The graph.
        sub_tour (list): Current sub-tour.
        n (int): Node to insert.
        weight (str): Edge weight attribute.
    
    Returns:
        list: Updated tour.
    """
    best_tour = None
    best_cost = 0

    for j in range(1, len(sub_tour)):
        new_tour = sub_tour.copy()
        new_tour.insert(j, n)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour


def insertion(G, start, mode='farthest', weight='weight'):
    """
    Construct a tour using insertion heuristic.
    
    Args:
        G (nx.Graph): The graph.
        start (int): Starting node.
        mode (str): 'random', 'nearest', or 'farthest'.
        weight (str): Edge weight attribute.
    
    Returns:
        list: Generated tour.
    """
    assert mode in ['random', 'nearest', 'farthest'], f'Unknown mode: {mode}'

    nodes = list(G.nodes)
    nodes.remove(start)
    tour = [start, start]

    while len(nodes) > 0:
        if mode == 'random':
            next_node = np.random.choice(nodes)

        else:
            next_node = None
            next_cost = 0

            for i in tour:
                for j in nodes:
                    if (mode == 'nearest' and G.edges[i, j][weight] < next_cost) or \
                            (mode == 'farthest' and G.edges[i, j][weight] > next_cost) or \
                            (next_node is None):
                        next_node = j
                        next_cost = G.edges[i, j][weight]

        nodes.remove(next_node)
        tour = cheapest_insertion(G, tour, next_node, weight)

    return tour


def local_search(init_tour, init_cost, D, first_improvement=False, t_lim=0):
    """
    Perform local search using 2-opt and relocate operators.
    
    Args:
        init_tour (list): Initial tour.
        init_cost (float): Initial cost.
        D (np.ndarray): Distance matrix.
        first_improvement (bool): Use first improvement strategy.
        t_lim (float): Time limit.
    
    Returns:
        tuple: (best_tour, best_cost, search_progress, iteration_count)
    """
    cur_tour, cur_cost = init_tour, init_cost
    search_progress = []
    cnt = 0
    improved = True
    while improved and cnt < 20 and time.time() < t_lim:

        improved = False
        for operator in [two_opt_a2a, relocate_a2a]:
            delta, new_tour = operator(cur_tour, D, first_improvement)
            delta = tour_cost2(new_tour, D) - cur_cost
            if delta < 0:
                improved = True
                cur_cost += delta
                if cur_cost != tour_cost2(new_tour, D):
                    print('Wrong cost Try again')
                cur_tour = new_tour
                search_progress.append({
                    'time': time.time(),
                    'cost': cur_cost
                })
            cnt += 1
    
    return cur_tour, cur_cost, search_progress,  cnt 


def guided_local_search(G, init_tour, init_cost, t_lim, weight='weight', guides=['weight'], perturbation_moves=30,
                        first_improvement=False):
    """
    Perform Guided Local Search (GLS) on the graph.
    
    Args:
        G (nx.Graph): The graph.
        init_tour (list): Initial tour.
        init_cost (float): Initial cost.
        t_lim (float): Time limit.
        weight (str): Weight attribute.
        guides (list): Guide attributes for penalization.
        perturbation_moves (int): Number of perturbation moves.
        first_improvement (bool): Use first improvement.
    
    Returns:
        tuple: (best_tour, best_cost, search_progress, total_iterations)
    """
    k = 0.1 * init_cost / len(G.nodes)
    nx.set_edge_attributes(G, 0, 'penalty')

    edge_weight = nx.to_numpy_array(G, weight=weight)
    cnt_ans = 0

    # Initial local search
    if time.time() >= t_lim:
        return init_tour, init_cost, [], 0
    cur_tour, cur_cost, search_progress, cnt = local_search(
        init_tour, init_cost, edge_weight, first_improvement, t_lim
    )
    cnt_ans += cnt
    best_tour, best_cost = cur_tour, cur_cost

    iter_i = 0
    while time.time() < t_lim:
        guide = guides[iter_i % len(guides)]

        # perturbation
        moves = 0
        cnt = 0
        while moves < perturbation_moves:
            if time.time() >= t_lim:
                return best_tour, best_cost, search_progress, cnt_ans

            # penalize edge
            max_util, max_util_e = -float("inf"), None
            for e in zip(cur_tour[:-1], cur_tour[1:]):
                if time.time() >= t_lim:
                    return best_tour, best_cost, search_progress, cnt_ans
                util = G[e[0]][e[1]][guide] / (1 + G[e[0]][e[1]]['penalty'])
                if util > max_util:
                    max_util, max_util_e = util, e

            # add penalty
            G[max_util_e[0]][max_util_e[1]]['penalty'] += 1.
            edge_penalties = nx.to_numpy_array(G, weight='penalty')
            edge_weight_guided = edge_weight + k * edge_penalties

            # apply operators
            for n in max_util_e:
                if time.time() >= t_lim:
                    return best_tour, best_cost, search_progress, cnt_ans
                if n != 0:  # not the start
                    i = cur_tour.index(n)
                    for operator in [two_opt_o2a, relocate_o2a]:
                        if time.time() >= t_lim:
                            return best_tour, best_cost, search_progress, cnt_ans

                        moved = False
                        delta, new_tour = operator(cur_tour, edge_weight_guided, i, first_improvement)
                        if delta < 0:
                            cur_cost = tour_cost(G, new_tour, weight)
                            cur_tour = new_tour
                            moved = True
                            search_progress.append({'time': time.time(), 'cost': cur_cost})

                        if not moved:
                            cnt += 1
                            if cnt == 2:
                                moved = True
                                cnt = 0
                                search_progress.append({'time': time.time(), 'cost': cur_cost})

                        if moved:
                            moves += 1
                        cnt_ans += 1

        # local search again
        if time.time() >= t_lim:
            return best_tour, best_cost, search_progress, cnt_ans

        cur_tour, cur_cost, new_search_progress, cnt = local_search(
            cur_tour, cur_cost, edge_weight, first_improvement, t_lim
        )
        search_progress += new_search_progress
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour, cur_cost

        iter_i += 1

    return best_tour, best_cost, search_progress, cnt_ans
