import time

import networkx as nx
import numpy as np

from . import tour_cost, operators

from gnngls import operators, tour_cost, tour_cost2

def nearest_neighbor(G, depot, weight='weight'):
    tour = [depot]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    return tour


def probabilistic_nearest_neighbour(G, depot, guide='weight', invert=True):
    tour = [depot]

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

    tour.append(depot)
    return tour


def best_probabilistic_nearest_neighbour(G, depot, n_iters, guide='weight', weight='weight'):
    best_tour = None
    best_cost = 0

    for _ in range(n_iters):
        new_tour = probabilistic_nearest_neighbour(G, depot, guide)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour


def cheapest_insertion(G, sub_tour, n, weight='weight'):
    best_tour = None
    best_cost = 0

    for j in range(1, len(sub_tour)):
        new_tour = sub_tour.copy()
        new_tour.insert(j, n)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour


def insertion(G, depot, mode='farthest', weight='weight'):
    assert mode in ['random', 'nearest', 'farthest'], f'Unknown mode: {mode}'

    nodes = list(G.nodes)
    nodes.remove(depot)
    tour = [depot, depot]

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

def compute_tour_cost(tour, adjacency_matrix):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += adjacency_matrix[tour[i], tour[i + 1]]  # Subtract 1 to convert 1-indexed to 0-indexed
    return cost

def local_search(init_tour, init_cost, D, first_improvement=False, t_lim=0):
    cur_tour, cur_cost = init_tour, init_cost
    search_progress = []
    cnt = 0
    improved = True
    while improved and cnt < 20 and time.time() < t_lim:

        improved = False
        for operator in [operators.two_opt_a2a, operators.relocate_a2a]:
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
    k = 0.1 * init_cost / len(G.nodes)
    nx.set_edge_attributes(G, 0, 'penalty')

    edge_weight, _ = nx.attr_matrix(G, weight)
    cnt_ans = 0
    cur_tour, cur_cost, search_progress, cnt = local_search(init_tour, init_cost, edge_weight, first_improvement, t_lim)
    cnt_ans += cnt
    best_tour, best_cost = cur_tour, cur_cost
    iter_i = 0
    while time.time() < t_lim:
        guide = guides[iter_i % len(guides)]  # option change guide ever iteration (as in KGLS)

        # perturbation
        moves = 0
        cnt = 0
        while moves < perturbation_moves and time.time() < t_lim:
            # penalize edge
            max_util = 0
            max_util_e = None
            for e in zip(cur_tour[:-1], cur_tour[1:]):
                util = G[e[0]][e[1]][guide] / (1 + G[e[0]][e[1]]['penalty'])
                
                if util > max_util or max_util_e is None:
                    max_util = util
                    max_util_e = e
            G[max_util_e[0]][max_util_e[1]]['penalty'] += 1.
            edge_penalties, _ = nx.attr_matrix(G, 'penalty')
            edge_weight_guided = edge_weight + k * edge_penalties
            # apply operator to edge
            for n in max_util_e:
                if time.time() >= t_lim:
                    break
                if n != 0:  # not the depot
                    i = cur_tour.index(n)

                    for operator in [operators.two_opt_o2a, operators.relocate_o2a]:
                        moved = False

                        delta, new_tour = operator(cur_tour, edge_weight_guided, i, first_improvement)
                        if delta < 0:
                            cur_cost = tour_cost(G, new_tour, weight)
                            cur_tour = new_tour
                            moved = True

                            search_progress.append({
                                'time': time.time(),
                                'cost': cur_cost
                            })
                        if moved == False:
                            cnt += 1
                            if cnt == 2:
                                moved = True
                                cnt = 0
                                search_progress.append({
                                'time': time.time(),
                                'cost': cur_cost
                                })
                        moves += moved
                        cnt_ans += 1
            
        # optimisation
        if time.time() < t_lim:
            cur_tour, cur_cost, new_search_progress, cnt = local_search(cur_tour, cur_cost, edge_weight, first_improvement, t_lim)
            search_progress += new_search_progress
            if cur_cost < best_cost:
                best_tour, best_cost = cur_tour, cur_cost

        iter_i += 1
    
    cnt_ans += cnt

    return best_tour, best_cost, search_progress, cnt_ans