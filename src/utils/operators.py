import itertools
import numpy as np


def two_opt(tour, i, j):
    """
    Perform a 2-opt swap on the tour between indices i and j.
    
    Args:
        tour (list): The current tour.
        i (int): First index for the swap.
        j (int): Second index for the swap.
    
    Returns:
        list: The new tour after the 2-opt swap.
    """
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j - 1:i - 1:-1] + tour[j:]


def two_opt_cost(tour, D, i, j):
    """
    Calculate the cost change from a 2-opt swap.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        i (int): First index.
        j (int): Second index.
    
    Returns:
        float: Delta cost from the swap.
    """
    if i == j:
        return 0
    elif j < i:
        i, j = j, i

    a = tour[i]
    b = tour[i - 1]
    c = tour[j]
    d = tour[j - 1]

    delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
    return delta


def two_opt_a2a(tour, D, first_improvement=False):
    """
    All-to-all 2-opt search for improvements.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        first_improvement (bool): Stop at first improvement if True.
    
    Returns:
        tuple: (delta, new_tour)
    """
    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for i, j in itertools.combinations(idxs, 2):
        if abs(i - j) < 2:
            continue

        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour


def two_opt_o2a(tour, D, i, first_improvement=False):
    """
    One-to-all 2-opt search starting from index i.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        i (int): Fixed index.
        first_improvement (bool): Stop at first improvement if True.
    
    Returns:
        tuple: (delta, new_tour)
    """
    assert i > 0 and i < len(tour) - 1

    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if abs(i - j) < 2:
            continue

        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, tour


def relocate(tour, i, j):
    """
    Relocate node at i to position j in the tour.
    
    Args:
        tour (list): The current tour.
        i (int): Index to relocate from.
        j (int): Index to relocate to.
    
    Returns:
        list: New tour after relocation.
    """
    new_tour = tour.copy()
    n = new_tour.pop(i)
    new_tour.insert(j, n)
    return new_tour


def relocate_cost(tour, D, i, j):
    """
    Calculate the cost change from relocating i to j.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        i (int): From index.
        j (int): To index.
    
    Returns:
        float: Delta cost.
    """
    if i == j:
        return 0

    a = tour[i - 1]
    b = tour[i]
    c = tour[i + 1]
    if i < j:
        d = tour[j]
        e = tour[j + 1]
    else:
        d = tour[j - 1]
        e = tour[j]

    delta = -D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]
    return delta


def relocate_o2a(tour, D, i, first_improvement=False):
    """
    One-to-all relocation search from i.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        i (int): Fixed index.
        first_improvement (bool): Stop at first improvement if True.
    
    Returns:
        tuple: (delta, new_tour)
    """
    assert i > 0 and i < len(tour) - 1

    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for j in idxs:
        if i == j:
            continue

        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour


def relocate_a2a(tour, D, first_improvement=False):
    """
    All-to-all relocation search.
    
    Args:
        tour (list): The current tour.
        D (np.ndarray): Distance matrix.
        first_improvement (bool): Stop at first improvement if True.
    
    Returns:
        tuple: (delta, new_tour)
    """
    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for i, j in itertools.permutations(idxs, 2):
        if i - j == 1:  # e.g. relocate 2 -> 3 == relocate 3 -> 2
            continue

        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0, tour