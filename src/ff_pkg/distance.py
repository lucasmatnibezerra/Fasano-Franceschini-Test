import numpy as np
from typing import List
from .matrix_util import get_row
from .range_tree import RangeTree

from numba import njit  # mandatory dependency


@njit(fastmath=True, cache=True)
def _brute_distance_numba(M, n1, n2, s, origin_ix):
    total, d = M.shape
    a = M[s[origin_ix]]
    n_orth = 1 << d
    c1 = np.zeros(n_orth, np.int64)
    c2 = np.zeros(n_orth, np.int64)
    for k in range(total):
        if k == origin_ix:
            continue
        pt = M[s[k]]
        code = 0
        valid = True
        for j in range(d):
            if pt[j] > a[j]:
                code |= 1 << j
            elif pt[j] < a[j]:
                pass
            else:
                valid = False
                break
        if not valid:
            continue
        if k < n1:
            c1[code] += 1
        else:
            c2[code] += 1
    best = 0
    for code in range(n_orth):
        diff = abs(n2 * c1[code] - n1 * c2[code])
        if diff > best:
            best = diff
    return best


def brute_distance(M, n1, n2, s, origin_ix):
    return _brute_distance_numba(M, n1, n2, np.asarray(s, dtype=np.int64), origin_ix)


def range_distance(tree1, tree2, n1, n2, origin):
    dim = origin.shape[0]
    inf = np.inf
    strict = [False] * dim
    d = 0
    for mask in range(1 << dim):
        lower, upper = [], []
        for j in range(dim):
            if mask & (1 << (dim - 1 - j)):
                lower.append(-inf)
                upper.append(origin[j])
            else:
                lower.append(origin[j])
                upper.append(inf)
        c1 = tree1.count_in_range(lower, upper, strict, strict)
        c2 = tree2.count_in_range(lower, upper, strict, strict)
        diff = abs(n2 * c1 - n1 * c2)
        if diff > d:
            d = diff
    return d


def build_range_trees(M: np.ndarray, r1: int, r2: int, s: List[int]):
    """Build two RangeTree instances (S1, S2) according to permutation s.

    Keeps the original API used by ff_core.
    """
    idx1 = s[:r1]
    idx2 = s[r1 : r1 + r2]
    pts1 = [M[i] for i in idx1]
    pts2 = [M[i] for i in idx2]
    return RangeTree(pts1), RangeTree(pts2)
