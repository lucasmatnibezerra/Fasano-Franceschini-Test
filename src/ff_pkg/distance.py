import numpy as np
import itertools
from .matrix_util import get_row
from .range_tree import RangeTree

def build_range_trees(M: np.ndarray, r1: int, r2: int, s: list[int]):
    pts1 = [get_row(M, s[i]) for i in range(r1)]
    pts2 = [get_row(M, s[i + r1]) for i in range(r2)]
    return RangeTree(pts1), RangeTree(pts2)

def brute_distance(M, n1, n2, s, origin_ix):
    origin = get_row(M, s[origin_ix])
    d = 0
    dim = origin.shape[0]
    for signs in itertools.product((-1,1), repeat=dim):
        mask1 = np.ones(n1, bool)
        mask2 = np.ones(n2, bool)
        for j,sign in enumerate(signs):
            if sign<0:
                mask1 &= (M[:n1,j] < origin[j])
                mask2 &= (M[n1:,j] < origin[j])
            else:
                mask1 &= (M[:n1,j] > origin[j])
                mask2 &= (M[n1:,j] > origin[j])
        c1 = mask1.sum()
        c2 = mask2.sum()
        diff = abs(n2*c1 - n1*c2)
        if diff> d:
            d = diff
    return d

def range_distance(tree1, tree2, n1, n2, origin):
    dim = origin.shape[0]
    inf = np.inf
    strict = [False]*dim
    d = 0
    for mask in range(1<<dim):
        lower, upper = [], []
        for j in range(dim):
            if mask & (1<<(dim-1-j)):
                lower.append(-inf); upper.append(origin[j])
            else:
                lower.append(origin[j]); upper.append(inf)
        c1 = tree1.count_in_range(lower, upper, strict, strict)
        c2 = tree2.count_in_range(lower, upper, strict, strict)
        diff = abs(n2*c1 - n1*c2)
        if diff > d:
            d = diff
    return d
