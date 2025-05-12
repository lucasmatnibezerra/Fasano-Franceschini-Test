# ff_core.py

import numpy as np
import random
from matrix_util import rbind
from distance import build_range_trees, brute_distance, range_distance

def test_statistic(
    M: np.ndarray,
    n1: int,
    n2: int,
    shuffle: bool = False,
    seed: int = None,
    method: str = 'b'
) -> int:
    """
    Compute D = d1 + d2 exactly as in the R/C++ code.

    d1 = max_{x in S1} D0(x)
    d2 = max_{y in S2} D0(y)
    """
    total = n1 + n2
    # build a list of indices and optionally shuffle
    s = list(range(total))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(s)

    # pre-build range-trees if requested
    if method == 'r':
        tree1, tree2 = build_range_trees(M, n1, n2, s)

    # compute d1 over origins in the first sample
    d1 = 0
    for i in range(n1):
        if method == 'r':
            val = range_distance(tree1, tree2, n1, n2, M[s[i]])
        else:
            val = brute_distance(M, n1, n2, s, i)
        if val > d1:
            d1 = val

    # compute d2 over origins in the second sample
    d2 = 0
    for j in range(n2):
        if method == 'r':
            val = range_distance(tree1, tree2, n1, n2, M[s[n1 + j]])
        else:
            val = brute_distance(M, n1, n2, s, n1 + j)
        if val > d2:
            d2 = val

    return d1 + d2

def ff_test_statistic(X: np.ndarray, Y: np.ndarray, method: str = 'b') -> int:
    """
    Wrapper to compute D without shuffling.
    """
    M = rbind(X, Y)
    return test_statistic(M, X.shape[0], Y.shape[0], shuffle=False, method=method)

def permutation_test_pvalue(
    z_less: int,
    z_equal: int,
    n_permutations: int,
    seed: int = None
) -> float:
    """
    Compute p-value using permutation test correction.
    """
    rng = random.Random(seed)
    return (z_less + (1 + z_equal) * rng.random()) / (1 + n_permutations)

def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    verbose: bool = True,
    seed: int = None,
    method: str = 'b'
):
    """
    Serial permutation test.

    Returns
    -------
    z_greater : int
        Count of permuted statistics > observed.
    z_equal : int
        Count of permuted statistics == observed.
    p_value : float
    """
    M = rbind(X, Y)
    n1 = X.shape[0]
    Z_obs = test_statistic(M, n1, Y.shape[0], shuffle=False, seed=seed, method=method)

    z_greater = 0
    z_equal = 0
    it = range(n_permutations)
    if verbose:
        from tqdm import tqdm
        it = tqdm(it, desc="Permutations")

    rng = random.Random(seed)
    for _ in it:
        s = list(range(len(M)))
        rng.shuffle(s)
        M_perm = M[s]
        Zp = test_statistic(M_perm, n1, Y.shape[0], shuffle=False, seed=seed, method=method)
        if Zp > Z_obs:
            z_greater += 1
        elif Zp == Z_obs:
            z_equal += 1

    p_value = permutation_test_pvalue(z_greater, z_equal, n_permutations, seed)
    return z_greater, z_equal, p_value

def permutation_test_parallel(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    method: str = 'b',
    seed: int = None
):
    """
    Parallel permutation test using multiprocessing.

    Returns
    -------
    z_greater : int
    z_equal : int
    p_value : float
    """
    from multiprocessing import Pool, cpu_count
    def _worker(_):
        zg, ze, _ = permutation_test(X, Y, 1, verbose=False, seed=seed, method=method)
        return zg, ze

    with Pool(cpu_count()) as pool:
        results = pool.map(_worker, range(n_permutations))

    z_greater = sum(r[0] for r in results)
    z_equal = sum(r[1] for r in results)
    p_value = permutation_test_pvalue(z_greater, z_equal, n_permutations, seed)
    return z_greater, z_equal, p_value
