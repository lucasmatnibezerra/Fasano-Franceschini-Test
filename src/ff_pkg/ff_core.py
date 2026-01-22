import numpy as np
import random
from .matrix_util import rbind
from .distance import build_range_trees, brute_distance, range_distance


def test_statistic(
    M: np.ndarray,
    n1: int,
    n2: int,
    shuffle: bool = False,
    seed: int | None = None,
    method: str = "b",
    s: list | None = None,
) -> int:
    total = n1 + n2
    if s is None:
        s = list(range(total))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(s)
    if method == "r":
        tree1, tree2 = build_range_trees(M, n1, n2, s)
    d1 = 0
    for i in range(n1):
        val = (
            range_distance(tree1, tree2, n1, n2, M[s[i]])
            if method == "r"
            else brute_distance(M, n1, n2, s, i)
        )
        if val > d1:
            d1 = val
    d2 = 0
    for j in range(n2):
        idx = n1 + j
        val = (
            range_distance(tree1, tree2, n1, n2, M[s[idx]])
            if method == "r"
            else brute_distance(M, n1, n2, s, idx)
        )
        if val > d2:
            d2 = val
    return d1 + d2


def ff_test_statistic(X: np.ndarray, Y: np.ndarray, method: str = "b") -> int:
    M = rbind(X, Y)
    return test_statistic(M, X.shape[0], Y.shape[0], shuffle=False, method=method)


def permutation_test_pvalue(
    z_less: int, z_equal: int, n_permutations: int, seed: int | None = None
) -> float:
    rng = random.Random(seed)
    return (z_less + (1 + z_equal) * rng.random()) / (1 + n_permutations)


def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    verbose: bool = True,
    seed: int | None = None,
    method: str = "b",
):
    M = rbind(X, Y)
    n1, n2 = X.shape[0], Y.shape[0]
    Z_obs = test_statistic(M, n1, n2, shuffle=False, seed=seed, method=method)
    z_greater = 0
    z_equal = 0
    it = range(n_permutations)
    if verbose and n_permutations > 0:
        from tqdm import tqdm

        it = tqdm(it, desc="Permutations")
    rng = random.Random(seed)
    total = n1 + n2
    base_indices = list(range(total))
    for _ in it:
        rng.shuffle(base_indices)
        Zp = test_statistic(
            M, n1, n2, shuffle=False, seed=seed, method=method, s=base_indices
        )
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
    method: str = "b",
    seed: int | None = None,
):
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


def compute_D(X: np.ndarray, Y: np.ndarray, method: str = "b") -> int:
    return ff_test_statistic(X, Y, method)
