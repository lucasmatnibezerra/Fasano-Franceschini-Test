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
    n_greater: int,
    n_equal: int,
    n_permutations: int,
    seed: int | None = None,
    rule: str = "paper",
) -> float:
    """Calibrate an observed statistic against its permutation null distribution.

    Two conventions are available:

    ``paper`` (default)
        ``p = (1 + #{D_b >= D_obs}) / (B + 1)``, the plug-in estimator stated in
        the FFWIN paper. Deterministic given the permutations, and never zero,
        so a drift alarm always carries a finite significance level.
    ``randomized``
        ``p = (#{D_b > D_obs} + (1 + #{D_b == D_obs}) * U) / (B + 1)`` with
        ``U ~ Uniform(0, 1)``, the tie-broken variant of Puritz et al. Exact in
        the continuous sense, but not reproducible unless ``seed`` is fixed.

    :param n_greater: the number of permutation statistics strictly above D_obs
    :param n_equal: the number of permutation statistics equal to D_obs
    :param n_permutations: the number of permutations B
    :param seed: the random seed, used by the ``randomized`` rule only
    :param rule: ``'paper'`` or ``'randomized'``
    :return: the p-value
    """
    if n_permutations <= 0:
        return 1.0
    if rule == "paper":
        return (1 + n_greater + n_equal) / (n_permutations + 1)
    if rule == "randomized":
        rng = random.Random(seed)
        return (n_greater + (1 + n_equal) * rng.random()) / (1 + n_permutations)
    raise ValueError("rule must be 'paper' or 'randomized'")


def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    verbose: bool = True,
    seed: int | None = None,
    method: str = "b",
    rule: str = "paper",
):
    """Run the permutation test comparing two multivariate samples.

    :param X: the first sample
    :param Y: the second sample
    :param n_permutations: the number of permutations B
    :param verbose: True to show a progress bar
    :param seed: the random seed
    :param method: 'b' for brute force or 'r' for the range tree
    :param rule: the p-value convention, see :func:`permutation_test_pvalue`
    :return: the counts of permutation statistics above and equal to the
        observed one, and the p-value
    """
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
    p_value = permutation_test_pvalue(
        z_greater, z_equal, n_permutations, seed, rule=rule
    )
    return z_greater, z_equal, p_value


def _permuted_statistic(
    M: np.ndarray, n1: int, n2: int, method: str, seed: int | None
) -> int:
    """Compute the test statistic for one random relabelling of the pooled sample.

    Defined at module level so that it can be pickled and sent to a worker
    process.

    :param M: the pooled sample
    :param n1: the size of the first sample
    :param n2: the size of the second sample
    :param method: 'b' for brute force or 'r' for the range tree
    :param seed: the seed for this permutation
    :return: the statistic under the relabelling
    """
    rng = random.Random(seed)
    order = list(range(n1 + n2))
    rng.shuffle(order)
    return test_statistic(M, n1, n2, shuffle=False, method=method, s=order)


def permutation_test_parallel(
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    method: str = "b",
    seed: int | None = None,
    n_jobs: int = -1,
    rule: str = "paper",
):
    """Run the permutation test with the permutations spread across workers.

    Equivalent to :func:`permutation_test`, but each permutation is drawn from
    its own seed so that the workers do not all reproduce the same relabelling.

    :param X: the first sample
    :param Y: the second sample
    :param n_permutations: the number of permutations B
    :param method: 'b' for brute force or 'r' for the range tree
    :param seed: the base seed; permutation i uses seed + i + 1
    :param n_jobs: the number of workers, -1 for all cores
    :param rule: the p-value convention, see :func:`permutation_test_pvalue`
    :return: the counts of permutation statistics above and equal to the
        observed one, and the p-value
    """
    from joblib import Parallel, delayed

    M = rbind(X, Y)
    n1, n2 = X.shape[0], Y.shape[0]
    Z_obs = test_statistic(M, n1, n2, shuffle=False, seed=seed, method=method)

    seeds = [None if seed is None else seed + i + 1 for i in range(n_permutations)]
    values = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_statistic)(M, n1, n2, method, s) for s in seeds
    )

    z_greater = sum(1 for v in values if v > Z_obs)
    z_equal = sum(1 for v in values if v == Z_obs)
    p_value = permutation_test_pvalue(
        z_greater, z_equal, n_permutations, seed, rule=rule
    )
    return z_greater, z_equal, p_value


def compute_D(X: np.ndarray, Y: np.ndarray, method: str = "b") -> int:
    return ff_test_statistic(X, Y, method)
