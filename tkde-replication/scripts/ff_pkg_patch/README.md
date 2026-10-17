# Fixes for `Fasano-Franceschini-Test`

Three defects found while building the revision harness, all validated against
the repository at `main`. Apply with:

```bash
cd Fasano-Franceschini-Test
git apply /path/to/0001-fix-pvalue-parallel-deps.patch
```

The individual fixed files are also in this folder if you prefer to copy them
over (`ff_core.py` goes to `src/ff_pkg/`).

## 1. The p-value did not match the published equation

The paper states

```
p = (1 + #{D_b >= D_obs}) / (B + 1)
```

while `permutation_test_pvalue` computed the randomized, tie-broken variant of
Puritz et al.,

```
p = (#{D_b > D_obs} + (1 + #{D_b == D_obs}) * U) / (B + 1),   U ~ Uniform(0,1)
```

These give different numbers, and a reviewer comparing the paper against the
public repository would see it. The function now implements the paper's rule by
default and keeps the randomized one behind `rule="randomized"` for comparison.
`permutation_test` and `permutation_test_parallel` both take the same `rule`
argument and pass it through.

Beyond matching the paper, the plug-in estimator is the safer default: it is
deterministic given the permutations, and it can never return exactly zero, so
every alarm carries a finite significance level.

## 2. `permutation_test_parallel` could not run, and would have been wrong

Two separate problems:

- It passed a locally-defined `_worker` closure to `multiprocessing.Pool.map`.
  Local functions are not picklable, so the call raised immediately.
- Every worker was handed **the same** `seed`. Had it run, all `B` workers would
  have produced the identical relabelling and therefore the identical statistic,
  collapsing the null distribution to a single point. Verified on the current
  code: six calls at a fixed seed return `[1560, 1560, 1560, 1560, 1560, 1560]`.

The permutation worker is now a module-level function, `_permuted_statistic`,
and permutation *i* draws from `seed + i + 1`, so the relabellings differ:
`[1320, 1260, 1140, 1200, 1320, 1380]`. Parallelism uses `joblib`, already a
dependency, with an `n_jobs` argument.

Nothing in the published results depends on this: the serial path was the one
used. But the function is public API and would fail for anyone who called it.

## 3. `rtree` and `numba` were undeclared mandatory dependencies

`ff_pkg.range_tree` imports `rtree` and `numba` at module level, `ff_pkg.distance`
imports `range_tree`, and `ff_pkg.ff_core` imports `distance`. So importing
anything from the package pulls in the full range-tree stack — even a
brute-force (`method="b"`) run, which never touches it.

Three files disagreed about this:

| File | Problem |
|---|---|
| `requirements.txt` | listed `Rtree`, `numba`, `scikit-learn` under `# Optional` |
| `setup.py` | omitted `numba` and `scikit-learn` entirely |
| `pyproject.toml` | listed them correctly, but declared `packages = ["ffwin"]` when the importable package is `ff_pkg` under `src/`, so it packaged nothing |

The net effect: `pip install` of the repository left the package unimportable.
All three now agree, declare the real dependencies, and point at `src/ff_pkg`.
`scikit-learn` is included because `ff_pkg.dim_reduction` needs it for the PCA
projection, which the paper's default configuration uses on any stream with
`d >= 9`.

Note that `rtree` also requires the libspatialindex C library, which pip ships
as a wheel on the usual platforms but which can need a system package elsewhere.
