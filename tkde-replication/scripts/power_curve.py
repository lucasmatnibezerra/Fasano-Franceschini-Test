"""Oracle-aligned power of the FF test against window size, per scenario.

Separates two things the streaming runs conflate: whether the test can see a
scenario's drift at all, and whether the sliding window happens to line up with
it. Here the windows are placed by hand on both sides of each prescribed drift,
so a failure is a power failure and nothing else.

Reports, per scenario and per W, the permutation p-value at each drift and the
effect size that goes with it (the change in the correlation matrix), plus a
same-regime pair as the null control.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("detector", "harness", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

from ff_pkg.ff_core import test_statistic
from ff_pkg.matrix_util import rbind

WINDOWS = [250, 500, 1000]
SEEDS = [11, 22, 33]
SCENARIOS = [
    ("S1 copula (gradual)", "example_copula_8d_40k"),
    ("S2 graph_precision (abrupt)", "example_graph_precision_8d_40k"),
    ("S3 dcc_dynamic (incremental)", "example_dcc_8d_40k"),
    ("S4 hmm_regimes (recurring)", "example_hmm_8d_40k"),
]
ALPHA = 0.05
B = 100


def load_generator():
    path = ROOT / "synthetic" / "generate_datasets.py"
    spec = importlib.util.spec_from_file_location("generate_datasets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_datasets"] = module
    spec.loader.exec_module(module)
    return module


def pvalue(a, b, seed=0):
    M = rbind(a, b)
    n1, n2 = len(a), len(b)
    observed = test_statistic(M, n1, n2)
    rng = np.random.default_rng(seed)
    ge = sum(
        test_statistic(M, n1, n2, shuffle=True, seed=int(rng.integers(1 << 31)))
        >= observed
        for _ in range(B)
    )
    return (1 + ge) / (B + 1)


def effect_size(ref, cur):
    delta = np.abs(np.corrcoef(ref.T) - np.corrcoef(cur.T))
    off = delta[~np.eye(delta.shape[0], dtype=bool)]
    return off.mean(), delta.max()


def main():
    gen = load_generator()
    print("alpha={} B={} seeds={}".format(ALPHA, B, SEEDS))
    for title, factory_name in SCENARIOS:
        print("\n=== {} ===".format(title))
        per_w = {w: [] for w in WINDOWS}
        nulls = {w: [] for w in WINDOWS}
        sizes = []
        for seed in SEEDS:
            stream = getattr(gen, factory_name)(seed=seed)
            samples = [x for x, _ in stream]
            drifts = [t for t in stream.ground_truth()["drift_start_indices"] if t > 0]
            X = np.array(
                [[row["x{}".format(i + 1)] for i in range(8)] for row in samples]
            )
            for W in WINDOWS:
                started = time.perf_counter()
                for t in drifts:
                    if t - W < 0 or t + W > len(X):
                        continue
                    ref, cur = X[t - W : t], X[t : t + W]
                    per_w[W].append(pvalue(ref, cur, seed=seed))
                    if W == WINDOWS[0]:
                        sizes.append(effect_size(ref, cur)[0])
                # one null pair per seed and W, mid-regime
                mid = drifts[0] // 2
                if mid - W >= 0:
                    nulls[W].append(pvalue(X[mid - W : mid], X[mid : mid + W], seed=seed))
                print(
                    "  seed={} W={} done in {:.0f}s".format(
                        seed, W, time.perf_counter() - started
                    ),
                    flush=True,
                )
        print("  mean effect size |dcorr| = {:.3f}".format(float(np.mean(sizes))))
        for W in WINDOWS:
            ps = np.array(per_w[W])
            null = np.array(nulls[W])
            print(
                "  W={:<5} detected {:>2}/{:<2} ({:.0%}) at alpha={}  median p={:.3f}"
                "   null median p={:.3f}".format(
                    W,
                    int((ps <= ALPHA).sum()),
                    len(ps),
                    (ps <= ALPHA).mean() if len(ps) else float("nan"),
                    ALPHA,
                    float(np.median(ps)) if len(ps) else float("nan"),
                    float(np.median(null)) if len(null) else float("nan"),
                )
            )


if __name__ == "__main__":
    main()
