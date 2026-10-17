"""FFWIN on the held-out seeds with the stream shifted by a random offset.

Why this script exists
----------------------
With s = W the detector compares windows on a fixed grid (multiples of W from
the start of the stream), and the drift times of S2, S3 and S4 fall on
multiples of 1,000. At W = 1,000 every comparison then sets a pure old-regime
window against a pure new-regime one, the best case, and every detection lands
at exactly t + 999. No real stream lines its drifts up with the detector's
grid. This control drops the first offset samples of each realisation, with
offset drawn from a seeded RNG in [1, 999] per (scenario, seed), and shifts
the drift times with it. The other detectors react sample by sample (their
detection delays take 35-63 distinct values) and need no such control.

Usage
-----
    python scripts/offset_control.py OUT.jsonl --scenarios S1 S2 S3 S4 --alpha 0.01
"""
import argparse, json, os, sys, time
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path[:0] = ["harness", "detector", "scripts"]
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import run_synthetic as rs
from tune_synthetic import detection_f1
from detectors import FasanoFranceschiniWindow as F
from metrics.drift import calculate_drift_metrics

FIX = dict(n_permutations=100, method="b", pca_threshold=9, pca_target_dim=8, pca_mode="concat")


def run(args):
    sc, seed, cfg = args
    off = int(np.random.default_rng([seed, int(sc[1])]).integers(1, 1000))
    X, known = rs.build_stream(sc, seed)
    X = X[off:]; known = [t - off for t in known if t - off > 0]
    d = F(seed=seed, **cfg, **FIX)
    t0 = time.time(); al = [i for i, x in enumerate(X) if d.update(x)]
    out = {"scenario": sc, "seed": seed, "offset": off, "known": known, "alarms": al,
           "f1": {D: detection_f1(known, al, D)[0] for D in (500, 1000, 2000)},
           "mdr": calculate_drift_metrics(known, al)["mdr"], "params": cfg, "runtime_s": round(time.time() - t0)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    cfg = dict(n_samples=a.window, step=a.window, alpha=a.alpha)
    jobs = [(sc, s, cfg) for sc in a.scenarios for s in (44, 55, 66, 77, 88, 99, 110)]
    with open(a.out, "w") as fh, ProcessPoolExecutor(a.workers) as p:
        for r in p.map(run, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["scenario"], r["seed"], r["offset"], "F1@500/1000/2000", {k: round(v, 2) for k, v in r["f1"].items()}, "mdr", round(r["mdr"], 2), "alarms", len(r["alarms"]), flush=True)
