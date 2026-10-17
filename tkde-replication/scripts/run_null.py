"""False alarms on drift-free streams.

Why this script exists
----------------------
Every alarm on a stream without drift is a false alarm, so a stationary stream
measures each detector's false-alarm behaviour with nothing else in the way.
FFWIN tests each window pair with a permutation test, so its per-test false
alarm probability is at most alpha by construction; the baselines' thresholds
carry no such guarantee. This script puts a number on both.

Null stream
-----------
One stationary Gaussian-copula stream per seed (seeds 11-110), d = 8,
N = 40,000, from the same generator as the synthetic scenarios: a single
``CopulaRegime`` with the block correlation ``make_R(0.35)``. A single moderate
block correlation keeps the marginals N(0, 1) and the dependence non-trivial,
exactly like one regime of the scenarios, while never changing. One family of
null stream suffices: under the null, what matters is that the stream is
stationary, not which regime it sits in, and building one per scenario would
multiply the most expensive runs (FFWIN at W = 1000) by four.

Detectors
---------
All eight at the configuration they use on the real streams
(``run_synthetic.detector_factories``), plus FFWIN at W = 1000, s = 500,
alpha = 0.05 (the overlapping configuration a first tuning pass selected) and at W in {500, 1000} with s = W and
alpha in {0.05, 0.01}.

For FFWIN the number of window comparisons is recorded, so the observed false
alarms can be set against alpha x comparisons.

Usage
-----
    python scripts/run_null.py run --workers 30
    python scripts/run_null.py report
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("detector", "harness", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

import run_synthetic as rs  # noqa: E402

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
N = 40_000
OUT = ROOT / "synthetic" / "null"
FFWIN_FIXED = dict(
    n_permutations=100, method="b", pca_threshold=9, pca_target_dim=8, pca_mode="concat"
)
# name -> (W, s, alpha) for the FFWIN variants beyond the published one
FFWIN_VARIANTS = {
    # The real-stream configuration (added 2026-10-03, after a reviewer asked
    # whether the configuration used on the real streams is calibrated too).
    "FFWIN_W250_s250_a0.05": (250, 250, 0.05),
    "FFWIN_W1000_s500_a0.05": (1000, 500, 0.05),
    "FFWIN_W500_s500_a0.05": (500, 500, 0.05),
    "FFWIN_W500_s500_a0.01": (500, 500, 0.01),
    "FFWIN_W1000_s1000_a0.05": (1000, 1000, 0.05),
    "FFWIN_W1000_s1000_a0.01": (1000, 1000, 0.01),
}
PUBLISHED_FFWIN = "FasanoFranceschiniWindow"


def null_stream(seed):
    gen = rs._load_generator()
    regime = gen.CopulaRegime(R=gen.make_R(0.35), n=N, transition=None, name="null")
    stream = gen.GaussianCopulaStream([regime], seed=seed)
    return [x for x, _ in stream]


def detector_names():
    return list(rs.detector_factories()) + list(FFWIN_VARIANTS)


def params_of(name):
    if name in FFWIN_VARIANTS:
        w, s, a = FFWIN_VARIANTS[name]
        return dict(n_samples=w, step=s, alpha=a, **FFWIN_FIXED)
    if name == PUBLISHED_FFWIN:
        return dict(n_samples=250, step=125, alpha=0.05, **FFWIN_FIXED)
    return None


def make(name, seed):
    if name in FFWIN_VARIANTS:
        from detectors import FasanoFranceschiniWindow

        return FasanoFranceschiniWindow(seed=seed, **params_of(name))
    return rs.detector_factories()[name](seed)


def raw_path(name, seed):
    return OUT / "_raw" / name / "seed{}.json".format(seed)


def run_task(name, seed):
    path = raw_path(name, seed)
    if path.exists():  # another invocation finished it meanwhile
        return json.loads(path.read_text(encoding="utf-8"))
    samples = null_stream(seed)
    det = make(name, seed)
    t0 = time.perf_counter()
    alarms = [i for i, x in enumerate(samples) if det.update(x)]
    elapsed = time.perf_counter() - t0
    history = getattr(det, "history", None)
    record = {
        "detector": name, "seed": seed, "n": len(samples), "alarms": alarms,
        "n_tests": len(history) if isinstance(history, list) else None,
        "alpha": getattr(det, "alpha", None) if history is not None else None,
        "params": params_of(name), "runtime_s": round(elapsed, 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(path)
    return record


def _cost(name):
    # SPLL first (hours per run), then FFWIN by window, then the cheap baselines.
    if name == "SemiParametricLogLikelihood":
        return -10**9
    p = params_of(name)
    return -(p["n_samples"] * 10 if p else 0)


def cmd_run(args):
    tasks = [(n, s) for n in detector_names() for s in SEEDS if not raw_path(n, s).exists()]
    tasks.sort(key=lambda t: _cost(t[0]))
    print("{} runs on {} workers".format(len(tasks), args.workers), flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            name, seed = futures[fut]
            try:
                r = fut.result()
                print("[{}/{}] {} seed={} alarms={} tests={} {:.1f}s".format(
                    i, len(tasks), name, seed, len(r["alarms"]), r["n_tests"], r["runtime_s"]), flush=True)
            except Exception as exc:
                print("FAILED {} seed={}: {!r}".format(name, seed, exc), flush=True)
    print("total {:.1f}s".format(time.perf_counter() - t0), flush=True)


LABELS = {
    "BayesianNonparametricDetectionMethod": "BNDM",
    "ClusteredStatisticalTestDriftDetectionMethod": "CSDDM",
    "DiscriminativeDriftDetector2019": "D3",
    "ImageBasedDriftDetector": "IBDD",
    "OneClassDriftDetector": "OCDD",
    "SemiParametricLogLikelihood": "SPLL",
    "UDetect": "UDetect",
    PUBLISHED_FFWIN: "FFWIN $W{=}250$, $s{=}125$, $\\alpha{=}0.05$",
}
for _n, (_w, _s, _a) in FFWIN_VARIANTS.items():
    LABELS[_n] = "FFWIN $W{{=}}{}$, $s{{=}}{}$, $\\alpha{{=}}{:.2f}$".format(_w, _s, _a)


def _ms(values):
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def cmd_report(args):
    rows = []
    for name in detector_names():
        recs = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((OUT / "_raw" / name).glob("seed*.json"))]
        if not recs:
            continue
        fa = [len(r["alarms"]) for r in recs]
        m, sd = _ms(fa)
        tests = [r["n_tests"] for r in recs if r["n_tests"] is not None]
        expected = None
        if tests and recs[0]["alpha"] is not None:
            expected = statistics.fmean(t * recs[0]["alpha"] for t in tests)
        rows.append({
            "detector": name, "label": LABELS.get(name, name), "n_seeds": len(recs),
            "false_alarms_mean": round(m, 3), "false_alarms_sd": round(sd, 3),
            "per_10k": round(m * 10_000 / recs[0]["n"], 3),
            "n_tests_mean": round(statistics.fmean(tests), 1) if tests else "",
            "expected_alpha_x_tests": round(expected, 3) if expected is not None else "",
            "runtime_s_mean": round(statistics.fmean(r["runtime_s"] for r in recs), 1),
        })
    out_csv = ROOT / "tables" / "table_null.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv)

    lines = [
        r"% Generated by scripts/run_null.py report -- do not edit by hand.",
        r"\begin{table}[!tb]",
        r"\centering",
        r"\caption{False alarms on drift-free synthetic streams (mean $\pm$ sd over "
        + str(max(r["n_seeds"] for r in rows))
        + r" seeds). Expected: $\alpha$ times the number of comparisons.}",
        r"\label{tab:null}",
        r"\tablestyle",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"\textbf{Detector} & \textbf{Alarms} & \textbf{Per 10k} & \textbf{Expected} \\",
        r"\midrule",
    ]
    for r in rows:
        exp = "--" if r["expected_alpha_x_tests"] == "" else "{:.1f}".format(r["expected_alpha_x_tests"])
        lines.append("{} & {:.1f} $\\pm$ {:.1f} & {:.2f} & {} \\\\".format(
            r["label"], r["false_alarms_mean"], r["false_alarms_sd"], r["per_10k"], exp))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex = ROOT / "tables" / "table_null.tex"
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", out_tex)
    for r in rows:
        print("{:<46} FA {:6.1f} +- {:5.1f}  per10k {:6.2f}  tests {}  exp {}".format(
            r["detector"], r["false_alarms_mean"], r["false_alarms_sd"], r["per_10k"],
            r["n_tests_mean"], r["expected_alpha_x_tests"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
