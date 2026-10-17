"""Per-detector hyperparameter search on the synthetic scenarios S1-S4.

Why this script exists
----------------------
``run_synthetic.py`` evaluates every detector at the configuration published
for the real streams. That answers "what happens if you deploy the detector
unchanged", but it cannot tell a detector that is blind to correlation drift
from one that is merely mistuned for it, and it gave FFWIN a window sweep the
baselines never received. This script gives every detector the same treatment.

Protocol
--------
* Search space: for the seven baselines, the grid of the upstream benchmark
  (``unsupervised-concept-drift-detection/config.py``), copied verbatim, with
  both UDetect variants. For FFWIN, W in {250, 500, 1000} x alpha in {0.05,
  0.01} x stride {W/2, W, W/4} (Table III of the paper), every other axis at
  its published value. The strides W and W/4 and the OCDD values nu < 0.5
  were added in a second pass; see ``grids``.
* Selection: every configuration runs on the tuning seeds (11, 22, 33). The
  best configuration per (detector, scenario) is the one with the highest mean
  detection F1 on those seeds; ties go to the smaller window, then to grid
  order. This is an oracle per scenario, i.e. an optimistic ceiling for each
  baseline, not a deployable setting.
* Evaluation: the selected configuration runs on the held-out seeds (44-110),
  which never influence selection.
* Detection F1 with tolerance ``delta``: a known drift at ``t`` is detected by
  the first alarm in ``[t, t + delta)``; that alarm is a true positive and
  every other alarm is a false positive. Recall = detected drifts / drifts,
  precision = true positives / alarms, F1 = 0 when there are no alarms. Unlike
  the benchmark's MDR, which credits any alarm before the next drift, F1 does
  not reward a detector that fires all the time.

Alarm positions are stored for every run, so F1 at any ``delta`` and the
benchmark's four metrics can be recomputed without rerunning anything.

Random offset (``--offset``)
----------------------------
With s = W, FFWIN compares windows on a fixed grid of multiples of W, and the
drift times of S2-S4 fall on multiples of 1,000: at W = 1,000 every comparison
then pits a pure old-regime window against a pure new-regime one. ``--offset``
drops the first ``offset`` samples of each realisation, drawn from a seeded RNG
in [1, 999] per (scenario, seed) exactly as ``offset_control.py`` does, shifts
the drift times with it, and keeps its runs in separate phases (``tune_offset``,
``eval_offset_delta<d>``). The other detectors react sample by sample and are
run without it.

Usage
-----
    python scripts/tune_synthetic.py tune  --workers 60
    python scripts/tune_synthetic.py probe --workers 60       # cost of the grid
    python scripts/tune_synthetic.py select --delta 1000
    python scripts/tune_synthetic.py eval  --delta 1000 --workers 60
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

# One BLAS/OpenMP thread per worker. Without this, 60 workers each spawn a
# thread pool per core and the first tuning pass ran at load 260 on 64 cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
for sub in ("detector", "harness", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

import run_synthetic as rs  # noqa: E402
from metrics.drift import calculate_drift_metrics  # noqa: E402

TUNE_SEEDS = [11, 22, 33]
EVAL_SEEDS = [44, 55, 66, 77, 88, 99, 110]
OUT = ROOT / "synthetic" / "tuning"

FFWIN_FIXED = dict(
    n_permutations=100, method="b", pca_threshold=9, pca_target_dim=8, pca_mode="concat"
)


def _product(**axes):
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def grids():
    """Detector name -> list of kwargs (without seed)."""
    ocdd_kwargs = {"nu": 0.5, "kernel": "rbf", "gamma": "auto"}
    udetect = _product(
        n_windows=[25, 50, 100], n_samples=[50, 100, 250, 500], disjoint_training_windows=[True]
    ) + _product(
        n_windows=[50, 100, 250], n_samples=[100, 250, 500, 1000], disjoint_training_windows=[False]
    )
    ffwin = [
        dict(n_samples=w, step=w // 2, alpha=a, **FFWIN_FIXED)
        for w in (250, 500, 1000)
        for a in (0.05, 0.01)
    ]
    # Extension (2026-09-30): the strides s = W and s = W/4 of the paper's
    # declared search space (Table III), which the first pass left out.
    # Appended so that the indices of the configurations already run hold.
    ffwin += [
        dict(n_samples=w, step=w // div, alpha=a, **FFWIN_FIXED)
        for div in (1, 4)
        for w in (250, 500, 1000)
        for a in (0.05, 0.01)
    ]
    # Extension (2026-09-30): the upstream grid fixes nu = 0.5. A one-class SVM
    # rejects about a fraction nu of its own training data, so the outlier rate
    # sits near 0.5 >= every threshold in the grid and OCDD fires once per
    # window on stationary data. Smaller nu gives the detector a chance to
    # stay quiet. Appended for the same reason as above.
    ocdd = [
        dict(c, outlier_detector_kwargs=ocdd_kwargs)
        for c in _product(n_samples=[100, 250, 500, 1000], threshold=[0.2, 0.3, 0.4, 0.5])
    ] + [
        dict(c, outlier_detector_kwargs=dict(ocdd_kwargs, nu=nu))
        for nu in (0.1, 0.05)
        for c in _product(n_samples=[100, 250, 500, 1000], threshold=[0.2, 0.3, 0.4, 0.5])
    ]
    return {
        "BayesianNonparametricDetectionMethod": _product(
            n_samples=[100, 250, 500, 1000], const=[0.5, 1.0], max_depth=[2, 3],
            threshold=[0.45, 0.5, 0.55],
        ),
        "ClusteredStatisticalTestDriftDetectionMethod": _product(
            n_samples=[100, 250, 500, 1000], confidence=[0.1, 0.01],
            feature_proportion=[0.1, 0.01], n_clusters=[2, 3],
        ),
        "DiscriminativeDriftDetector2019": _product(
            n_reference_samples=[50, 125, 250, 500], recent_samples_proportion=[0.1, 0.5, 1.0],
            threshold=[0.6, 0.7, 0.8],
        ),
        "ImageBasedDriftDetector": _product(
            n_samples=[100, 250, 500, 1000], n_permutations=[10, 20, 40],
            update_interval=[50, 100, 250], n_consecutive_deviations=[1, 4],
        ),
        "OneClassDriftDetector": ocdd,
        "SemiParametricLogLikelihood": _product(
            n_samples=[100, 250, 500, 1000], n_clusters=[2, 3], threshold=[0.05, 0.005],
        ),
        "UDetect": udetect,
        "FasanoFranceschiniWindow": ffwin,
    }


def _window(kwargs):
    return kwargs.get("n_samples", kwargs.get("n_reference_samples", 0))


def detection_f1(known, alarms, delta):
    alarms = sorted(alarms)
    hits = 0
    for t in known:
        if any(t <= a < t + delta for a in alarms):
            hits += 1
    if not alarms or not known:
        return 0.0, 0.0, 0.0
    recall = hits / len(known)
    precision = hits / len(alarms)
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


@lru_cache(maxsize=4)
def _stream(scenario, seed):
    return rs.build_stream(scenario, seed)


def stream_offset(scenario, seed):
    """Seeded offset in [1, 999]; identical to offset_control.py."""
    import numpy as np

    return int(np.random.default_rng([seed, int(scenario[1])]).integers(1, 1000))


def _shifted(scenario, seed):
    samples, known = _stream(scenario, seed)
    off = stream_offset(scenario, seed)
    return samples[off:], [t - off for t in known if t - off > 0], off


def _make(name, kwargs, seed):
    import detectors

    return getattr(detectors, name)(seed=seed, **kwargs)


def _raw_path(phase, name, cfg, scenario, seed):
    return OUT / phase / "_raw" / name / "cfg{:03d}".format(cfg) / "{}_seed{}.json".format(scenario, seed)


def run_task(phase, name, cfg, kwargs, scenario, seed, write=True):
    offset = 0
    if "offset" in phase:
        samples, known, offset = _shifted(scenario, seed)
    else:
        samples, known = _stream(scenario, seed)
    detector = _make(name, kwargs, seed)
    started = time.perf_counter()
    alarms = [i for i, x in enumerate(samples) if detector.update(x)]
    elapsed = time.perf_counter() - started
    record = {
        "phase": phase, "detector": name, "cfg": cfg, "params": kwargs,
        "scenario": scenario, "seed": seed, "known": known, "alarms": alarms,
        "runtime_s": round(elapsed, 3), "offset": offset,
    }
    if write:
        path = _raw_path(phase, name, cfg, scenario, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)
    return record


def _execute(tasks, workers, write=True):
    # Longest windows first, so the slow tail starts early.
    tasks = sorted(tasks, key=lambda t: -_window(t[3]))
    print("{} runs on {} workers".format(len(tasks), workers), flush=True)
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, *t, write=write): t for t in tasks}
        for future in as_completed(futures):
            phase, name, cfg, kwargs, scenario, seed = futures[future]
            done += 1
            try:
                rec = future.result()
                print("[{}/{}] {} cfg{:03d} {} seed={} alarms={} {:.1f}s".format(
                    done, len(tasks), name, cfg, scenario, seed, len(rec["alarms"]),
                    rec["runtime_s"]), flush=True)
            except Exception as exc:  # keep the campaign alive, report the hole
                print("FAILED {} cfg{:03d} {} seed={}: {!r}".format(name, cfg, scenario, seed, exc),
                      flush=True)
    print("total {:.1f}s".format(time.perf_counter() - started), flush=True)


def _read(phase):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((OUT / phase / "_raw").rglob("*.json"))]


def _phase(base, args):
    return base + "_offset" if args.offset else base


def cmd_tune(args):
    phase = _phase("tune", args)
    tasks = []
    for name, configs in grids().items():
        if args.detectors and name not in args.detectors:
            continue
        for cfg, kwargs in enumerate(configs):
            for scenario in args.scenarios:
                for seed in TUNE_SEEDS:
                    if not _raw_path(phase, name, cfg, scenario, seed).exists():
                        tasks.append((phase, name, cfg, kwargs, scenario, seed))
    _execute(tasks, args.workers)


def cmd_probe(args):
    """Time the most expensive configuration of each detector on 2000 samples."""
    import numpy as np

    samples, _ = _stream("S2", 11)
    samples = samples[:2000]
    total = 0.0
    for name, configs in grids().items():
        costs = []
        for cfg in sorted(range(len(configs)), key=lambda c: -_window(configs[c]))[:3]:
            det = _make(name, configs[cfg], 11)
            t0 = time.perf_counter()
            for x in samples:
                det.update(x)
            costs.append(time.perf_counter() - t0)
        # Extrapolate: worst-of-3 per 2000 samples, 40000 samples, every config and tuning run.
        per_run = max(costs) * 20
        runs = len(configs) * len(TUNE_SEEDS) * 4
        total += per_run * runs
        print("{:<46} configs={:<3} worst run ~{:7.0f}s  upper bound {:6.1f} core-h".format(
            name, len(configs), per_run, per_run * runs / 3600), flush=True)
    print("upper bound total {:.0f} core-h = {:.1f} h on {} workers".format(
        total / 3600, total / 3600 / args.workers, args.workers))


def _score(records, delta):
    """(detector, scenario, cfg) -> mean F1 plus the params."""
    table = {}
    for r in records:
        f1, _, _ = detection_f1(r["known"], r["alarms"], delta)
        key = (r["detector"], r["scenario"], r["cfg"])
        entry = table.setdefault(key, {"params": r["params"], "f1": []})
        entry["f1"].append(f1)
    return table


def cmd_select(args):
    table = _score(_read(_phase("tune", args)), args.delta)
    best = {}
    for (name, scenario, cfg), entry in sorted(table.items()):
        if len(entry["f1"]) < len(TUNE_SEEDS):
            continue  # incomplete configuration, never selected
        mean = sum(entry["f1"]) / len(entry["f1"])
        key = (name, scenario)
        rank = (mean, -_window(entry["params"]), -cfg)
        if key not in best or rank > best[key]["rank"]:
            best[key] = {"rank": rank, "cfg": cfg, "f1_tune": mean, "params": entry["params"]}
    out = OUT / "selected{}_delta{}.json".format("_offset" if args.offset else "", args.delta)
    out.write_text(json.dumps(
        [{"detector": n, "scenario": s, "cfg": b["cfg"], "f1_tune": b["f1_tune"],
          "params": b["params"]} for (n, s), b in sorted(best.items())], indent=1),
        encoding="utf-8")
    for (n, s), b in sorted(best.items()):
        print("{} {:<46} cfg{:03d} F1_tune={:.3f} {}".format(s, n, b["cfg"], b["f1_tune"], b["params"]))
    print("wrote", out)


def cmd_eval(args):
    tag = "_offset" if args.offset else ""
    selected = json.loads((OUT / "selected{}_delta{}.json".format(tag, args.delta)).read_text(encoding="utf-8"))
    phase = "eval{}_delta{}".format(tag, args.delta)
    tasks = [(phase, s["detector"], s["cfg"], s["params"], s["scenario"], seed)
             for s in selected for seed in EVAL_SEEDS
             if not _raw_path(phase, s["detector"], s["cfg"], s["scenario"], seed).exists()]
    _execute(tasks, args.workers)
    # _raw also keeps runs of configurations selected by an earlier grid.
    chosen = {(s["detector"], s["scenario"], s["cfg"]) for s in selected}
    rows = []
    for r in _read(phase):
        if (r["detector"], r["scenario"], r["cfg"]) not in chosen:
            continue
        f1, p, rec = detection_f1(r["known"], r["alarms"], args.delta)
        m = calculate_drift_metrics(r["known"], r["alarms"])
        rows.append({"scenario": r["scenario"], "detector": r["detector"], "cfg": r["cfg"],
                     "seed": r["seed"], "n_alarms": len(r["alarms"]), "f1": f1,
                     "precision": p, "recall": rec, **m, "runtime_s": r["runtime_s"],
                     "params": json.dumps(r["params"])})
    out = OUT / "{}.csv".format(phase)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scenario"], r["detector"], r["seed"])))
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tune", "probe", "select", "eval"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--delta", type=int, default=1000)
    parser.add_argument("--scenarios", nargs="+", default=list(rs.SCENARIOS))
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--offset", action="store_true",
                        help="shift each realisation by a seeded random offset (FFWIN grid control)")
    args = parser.parse_args()
    {"tune": cmd_tune, "probe": cmd_probe, "select": cmd_select, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
