"""Synthetic-scenario evaluation: S1-S4 x 8 detectors, scored by the benchmark.

Why this script exists
----------------------
The manuscript's synthetic table (Table V) reports MDR/MTD/MTFA/MTR for every
detector on four correlation-drift scenarios, but ``synthetic/scored/`` holds
FFWIN alone and was scored under a protocol that does not match
``harness/metrics/drift.py`` -- the scorer behind every real-stream table. This
script closes that gap: it drives the same detector implementations, at the same
published hyperparameters as ``config_revision.py``, over the streams emitted by
``synthetic/generate_datasets.py``, and scores them with the benchmark's own
``calculate_drift_metrics``.

Protocol
--------
* Detection indices are stream positions, as in ``ModelOptimizer.optimize``:
  the detector sees every sample and an alarm at sample ``i`` is recorded as
  ``i``. There is no classifier in the loop, so no training prefix is reserved.
* Ground truth is the generator's ``drift_start_indices``, minus the leading
  ``0`` -- the start of the first concept is not a change point.
* Every detector runs all ten seeds. This differs on purpose from the
  real-stream run, where the three deterministic detectors ran a single seed:
  here the seed also draws the stream, so each seed is a distinct realisation
  and all detectors face the same ten.

Usage
-----
    python scripts/run_synthetic.py --out synthetic/results
    python scripts/run_synthetic.py --scenarios S2 --seeds 11 --probe
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("detector", "harness", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

from metrics.drift import calculate_drift_metrics  # noqa: E402  (harness)

# Fixed seeds, identical to config_revision.SEEDS.
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

# Paper scenario -> generator factory in synthetic/generate_datasets.py.
# S1 gradual, S2 abrupt conditional-dependence, S3 incremental, S4 recurring.
SCENARIOS = {
    "S1": ("copula", "example_copula_8d_40k"),
    "S2": ("graph_precision", "example_graph_precision_8d_40k"),
    "S3": ("dcc_dynamic", "example_dcc_8d_40k"),
    "S4": ("hmm_regimes", "example_hmm_8d_40k"),
}


def _load_generator():
    """Import synthetic/generate_datasets.py by path (it is not a package)."""
    if "generate_datasets" in sys.modules:
        return sys.modules["generate_datasets"]
    path = ROOT / "synthetic" / "generate_datasets.py"
    spec = importlib.util.spec_from_file_location("generate_datasets", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and dataclass
    # processing resolves cls.__module__ through sys.modules.
    sys.modules["generate_datasets"] = module
    spec.loader.exec_module(module)
    return module


def detector_factories(window=250, step=125):
    """The eight detectors of the main experiment, at published values.

    Mirrors ``EXPERIMENT_MAIN`` in ``scripts/config_revision.py``. That module
    is not imported directly because it pulls in the benchmark's dataset
    loaders, which require ``river``; nothing here needs it.

    ``window`` and ``step`` exist for the window-size sweep and default to the
    published ``W=250, s=125``. They move FFWIN only: the baselines keep the
    sample size they were tuned at, so a sweep never silently retunes them.
    """
    from detectors import (
        BayesianNonparametricDetectionMethod,
        ClusteredStatisticalTestDriftDetectionMethod,
        DiscriminativeDriftDetector2019,
        FasanoFranceschiniWindow,
        ImageBasedDriftDetector,
        OneClassDriftDetector,
        SemiParametricLogLikelihood,
        UDetect,
    )

    return {
        "BayesianNonparametricDetectionMethod": lambda s: BayesianNonparametricDetectionMethod(
            n_samples=250, const=1.0, max_depth=2, threshold=0.5, seed=s
        ),
        "ClusteredStatisticalTestDriftDetectionMethod": lambda s: ClusteredStatisticalTestDriftDetectionMethod(
            n_samples=250, confidence=0.01, feature_proportion=0.1, n_clusters=2, seed=s
        ),
        "DiscriminativeDriftDetector2019": lambda s: DiscriminativeDriftDetector2019(
            n_reference_samples=250, recent_samples_proportion=0.5, threshold=0.7, seed=s
        ),
        "ImageBasedDriftDetector": lambda s: ImageBasedDriftDetector(
            n_samples=250,
            n_permutations=20,
            update_interval=100,
            n_consecutive_deviations=1,
            seed=s,
        ),
        "OneClassDriftDetector": lambda s: OneClassDriftDetector(
            n_samples=250,
            threshold=0.3,
            outlier_detector_kwargs={"nu": 0.5, "kernel": "rbf", "gamma": "auto"},
            seed=s,
        ),
        "SemiParametricLogLikelihood": lambda s: SemiParametricLogLikelihood(
            n_samples=250, n_clusters=2, threshold=0.05, seed=s
        ),
        # disjoint_training_windows=False is what the real-stream results were
        # produced with; True raises no alarms on any stream, real or synthetic.
        "UDetect": lambda s: UDetect(
            n_windows=50, n_samples=250, disjoint_training_windows=False, seed=s
        ),
        "FasanoFranceschiniWindow": lambda s: FasanoFranceschiniWindow(
            n_samples=window,
            step=step,
            alpha=0.05,
            n_permutations=100,
            method="b",
            pca_threshold=9,
            pca_target_dim=8,
            pca_mode="concat",
            seed=s,
        ),
    }


def build_stream(scenario, seed):
    """Materialise one scenario realisation: samples plus ground-truth drifts."""
    gen = _load_generator()
    _, factory_name = SCENARIOS[scenario]
    stream = getattr(gen, factory_name)(seed=seed)
    samples = [x for x, _ in stream]
    starts = stream.ground_truth()["drift_start_indices"]
    drifts = [t for t in starts if t > 0]  # index 0 starts the first concept
    return samples, drifts


def run_one(scenario, seed, detector_names, raw_dir=None, window=250, step=125):
    """Run every requested detector over one realisation of one scenario.

    When ``raw_dir`` is given the rows are also written there immediately, so a
    campaign that dies after hours keeps what it has earned and can resume.
    """
    samples, known = build_stream(scenario, seed)
    factories = detector_factories(window=window, step=step)
    rows = []
    for name in detector_names:
        detector = factories[name](seed)
        started = time.perf_counter()
        alarms = [i for i, x in enumerate(samples) if detector.update(x)]
        elapsed = time.perf_counter() - started
        scores = calculate_drift_metrics(known, alarms)
        rows.append(
            {
                "scenario": scenario,
                "dataset": SCENARIOS[scenario][0],
                "detector": name,
                "seed": seed,
                "n_known_drifts": len(known),
                "n_alarms": len(alarms),
                "window": window if name == "FasanoFranceschiniWindow" else "",
                "step": step if name == "FasanoFranceschiniWindow" else "",
                "mdr": scores["mdr"],
                "mtd": scores["mtd"],
                "mtfa": scores["mtfa"],
                "mtr": scores["mtr"],
                "runtime_s": round(elapsed, 3),
                "drifts": " ".join(str(a) for a in alarms),
            }
        )
        print(
            "  {} seed={:<4} {:<46} alarms={:<5} mdr={:.2f} {:7.1f}s".format(
                scenario, seed, name, len(alarms), scores["mdr"], elapsed
            ),
            flush=True,
        )
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / "{}_seed{}.csv".format(scenario, seed)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows


FIELDS = [
    "scenario",
    "dataset",
    "detector",
    "seed",
    "n_known_drifts",
    "n_alarms",
    "window",
    "step",
    "mdr",
    "mtd",
    "mtfa",
    "mtr",
    "runtime_s",
    "drifts",
]


def read_raw(raw_dir):
    """Collect every per-realisation CSV written by ``run_one``."""
    rows = []
    for path in sorted(Path(raw_dir).glob("*_seed*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["seed"] = int(row["seed"])
                rows.append(row)
    return rows


def write_rows(out_dir, rows):
    """One CSV per (scenario, detector), mirroring results/<stream>/<detector>.csv."""
    by_file = {}
    for row in rows:
        by_file.setdefault((row["scenario"], row["detector"]), []).append(row)
    for (scenario, detector), group in sorted(by_file.items()):
        target = out_dir / scenario
        target.mkdir(parents=True, exist_ok=True)
        path = target / (detector + "_synthetic.csv")
        group.sort(key=lambda r: r["seed"])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(group)
        print("wrote {} ({} seeds)".format(path, len(group)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic/results")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--detectors", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--window",
        type=int,
        default=250,
        help="FFWIN window W; the published value is 250. Baselines are unaffected.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="FFWIN stride; defaults to W/2, as published.",
    )
    parser.add_argument("--probe", action="store_true", help="do not write CSVs")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild the per-detector CSVs from whatever is already in _raw/",
    )
    args = parser.parse_args()
    step = args.step if args.step is not None else args.window // 2

    out_dir = ROOT / args.out
    raw_dir = out_dir / "_raw"

    if args.aggregate_only:
        write_rows(out_dir, read_raw(raw_dir))
        return

    names = args.detectors or list(detector_factories(window=args.window, step=step))
    tasks = [
        (s, seed)
        for s in args.scenarios
        for seed in args.seeds
        if args.probe or not (raw_dir / "{}_seed{}.csv".format(s, seed)).exists()
    ]
    print(
        "{} realisations x {} detectors on {} workers".format(
            len(tasks), len(names), args.workers
        ),
        flush=True,
    )

    sink = None if args.probe else raw_dir
    started = time.perf_counter()
    rows = []
    if args.workers == 1:
        for scenario, seed in tasks:
            rows.extend(run_one(scenario, seed, names, sink, args.window, step))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_one, scenario, seed, names, sink, args.window, step
                ): (scenario, seed)
                for scenario, seed in tasks
            }
            for future in as_completed(futures):
                scenario, seed = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:  # keep the campaign alive, report the hole
                    print("FAILED {} seed={}: {!r}".format(scenario, seed, exc), flush=True)

    print("total {:.1f}s, {} rows".format(time.perf_counter() - started, len(rows)))
    if not args.probe:
        # Aggregate from _raw/, not from this run's rows: a resumed campaign
        # must produce the same tables as an uninterrupted one.
        write_rows(out_dir, read_raw(raw_dir))


if __name__ == "__main__":
    main()
