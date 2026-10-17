"""Launch the revision run as parallel shards and report progress.

Work is sharded by (stream, seed group) so that the eleven streams can occupy
all cores instead of running one after another. Poker Hands is 829k samples —
roughly 63% of the total cost — so it is split one seed per shard, while the
smaller streams take all ten seeds in a single shard.

Each shard is an independent process writing to results/<Stream>/, so shards
never collide. A shard that has already produced its rows is skipped, which
makes the whole run resumable: rerun this script after an interruption and only
the missing work executes.

Usage:
    python launch.py                 # run everything
    python launch.py --workers 10
    python launch.py --dry-run       # print the plan and exit
    python launch.py --status        # summarise what has completed so far

    # the revised FFWIN alone (s = W), as experiment revision_stride
    python launch.py --experiment revision_stride --models ffwin_stride
"""

from __future__ import annotations

import argparse
import csv

# The 'drifts' column holds every alarm timestamp; on Poker Hands one detector
# emitted 18,683 of them, well past the csv module's default field limit.
_ = csv.field_size_limit(min(2**31 - 1, __import__('sys').maxsize))
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The benchmark checkout: ``bench/`` next to this script on the machine that ran
# the campaign, ``harness/`` in the replication package.
BENCH = HERE / "bench" if (HERE / "bench").is_dir() else HERE.parent / "harness"


def _python() -> Path:
    for root in (HERE, HERE.parent):
        for rel in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
            candidate = root.joinpath(*rel)
            if candidate.exists():
                return candidate
    return Path(sys.executable)


PYTHON = _python()

EXPERIMENT = "revision_main"
MODELS: list[str] = []          # detector keys of harness/config.py; empty = all
# Model key -> detector class, so --models restricts what counts as "done".
KEY_CLASS = {
    "ffwin": "FasanoFranceschiniWindow", "ffwin_stride": "FasanoFranceschiniWindow",
    "ffwin_pca": "FasanoFranceschiniWindow", "spll": "SemiParametricLogLikelihood",
    "d3": "DiscriminativeDriftDetector2019", "ibdd": "ImageBasedDriftDetector",
    "csddm": "ClusteredStatisticalTestDriftDetectionMethod",
    "ocdd": "OneClassDriftDetector", "udetect": "UDetect", "udetect_joint": "UDetect",
    "edfs": "EDFS", "bndm": "BayesianNonparametricDetectionMethod",
}
ALL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
# Detectors that carry no randomness run a single seed; the rest run all ten.
DETERMINISTIC = {"OneClassDriftDetector", "UDetect",
                 "BayesianNonparametricDetectionMethod"}
STOCHASTIC = {"FasanoFranceschiniWindow", "SemiParametricLogLikelihood",
              "DiscriminativeDriftDetector2019", "ImageBasedDriftDetector",
              "ClusteredStatisticalTestDriftDetectionMethod"}

# Measured cost of one FFWIN seed, in minutes, from a timed slice of each
# stream. Used only to order shards longest-first and to print an estimate.
COST = {
    "PokerHand": 131.6,
    "InsectsIncrementalAbruptBalanced": 14.7,
    "InsectsIncrementalReoccurringBalanced": 14.7,
    "RialtoBridgeTimelapse": 13.6,
    "InsectsIncrementalBalanced": 10.5,
    "InsectsAbruptBalanced": 9.7,
    "Electricity": 6.6,
    "InsectsGradualBalanced": 4.4,
    "Powersupply": 3.1,
    "NOAAWeather": 2.9,
    "OutdoorObjects": 0.7,
}

# Streams split one seed per shard because a single shard would dominate the
# wall clock otherwise.
SPLIT_BY_SEED = {"PokerHand"}


def build_shards() -> list[dict]:
    shards = []
    for stream in sorted(COST, key=lambda s: -COST[s]):
        if stream in SPLIT_BY_SEED:
            for seed in ALL_SEEDS:
                shards.append({
                    "stream": stream, "seeds": [seed],
                    "minutes": COST[stream],
                    "label": f"{stream}:seed{seed}",
                })
        else:
            shards.append({
                "stream": stream, "seeds": list(ALL_SEEDS),
                "minutes": COST[stream] * len(ALL_SEEDS),
                "label": stream,
            })
    return sorted(shards, key=lambda s: -s["minutes"])


def seeds_done(stream: str) -> set[int]:
    """
    Seeds fully covered for a stream.

    A seed counts as done when every stochastic detector has produced a row for
    it and every deterministic detector has produced its single row. Requiring
    all ten seeds from a deterministic detector would never be satisfied.

    :param stream: the stream class name
    :return: the set of covered seeds
    """
    directory = BENCH / "results" / stream
    if not directory.is_dir():
        return set()

    by_detector = {}
    for path in directory.glob(f"*_{EXPERIMENT}.csv"):
        name = path.stem[: -len(f"_{EXPERIMENT}")]
        seeds = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("seed"):
                    seeds.add(int(row["seed"]))
        by_detector[name] = seeds

    deterministic, stochastic = DETERMINISTIC, STOCHASTIC
    if MODELS:
        wanted = {KEY_CLASS[k] for k in MODELS}
        deterministic, stochastic = deterministic & wanted, stochastic & wanted
    for name in deterministic:
        if not by_detector.get(name):
            return set()
    if not stochastic.issubset(by_detector):
        return set()
    if not stochastic:
        return set(ALL_SEEDS)
    return set.intersection(*(by_detector[n] for n in stochastic))


def run_shard(shard: dict) -> tuple[dict, int, str]:
    env = dict(os.environ)
    env["FFWIN_STREAMS"] = shard["stream"]
    env["FFWIN_SEEDS"] = ",".join(str(s) for s in shard["seeds"])
    if MODELS:
        env["FFWIN_MODELS"] = ",".join(MODELS)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[var] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{shard['label'].replace(':', '_')}.log"

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(PYTHON), "main.py", EXPERIMENT],
            cwd=BENCH, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    elapsed = (time.time() - started) / 60
    return shard, process.returncode, f"{elapsed:.1f} min"


def status() -> None:
    print(f"{'stream':<40s} {'seeds done':>12s}")
    print("-" * 54)
    for stream in sorted(COST):
        done = seeds_done(stream)
        marker = "complete" if len(done) == len(ALL_SEEDS) else f"{len(done)}/10"
        print(f"{stream:<40s} {marker:>12s}")


def main() -> None:
    global EXPERIMENT, MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run completed shards")
    parser.add_argument("--experiment", default=EXPERIMENT,
                        help="experiment name, i.e. the suffix of the results CSVs")
    parser.add_argument("--models", default="",
                        help="comma-separated detector keys of harness/config.py (default: all)")
    args = parser.parse_args()

    EXPERIMENT = args.experiment
    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in MODELS if m not in KEY_CLASS]
    if unknown:
        parser.error("unknown model key(s): " + ", ".join(unknown))

    if args.status:
        status()
        return

    shards = build_shards()
    if not args.force:
        pending = []
        for shard in shards:
            done = seeds_done(shard["stream"])
            if not all(seed in done for seed in shard["seeds"]):
                pending.append(shard)
        skipped = len(shards) - len(pending)
        if skipped:
            print(f"skipping {skipped} shard(s) already complete")
        shards = pending

    if not shards:
        print("nothing to do; every shard has results")
        return

    serial = sum(s["minutes"] for s in shards) / 60
    print(f"{len(shards)} shard(s), {args.workers} workers")
    print(f"FFWIN cost if run serially: {serial:.1f} h")
    print(f"longest single shard: {max(s['minutes'] for s in shards) / 60:.1f} h "
          "(this is the wall-clock floor)")
    print("baselines add to this; they are much cheaper than FFWIN\n")
    for shard in shards:
        print(f"  {shard['label']:<40s} {shard['minutes'] / 60:5.1f} h")

    if args.dry_run:
        return

    print(f"\nstarting at {time.strftime('%H:%M:%S')}\n", flush=True)
    started = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_shard, s): s for s in shards}
        for done, future in enumerate(as_completed(futures), start=1):
            shard, code, elapsed = future.result()
            state = "ok" if code == 0 else f"FAILED ({code})"
            if code != 0:
                failures.append(shard["label"])
            print(f"[{done}/{len(shards)}] {shard['label']:<40s} {state:>12s} "
                  f"{elapsed:>10s}  (elapsed {(time.time() - started) / 3600:.1f} h)",
                  flush=True)

    print(f"\nfinished in {(time.time() - started) / 3600:.1f} h")
    if failures:
        print("failed shards (see logs/):")
        for label in failures:
            print(f"  {label}")
        sys.exit(1)
    print(f"\nnext: python make_tables.py --experiment {EXPERIMENT} --out tables")


if __name__ == "__main__":
    main()
