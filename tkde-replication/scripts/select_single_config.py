"""One configuration per detector for all four synthetic scenarios.

Why this script exists
----------------------
``tune_synthetic.py select`` picks the best configuration separately for each
scenario, using the drift labels of that scenario. No deployed unsupervised
detector can do that: it does not know which kind of drift is coming. This
script asks the deployable question instead. For each detector it selects the
single configuration with the highest mean detection F1 over all four
scenarios x three tuning seeds, then scores that one configuration on the
held-out seeds of every scenario.

Selection uses only the tuning runs already stored by ``tune_synthetic.py``
(``synthetic/tuning/tune/_raw``); a configuration is eligible only if all
twelve of its tuning runs exist. Ties go to the smaller window, then to grid
order, as in ``tune_synthetic.py``. Held-out runs that ``tune_synthetic.py
eval`` already produced for the same (detector, cfg, scenario, seed) are
copied rather than rerun.

Usage
-----
    python scripts/select_single_config.py select            # selection only
    python scripts/select_single_config.py eval --workers 30 # + held-out runs, CSV, table
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("detector", "harness", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

import tune_synthetic as ts  # noqa: E402
from metrics.drift import calculate_drift_metrics  # noqa: E402

PHASE = "eval_single_delta{}"
SCENARIOS = ["S1", "S2", "S3", "S4"]
FF = "FasanoFranceschiniWindow"


def _tag(offset):
    return "_offset" if offset else ""


def _tune_records(offset):
    """Tuning runs; with ``offset``, FFWIN's come from the shifted streams."""
    records = ts._read("tune")
    if offset:
        records = [r for r in records if r["detector"] != FF] + ts._read("tune_offset")
    return records


def select(delta, offset=False):
    scores = {}
    params = {}
    for r in _tune_records(offset):
        key = (r["detector"], r["cfg"])
        f1, _, _ = ts.detection_f1(r["known"], r["alarms"], delta)
        scores.setdefault(key, []).append(f1)
        params[key] = r["params"]
    best = {}
    for (name, cfg), values in sorted(scores.items()):
        if len(values) < len(SCENARIOS) * len(ts.TUNE_SEEDS):
            continue
        mean = sum(values) / len(values)
        rank = (mean, -ts._window(params[name, cfg]), -cfg)
        if name not in best or rank > best[name]["rank"]:
            best[name] = {"rank": rank, "cfg": cfg, "f1_tune": mean, "params": params[name, cfg]}
    selected = [{"detector": n, "cfg": b["cfg"], "f1_tune": b["f1_tune"], "params": b["params"]}
                for n, b in sorted(best.items())]
    out = ts.OUT / "selected_single{}_delta{}.json".format(_tag(offset), delta)
    out.write_text(json.dumps(selected, indent=1), encoding="utf-8")
    for s in selected:
        print("{:<46} cfg{:03d} F1_tune={:.3f} {}".format(s["detector"], s["cfg"], s["f1_tune"], s["params"]))
    print("wrote", out)
    return selected


def cmd_select(args):
    select(args.delta, args.offset)


def cmd_eval(args):
    selected = select(args.delta, args.offset)
    if args.offset:
        # Only FFWIN is shifted; the baselines keep their unshifted runs.
        _eval_offset_ffwin([s for s in selected if s["detector"] == FF][0], args)
        return
    phase = PHASE.format(args.delta)
    per_scenario = "eval_delta{}".format(args.delta)
    tasks, copied = [], 0
    for s in selected:
        for scenario in SCENARIOS:
            for seed in ts.EVAL_SEEDS:
                target = ts._raw_path(phase, s["detector"], s["cfg"], scenario, seed)
                if target.exists():
                    continue
                source = ts._raw_path(per_scenario, s["detector"], s["cfg"], scenario, seed)
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    copied += 1
                    continue
                tasks.append((phase, s["detector"], s["cfg"], s["params"], scenario, seed))
    print("copied {} held-out runs from {}".format(copied, per_scenario), flush=True)
    ts._execute(tasks, args.workers)

    chosen = {(s["detector"], s["cfg"]) for s in selected}
    rows = []
    for r in ts._read(phase):
        if (r["detector"], r["cfg"]) not in chosen:
            continue
        f1, p, rec = ts.detection_f1(r["known"], r["alarms"], args.delta)
        m = calculate_drift_metrics(r["known"], r["alarms"])
        rows.append({"scenario": r["scenario"], "detector": r["detector"], "cfg": r["cfg"],
                     "seed": r["seed"], "n_alarms": len(r["alarms"]), "f1": f1,
                     "precision": p, "recall": rec, **m, "runtime_s": r["runtime_s"],
                     "params": json.dumps(r["params"])})
    out_csv = ts.OUT / "{}.csv".format(phase)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scenario"], r["detector"], r["seed"])))
    print("wrote", out_csv)
    render_table(out_csv, args.delta)


PER_SCENARIO_CAPTION = r"with the best configuration per scenario, an oracle "
SINGLE_CAPTION = r"with one configuration per detector "


OFFSET_PHASE = "eval_single_offset_delta{}"


def _eval_offset_ffwin(s, args):
    phase = OFFSET_PHASE.format(args.delta)
    per_scenario = "eval_offset_delta{}".format(args.delta)
    tasks, copied = [], 0
    for scenario in SCENARIOS:
        for seed in ts.EVAL_SEEDS:
            target = ts._raw_path(phase, FF, s["cfg"], scenario, seed)
            source = ts._raw_path(per_scenario, FF, s["cfg"], scenario, seed)
            if target.exists():
                continue
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied += 1
            else:
                tasks.append((phase, FF, s["cfg"], s["params"], scenario, seed))
    print("copied {} shifted held-out runs from {}".format(copied, per_scenario), flush=True)
    ts._execute(tasks, args.workers)
    cmd_table(args)


# The caption only flags the offset; the paper's text (Sec. VI-A) explains it.
OFFSET_NOTE = r"; \ac{FFWIN} on streams shifted by a random offset"


def cmd_table(args):
    """Table with the baselines' unshifted runs and FFWIN's shifted ones."""
    base = ts.OUT / "{}.csv".format(PHASE.format(args.delta))
    with base.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["detector"] != FF]
    ffwin = [s for s in select(args.delta, offset=True) if s["detector"] == FF][0]
    for r in ts._read(OFFSET_PHASE.format(args.delta)):
        if r["cfg"] != ffwin["cfg"]:
            continue
        f1, p, rec = ts.detection_f1(r["known"], r["alarms"], args.delta)
        m = calculate_drift_metrics(r["known"], r["alarms"])
        rows.append({"scenario": r["scenario"], "detector": FF,
                     "cfg": ffwin["cfg"], "seed": r["seed"], "n_alarms": len(r["alarms"]),
                     "f1": f1, "precision": p, "recall": rec, **m,
                     "runtime_s": r["runtime_s"], "params": json.dumps(ffwin["params"]),
                     "offset": r["offset"]})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out_csv = ts.OUT / "{}.csv".format(OFFSET_PHASE.format(args.delta))
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["scenario"], r["detector"], int(r["seed"]))))
    print("wrote", out_csv)
    render_table(out_csv, args.delta, note=OFFSET_NOTE)


def render_table(csv_path, delta, note=""):
    import make_tuning_table as mt

    data = mt.load(csv_path)
    n_seeds = max(len({r["seed"] for r in rows}) for rows in data.values())
    tex = mt.render(data, delta, n_seeds)
    for old, new in (
        (PER_SCENARIO_CAPTION, SINGLE_CAPTION),
        (r"\label{tab:synth_tuned}", r"\label{tab:synth_single}"),
        (r"make_tuning_table.py", r"select_single_config.py"),
        (r"alarms per stream).", r"alarms per stream" + note + ")."),
    ):
        if old not in tex:
            raise SystemExit("caption layout of make_tuning_table.py changed; update " + __file__)
        tex = tex.replace(old, new)
    out = ROOT / "tables" / "table_synth_single.tex"
    out.write_text(tex, encoding="utf-8", newline="\n")
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["select", "eval", "table"])
    parser.add_argument("--delta", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--offset", action="store_true",
                        help="select/eval FFWIN on randomly shifted streams (see tune_synthetic.py)")
    args = parser.parse_args()
    {"select": cmd_select, "eval": cmd_eval, "table": cmd_table}[args.command](args)


if __name__ == "__main__":
    main()
