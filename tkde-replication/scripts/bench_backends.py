"""Measure the cost of the Fasano-Franceschini statistic: brute force vs range tree.

Reviewer 3 and the Associate Editor both asked for a concrete answer on
computational feasibility, the paper reporting roughly 12 minutes for a single
stream. The range-tree backend is implemented in ff_pkg but the published
experiments used brute force only, so the cheaper option was never measured.

This script times one window comparison over a grid of (backend, k, W, B) on
real data, and verifies that both backends return the same statistic, which they
must, since they compute the same quantity by different means.

Output: a CSV of timings and a ready-to-include LaTeX table.

Usage:
    python bench_backends.py --data datasets/files/INSECTS-abrupt_balanced_norm.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ff_pkg.dim_reduction import PCAReducer
from ff_pkg.ff_core import test_statistic
from ff_pkg.matrix_util import rbind


def load_windows(path: str, window: int, step: int, offset: int) -> tuple:
    """
    Read a stream and cut one reference/current window pair out of it.

    :param path: the CSV file, last column assumed to be the label
    :param window: the window length W
    :param step: the stride s
    :param offset: where in the stream to cut
    :return: the reference and current windows as float arrays
    """
    frame = pd.read_csv(path)
    features = frame.iloc[:, :-1].to_numpy(dtype=float)
    needed = offset + step + window
    if features.shape[0] < needed:
        raise SystemExit(
            f"stream has {features.shape[0]} rows, need {needed} for this offset"
        )
    reference = features[offset : offset + window]
    current = features[offset + step : offset + step + window]
    return reference, current


def time_once(reference: np.ndarray, current: np.ndarray, method: str,
              n_permutations: int, rng: np.random.Generator) -> tuple:
    """
    Time one full comparison: the observed statistic plus B permutations.

    :return: the observed statistic, the time for it, and the time for the
        permutation loop
    """
    pooled = rbind(reference, current)
    n1, n2 = reference.shape[0], current.shape[0]

    start = time.perf_counter()
    observed = test_statistic(pooled, n1, n2, shuffle=False, method=method)
    t_statistic = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_permutations):
        order = rng.permutation(n1 + n2).tolist()
        test_statistic(pooled, n1, n2, shuffle=False, method=method, s=order)
    t_permutations = time.perf_counter() - start

    return observed, t_statistic, t_permutations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="stream CSV")
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=125)
    parser.add_argument("--offset", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--ks", type=int, nargs="*", default=[2, 4, 6, 8, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stream-length", type=int, default=52_848,
                        help="used to extrapolate the full-stream runtime")
    parser.add_argument("--out", default="tables")
    args = parser.parse_args()

    reference_raw, current_raw = load_windows(
        args.data, args.window, args.step, args.offset
    )
    d = reference_raw.shape[1]
    n_windows = max(1, (args.stream_length - args.window - args.step) // args.step + 1)
    print(f"d = {d}, W = {args.window}, s = {args.step}, B = {args.permutations}")
    print(f"full stream = {args.stream_length} samples -> {n_windows} comparisons\n")

    records: List[dict] = []
    for k in args.ks:
        if k >= d:
            print(f"skipping k={k}: not below d={d}")
            continue
        reducer = PCAReducer(threshold=1, target_dim=k, random_state=0,
                             fit_mode="concat")
        reference, current, _ = reducer.apply(reference_raw, current_raw)

        for method in ("b", "r"):
            statistics, totals = [], []
            for repeat in range(args.repeats):
                rng = np.random.default_rng(1000 + repeat)
                observed, t_stat, t_perm = time_once(
                    reference, current, method, args.permutations, rng
                )
                statistics.append(observed)
                totals.append(t_stat + t_perm)

            mean_total = float(np.mean(totals))
            records.append({
                "k": k,
                "method": method,
                "D": statistics[0],
                "seconds_per_window": mean_total,
                "seconds_per_window_std": float(np.std(totals, ddof=1))
                if len(totals) > 1 else 0.0,
                "minutes_full_stream": mean_total * n_windows / 60.0,
            })
            print(f"  k={k:<3d} method={method}  D={statistics[0]:<6d} "
                  f"{mean_total:7.3f} s/window  "
                  f"{mean_total * n_windows / 60.0:7.1f} min/stream")

        brute = [r for r in records if r["k"] == k and r["method"] == "b"]
        tree = [r for r in records if r["k"] == k and r["method"] == "r"]
        if brute and tree and brute[0]["D"] != tree[0]["D"]:
            print(f"  WARNING k={k}: backends disagree, "
                  f"{brute[0]['D']} != {tree[0]['D']}. Both compute the same "
                  f"statistic, so this is a bug worth reporting.")

    results = pd.DataFrame(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "backend_timings.csv", index=False)

    rows = []
    for k in sorted(results["k"].unique()):
        subset = results[results["k"] == k]
        brute = subset[subset["method"] == "b"]
        tree = subset[subset["method"] == "r"]
        if brute.empty or tree.empty:
            continue
        b, t = brute.iloc[0], tree.iloc[0]
        rows.append(
            f"{int(k)} & {b['seconds_per_window']:.2f} & "
            f"{b['minutes_full_stream']:.1f} & {t['seconds_per_window']:.2f} & "
            f"{t['minutes_full_stream']:.1f} & "
            f"{b['seconds_per_window'] / t['seconds_per_window']:.2f}$\\times$ \\\\"
        )

    latex = "\n".join([
        "\\begin{table}[!t]", "\\centering",
        f"\\caption{{Cost of one FFWIN window comparison on \\textit{{INSECTS "
        f"(abrupt)}} ($d{{=}}{d}$, $W{{=}}{args.window}$, $B{{=}}"
        f"{args.permutations}$), for both backends. Full-stream time is "
        f"extrapolated over {n_windows} comparisons. Both backends compute the "
        "same statistic and therefore yield identical alarms.}}",
        "\\label{tab:backend_cost}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Brute force}} & "
        "\\multicolumn{2}{c}{\\textbf{Range tree}} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "$k$ & s/window & min/stream & s/window & min/stream & \\textbf{Speedup} \\\\",
        "\\midrule", "\n".join(rows), "\\bottomrule",
        "\\end{tabular}", "\\end{table}",
    ]) + "\n"
    (out / "table_backend_cost.tex").write_text(latex, encoding="utf-8")

    print(f"\nwrote {out / 'backend_timings.csv'}")
    print(f"wrote {out / 'table_backend_cost.tex'}")


if __name__ == "__main__":
    main()
