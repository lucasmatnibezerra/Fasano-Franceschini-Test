"""Turn the benchmark's results/ directory into the LaTeX tables the revision needs.

Reads ``results/<Stream>/<Detector>_<experiment>.csv`` as written by the
benchmark's ExperimentLogger and emits, into ``tables/``:

    table_alarm_counts.tex     alarms per detector and stream          (Reviewer 2)
    table_accuracy.tex         downstream accuracy with 95% CIs        (Reviewers 1, 2)
    table_lpd.tex              lift-per-drift with 95% CIs             (Reviewers 1, 2)
    table_insects_gt.tex       MDR/MTD/MTFA/MTR on INSECTS abrupt      (Reviewer 2)
    table_classifier.tex       Hoeffding Tree vs Naive Bayes           (Reviewer 3)
    table_pca_ablation.tex     PCA mode x k sweep                      (Reviewers 1, 2)
    stats_friedman.txt         Friedman + Nemenyi critical difference  (Reviewer 2)

Every number is computed from the logged rows; nothing is hard-coded.

Usage:
    python make_tables.py --results results --experiment revision_main
    python make_tables.py --results results --experiment revision_pca --only pca
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Studentized range statistic q_alpha for alpha=0.05, infinite degrees of
# freedom, indexed by the number of compared methods. Demsar (2006), Table 5.
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}

# How detector class names and stream class names should appear in the paper.
DETECTOR_LABELS = {
    "BayesianNonparametricDetectionMethod": "BNDM",
    "ClusteredStatisticalTestDriftDetectionMethod": "CSDDM",
    "DiscriminativeDriftDetector2019": "D3",
    "EDFS": "EDFS",
    "ImageBasedDriftDetector": "IBDD",
    "NNDVI": "NN-DVI",
    "OneClassDriftDetector": "OCDD",
    "SemiParametricLogLikelihood": "SPLL",
    "UCDD": "UCDD",
    "UDetect": "UDetect",
    "FasanoFranceschiniWindow": "FFWIN",
}

STREAM_LABELS = {
    "Electricity": "Elec2",
    "InsectsAbruptBalanced": "INSECTS (abrupt)",
    "InsectsGradualBalanced": "INSECTS (gradual)",
    "InsectsIncrementalAbruptBalanced": "INSECTS (inc.-abrupt)",
    "InsectsIncrementalBalanced": "INSECTS (incremental)",
    "InsectsIncrementalReoccurringBalanced": "INSECTS (inc.-reocc.)",
    "NOAAWeather": "NOAA Weather",
    "OutdoorObjects": "Outdoor Objects",
    "PokerHand": "Poker Hands",
    "Powersupply": "Powersupply",
    "RialtoBridgeTimelapse": "Rialto Bridge Timelapse",
}

# Order used in the paper's tables; FFWIN last.
DETECTOR_ORDER = [
    "BNDM", "CSDDM", "D3", "EDFS", "IBDD", "NN-DVI", "OCDD", "SPLL", "UCDD",
    "UDetect", "FFWIN",
]
STREAM_ORDER = list(STREAM_LABELS.values())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def count_alarms(value) -> float:
    """Count the detections stored in a logged ``drifts`` cell."""
    if isinstance(value, (list, tuple)):
        return float(len(value))
    if not isinstance(value, str) or not value.strip():
        return np.nan
    try:
        return float(len(ast.literal_eval(value)))
    except (ValueError, SyntaxError):
        return np.nan


def parse_alarms(value) -> List[int]:
    """Parse a logged ``drifts`` cell into a list of timesteps."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return list(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return []


def load(results_dir: str, experiment: str) -> pd.DataFrame:
    """
    Load every results CSV belonging to one experiment into a single frame.

    :param results_dir: the benchmark's results directory
    :param experiment: the experiment name used when running main.py
    :return: one row per logged run, with ``stream`` and ``detector`` columns
    """
    root = Path(results_dir)
    if not root.is_dir():
        raise SystemExit(f"results directory not found: {root.resolve()}")

    pattern = re.compile(rf"^(?P<detector>.+)_{re.escape(experiment)}\.csv$")
    frames = []
    for stream_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in sorted(stream_dir.glob("*.csv")):
            match = pattern.match(csv_path.name)
            if not match:
                continue
            frame = pd.read_csv(csv_path)
            if frame.empty:
                continue
            frame["stream"] = STREAM_LABELS.get(stream_dir.name, stream_dir.name)
            raw_detector = match.group("detector")
            frame["detector"] = DETECTOR_LABELS.get(raw_detector, raw_detector)
            frames.append(frame)

    if not frames:
        raise SystemExit(
            f"no CSV matching '*_{experiment}.csv' under {root.resolve()}"
        )

    data = pd.concat(frames, ignore_index=True)
    if "drifts" in data.columns:
        data["n_alarms"] = data["drifts"].map(count_alarms)
    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def mean_ci(values: pd.Series, decimals: int = 3) -> str:
    """Format a mean with its 95% normal-approximation confidence interval."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "--"
    mean = clean.mean()
    if len(clean) < 2:
        return f"{mean:.{decimals}f}"
    half_width = 1.96 * clean.std(ddof=1) / np.sqrt(len(clean))
    return f"{mean:.{decimals}f} $\\pm$ {half_width:.{decimals}f}"


def pivot(
    data: pd.DataFrame, column: str, formatter, streams=None, detectors=None
) -> pd.DataFrame:
    """Build a stream x detector table of formatted cells."""
    streams = streams or [s for s in STREAM_ORDER if s in set(data["stream"])]
    detectors = detectors or [
        d for d in DETECTOR_ORDER if d in set(data["detector"])
    ]
    table = pd.DataFrame(index=streams, columns=detectors, dtype=object)
    for (stream, detector), group in data.groupby(["stream", "detector"]):
        if stream in table.index and detector in table.columns:
            table.loc[stream, detector] = formatter(group[column])
    return table.fillna("--")


def _cell_mean(cell: str) -> Optional[float]:
    """The mean out of a formatted cell such as '0.630 $\\pm$ 0.016'."""
    if not isinstance(cell, str) or cell.strip() in {"--", ""}:
        return None
    head = cell.split("$")[0].replace("\\textbf{", "").replace("}", "").strip()
    try:
        return float(head)
    except ValueError:
        return None


def _embolden(cell: str) -> str:
    """Bold the mean but leave the interval upright, as elsewhere in the paper."""
    if "\\textbf{" in cell:
        return cell
    if "$\\pm$" in cell:
        head, tail = cell.split("$\\pm$", 1)
        return f"\\textbf{{{head.strip()}}} $\\pm${tail}"
    return f"\\textbf{{{cell.strip()}}}"


def bold_best(table: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Bold the best cell of every row. Ties are all bolded; a row that is
    entirely undefined, or in which every detector agrees, is left alone."""
    out = table.copy()
    for idx, row in table.iterrows():
        values = {c: _cell_mean(v) for c, v in row.items()}
        values = {c: v for c, v in values.items() if v is not None}
        if len(values) < 2 or len(set(values.values())) == 1:
            continue
        best = max(values.values()) if higher_is_better else min(values.values())
        for c, v in values.items():
            if v == best:
                out.loc[idx, c] = _embolden(str(row[c]))
    return out


def bold_best_in_columns(rows: list, directions: list) -> list:
    """Bold the best cell of every column, for tables whose rows are detectors.
    `directions` gives one of 'high', 'low' or None per column after the first."""
    out = [list(r) for r in rows]
    for j, direction in enumerate(directions, start=1):
        if direction is None:
            continue
        values = {i: _cell_mean(r[j]) for i, r in enumerate(out)}
        values = {i: v for i, v in values.items() if v is not None}
        if len(values) < 2 or len(set(values.values())) == 1:
            continue
        best = max(values.values()) if direction == "high" else min(values.values())
        for i, v in values.items():
            if v == best:
                out[i][j] = _embolden(str(out[i][j]))
    return out


def to_latex(
    table: pd.DataFrame,
    caption: str,
    label: str,
    note: Optional[str] = None,
    star: bool = True,
) -> str:
    """Render a formatted table as a booktabs LaTeX table."""
    env = "table*" if star else "table"
    columns = "l" + "c" * len(table.columns)
    header = " & ".join(["\\textbf{Data stream}"]
                        + [f"\\textbf{{{c}}}" for c in table.columns])
    body = "\n".join(
        f"{index} & " + " & ".join(str(v) for v in row) + r" \\"
        for index, row in table.iterrows()
    )
    lines = [
        f"\\begin{{{env}}}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
        "\\toprule",
        header + r" \\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize {note}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def write(out_dir: Path, name: str, content: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n" on purpose: the manuscript claims these files regenerate
    # byte-identically, which must not depend on the platform running them.
    with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    print(f"  wrote {out_dir / name}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_alarm_counts(data: pd.DataFrame, out: Path) -> None:
    """Alarms per detector and stream, so the LPD ratios can be read."""
    if "n_alarms" not in data.columns:
        print("  skipping alarm counts: no 'drifts' column")
        return
    table = pivot(data, "n_alarms", lambda s: mean_ci(s, decimals=1))
    write(out, "table_alarm_counts.tex", to_latex(
        table,
        "Number of alarms raised by each detector on each real stream "
        "(mean $\\pm$ 95\\% CI over seeds). Reported so that the lift-per-drift "
        "ratios of Table~\\ref{tab:lpd_ffwin} can be interpreted: a detector "
        "that alarms rarely can attain a large ratio from a modest absolute gain.",
        "tab:alarm_counts",
    ))


def table_accuracy(data: pd.DataFrame, out: Path) -> None:
    """Downstream Hoeffding Tree accuracy with confidence intervals."""
    if "acc (ht-dd)" not in data.columns:
        print("  skipping accuracy: no 'acc (ht-dd)' column")
        return
    table = bold_best(pivot(data, "acc (ht-dd)", mean_ci), higher_is_better=True)
    write(out, "table_accuracy.tex", to_latex(
        table,
        "Downstream mean accuracy of the Hoeffding Tree under each detector on "
        "real streams (mean $\\pm$ 95\\% CI over seeds).",
        "tab:acc_ht_detectors",
    ))


def table_lpd(data: pd.DataFrame, out: Path) -> None:
    """Lift-per-drift with confidence intervals."""
    if "lpd (ht)" not in data.columns:
        print("  skipping LPD: no 'lpd (ht)' column")
        return
    table = bold_best(pivot(data, "lpd (ht)", lambda s: mean_ci(s, decimals=4)),
                      higher_is_better=True)
    write(out, "table_lpd.tex", to_latex(
        table,
        "Lift-per-drift of the Hoeffding Tree with different detectors on real "
        "streams (mean $\\pm$ 95\\% CI over seeds). Read together with the alarm "
        "counts in Table~\\ref{tab:alarm_counts}.",
        "tab:lpd_ffwin",
    ))


def table_classifier_sensitivity(data: pd.DataFrame, out: Path) -> None:
    """Hoeffding Tree against Gaussian Naive Bayes, per detector."""
    needed = {"lpd (ht)", "lpd (nb)", "acc (ht-dd)", "acc (nb-dd)"}
    if not needed.issubset(data.columns):
        print("  skipping classifier sensitivity: missing NB columns")
        return

    detectors = [d for d in DETECTOR_ORDER if d in set(data["detector"])]
    rows = []
    for detector in detectors:
        subset = data[data["detector"] == detector]
        rows.append([
            detector,
            mean_ci(subset["acc (ht-dd)"]),
            mean_ci(subset["acc (nb-dd)"]),
            mean_ci(subset["lpd (ht)"], decimals=4),
            mean_ci(subset["lpd (nb)"], decimals=4),
        ])

    rows = bold_best_in_columns(rows, ["high", "high", "high", "high"])

    header = " & ".join([
        "\\textbf{Detector}", "\\textbf{Acc.\\ (HT)}", "\\textbf{Acc.\\ (NB)}",
        "\\textbf{LPD (HT)}", "\\textbf{LPD (NB)}",
    ])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    content = "\n".join([
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Sensitivity of the proxy metrics to the downstream "
        "classifier: Hoeffding Tree (HT) against Gaussian Naive Bayes (NB), "
        "averaged over all real streams and seeds.}",
        "\\label{tab:classifier_sensitivity}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule", header + r" \\",
        "\\midrule", body, "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]) + "\n"
    write(out, "table_classifier.tex", content)


def table_insects_ground_truth(data: pd.DataFrame, out: Path) -> None:
    """MDR, MTD, MTFA and MTR on the one real stream with exact drift times."""
    if not {"mdr", "mtd", "mtfa", "mtr"}.issubset(data.columns):
        print("  skipping INSECTS ground truth: no drift-metric columns")
        return
    subset = data[data["stream"] == "INSECTS (abrupt)"]
    if subset.empty:
        print("  skipping INSECTS ground truth: stream not in results")
        return

    detectors = [d for d in DETECTOR_ORDER if d in set(subset["detector"])]
    rows = []
    for detector in detectors:
        group = subset[subset["detector"] == detector]
        rows.append([
            detector,
            mean_ci(group["mdr"], decimals=2),
            mean_ci(group["mtd"], decimals=1),
            mean_ci(group["mtfa"], decimals=1),
            mean_ci(group["mtr"], decimals=2),
        ])

    rows = bold_best_in_columns(rows, ["low", "low", "high", "high"])

    header = " & ".join([
        "\\textbf{Detector}",
        "\\textbf{MDR}\\,$\\downarrow$", "\\textbf{MTD}\\,$\\downarrow$",
        "\\textbf{MTFA}\\,$\\uparrow$", "\\textbf{MTR}\\,$\\uparrow$",
    ])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    content = "\n".join([
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Detection quality on \\textit{INSECTS (abrupt)}, the only "
        "real stream with exact drift timestamps (mean $\\pm$ 95\\% CI over "
        "seeds, tolerance $\\Delta$ as set by the benchmark). MTFA is undefined "
        "when fewer than two false alarms occur.}",
        "\\label{tab:insects_gt}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule", header + r" \\",
        "\\midrule", body, "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]) + "\n"
    write(out, "table_insects_gt.tex", content)


def _pca_cell(values: pd.Series, decimals: int = 2) -> str:
    """Format one sweep cell: mean, CI, and the seed count when it is partial.

    Two refinements over ``mean_ci``. A confidence interval that rounds away to
    zero is dropped, because several cells are identical across all ten seeds
    and "$\\pm$ 0.00" costs a third of the column width to say nothing. And a
    cell averaged over fewer seeds than the run provides carries that count, so
    a mean over six seeds is never read as a mean over ten -- which matters
    here, since a seed drops out of MTFA exactly when the detector alarmed
    rarely, the same condition that inflates the ratio in the seeds that stay.
    """
    text = mean_ci(values, decimals=decimals)
    zero = " $\\pm$ " + f"{0:.{decimals}f}"
    if text.endswith(zero):
        text = text[: -len(zero)]
    return text


def _seed_coverage(values: pd.Series):
    """(usable, total) seeds behind one cell."""
    numeric = pd.to_numeric(values, errors="coerce")
    return len(numeric.dropna()), len(numeric)


def table_pca_ablation(data: pd.DataFrame, out: Path) -> None:
    """The pca_mode x k sweep: MTR and MDR, ref against concat.

    Reported for the stream that has exact drift timestamps, since MTR and MDR
    are undefined without them. Streams whose ground truth is unknown yield an
    all-dashes block and are listed in the caption instead of printed.
    """
    if "pca_mode" not in data.columns or "pca_target_dim" not in data.columns:
        print("  skipping PCA ablation: run EXPERIMENT_PCA first")
        return

    ks = sorted(data["pca_target_dim"].dropna().unique())
    metrics = [("mtr", "\\ac{MTR}\\,$\\uparrow$", "high"),
               ("mdr", "\\ac{MDR}\\,$\\downarrow$", "low")]
    modes = ["ref", "concat"]

    scored, unscored = [], []
    for stream in [s for s in STREAM_ORDER if s in set(data["stream"])]:
        subset = data[data["stream"] == stream]
        has_ground_truth = pd.to_numeric(subset.get("mdr"), errors="coerce").notna().any()
        (scored if has_ground_truth else unscored).append(stream)

    if not scored:
        print("  skipping PCA ablation: no stream carries ground-truth drifts")
        return

    blocks, partial = [], []
    for stream in scored:
        for metric, label, direction in metrics:
            block = []
            for mode in modes:
                subset = data[
                    (data["stream"] == stream) & (data["pca_mode"] == mode)
                ]
                if subset.empty:
                    continue
                cells = []
                for k in ks:
                    at_k = subset[subset["pca_target_dim"] == k][metric]
                    cells.append(_pca_cell(at_k))
                    usable, total = _seed_coverage(at_k)
                    if usable and usable < total:
                        partial.append((metric.upper(), mode, int(k), usable, total))
                block.append([f"\\texttt{{{mode}}}"] + cells)
            if not block:
                continue
            # No cell is marked best here. This table compares two fittings of
            # one detector, not competitors, and the largest MTR in the sweep is
            # produced by going quiet (MDR 0.20), so a bold mark on it would
            # assert the opposite of what the row shows.
            block[0] = [label] + block[0]
            for row in block[1:]:
                row.insert(0, "")
            blocks.append(block)

    omitted = ""
    if unscored:
        omitted = (
            " On " + ", ".join(f"\\textit{{{s}}}" for s in unscored)
            + " the drift times are unknown, so these metrics are undefined and"
            " the rows are omitted."
        )

    header = " & ".join([" ", "\\textbf{fit}"] + [f"$k{{=}}{int(k)}$" for k in ks])
    body = "\n".join(
        " & ".join(row) + r" \\" for block in blocks for row in block
    )
    note = ""
    if partial:
        # Stated under the table rather than inside the cell, which would make
        # the tabular overfull, and never silently: a mean over six seeds must
        # not read as a mean over ten.
        items = "; ".join(
            "\\ac{{{}}} at $k{{=}}{}$ under \\texttt{{{}}} covers {} of {} seeds".format(
                metric, k, mode, usable, total
            )
            for metric, mode, k, usable, total in partial
        )
        note = (
            "\\vspace{2pt}{\\scriptsize\\raggedright \\ac{MTFA} is undefined for "
            "a seed with fewer than two false alarms, so " + items
            + ". A seed drops out exactly when the detector alarmed rarely, "
            "which is the condition that inflates the ratio in those that "
            "remain.\\par}"
        )
    content = "\n".join([
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Effect of the PCA projection on \\ac{FFWIN} on "
        f"\\textit{{{scored[0]}}} (mean $\\pm$ 95\\% CI over seeds; an interval "
        "that rounds to zero is omitted). \\texttt{ref} fits the subspace on the "
        "reference window, \\texttt{concat} on both windows stacked, which is "
        "the published default. No cell is marked best: at $k{=}12$ the large "
        "\\ac{MTR} is produced by silence, as the \\ac{MDR} row shows."
        + omitted + "}",
        "\\label{tab:pca_ablation}", "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{@{{}}ll{'c' * len(ks)}@{{}}}}",
        "\\toprule", header + r" \\", "\\midrule", body,
        "\\bottomrule", "\\end{tabular}",
    ] + ([note] if note else []) + [
        "\\end{table}",
    ]) + "\n"
    write(out, "table_pca_ablation.tex", content)


def stats_friedman(data: pd.DataFrame, out: Path) -> None:
    """Friedman test across detectors, followed by Nemenyi critical difference."""
    from scipy.stats import friedmanchisquare

    lines = []
    for column, name in [("lpd (ht)", "lift-per-drift"), ("acc (ht-dd)", "accuracy")]:
        if column not in data.columns:
            continue
        matrix = (
            data.groupby(["stream", "detector"])[column].mean().unstack("detector")
        )
        matrix = matrix[[d for d in DETECTOR_ORDER if d in matrix.columns]].dropna()
        n_streams, n_detectors = matrix.shape
        if n_streams < 3 or n_detectors < 3:
            lines.append(f"{name}: too few streams/detectors for Friedman\n")
            continue

        statistic, p_value = friedmanchisquare(*[matrix[c] for c in matrix.columns])
        # Rank 1 = best, so rank the negated values (higher metric is better).
        ranks = (-matrix).rank(axis=1).mean().sort_values()

        q = NEMENYI_Q05.get(n_detectors)
        cd = (
            q * np.sqrt(n_detectors * (n_detectors + 1) / (6.0 * n_streams))
            if q else float("nan")
        )

        lines.append(f"=== {name} ===")
        lines.append(f"streams (blocks) N = {n_streams}, detectors k = {n_detectors}")
        lines.append(f"Friedman chi2 = {statistic:.4f}, p = {p_value:.6f}")
        lines.append(
            "significant at 0.05: "
            + ("yes" if p_value < 0.05 else "no -- do not run the post-hoc")
        )
        lines.append(f"Nemenyi critical difference (alpha=0.05) = {cd:.4f}")
        lines.append("mean ranks (1 = best):")
        for detector, rank in ranks.items():
            lines.append(f"  {detector:<10s} {rank:.3f}")
        best = ranks.iloc[0]
        tied = [d for d, r in ranks.items() if r - best <= cd]
        lines.append(
            f"not separable from the best rank at CD={cd:.3f}: {', '.join(tied)}"
        )
        lines.append("")

    if not lines:
        print("  skipping Friedman: no usable metric columns")
        return
    write(out, "stats_friedman.txt", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

BUILDERS = {
    "alarms": table_alarm_counts,
    "accuracy": table_accuracy,
    "lpd": table_lpd,
    "classifier": table_classifier_sensitivity,
    "insects": table_insects_ground_truth,
    "pca": table_pca_ablation,
    "friedman": stats_friedman,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="results directory")
    parser.add_argument("--experiment", required=True, help="experiment name")
    parser.add_argument("--out", default="tables", help="output directory")
    parser.add_argument(
        "--only", nargs="*", choices=sorted(BUILDERS), help="build only these"
    )
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="DETECTOR=EXPERIMENT",
        help="take this detector's rows from another experiment, e.g. "
             "FasanoFranceschiniWindow=revision_stride",
    )
    args = parser.parse_args()

    data = load(args.results, args.experiment)
    for spec in args.override:
        raw, _, experiment = spec.partition("=")
        label = DETECTOR_LABELS.get(raw, raw)
        other = load(args.results, experiment)
        other = other[other["detector"] == label]
        if other.empty:
            raise SystemExit(f"--override {spec}: no rows for {raw} in '{experiment}'")
        data = pd.concat([data[data["detector"] != label], other], ignore_index=True)
        print(f"{label}: rows taken from '{experiment}'")
    print(
        f"loaded {len(data)} rows: "
        f"{data['detector'].nunique()} detectors x {data['stream'].nunique()} streams"
    )
    seeds = data["seed"].nunique() if "seed" in data.columns else 0
    print(f"seeds per configuration: {seeds}")
    if seeds < 2:
        print("  warning: fewer than 2 seeds, confidence intervals will be omitted")

    out = Path(args.out)
    for name in (args.only or BUILDERS):
        BUILDERS[name](data, out)


if __name__ == "__main__":
    main()
