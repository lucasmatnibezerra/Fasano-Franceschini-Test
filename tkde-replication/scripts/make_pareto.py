"""Accuracy against retraining cost on the real streams (Fig. fig:pareto).

Reads the same per-seed CSVs as ``make_tables.py`` (experiment
``revision_main``), so every point is a cell of Table ``tab:acc_ht_detectors``
and of the alarm-count table: accuracy is ``acc (ht-dd)`` and the cost is the
number of alarms, i.e. of retrainings, both averaged over seeds.

Each baseline is placed relative to FFWIN on the same stream: x is its alarm
count divided by FFWIN's, y its accuracy minus FFWIN's in percentage points.
The small dots are the eleven streams, the large marker the median over
streams. FFWIN sits at (1, 0) by construction.

It also prints, per stream, which detectors lie on the Pareto front (maximum
accuracy, minimum alarms), so the claims in the text can be checked.

    python scripts/make_pareto.py --out ../paper_revision/figures/pareto_accuracy_alarms.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables as mt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

FFWIN_BLUE = "#2a78d6"
CONTEXT = "#8a8983"
INK = "#0b0b0b"
INK_2 = "#52514e"
PERIODIC = {"OCDD"}  # 99.8% of its inter-alarm gaps are exactly one window


def per_stream(results: str, experiment: str) -> pd.DataFrame:
    data = mt.load(results, experiment)
    return (
        data.groupby(["stream", "detector"])
        .agg(acc=("acc (ht-dd)", "mean"), alarms=("n_alarms", "mean"))
        .reset_index()
    )


def pareto_front(sub: pd.DataFrame) -> list[str]:
    on = []
    for _, r in sub.iterrows():
        dominated = (
            (sub.acc >= r.acc) & (sub.alarms <= r.alarms)
            & ((sub.acc > r.acc) | (sub.alarms < r.alarms))
        ).any()
        if not dominated:
            on.append(r.detector)
    return on


def relative(g: pd.DataFrame) -> pd.DataFrame:
    ref = g[g.detector == "FFWIN"].set_index("stream")
    out = g[g.detector != "FFWIN"].copy()
    out["ratio"] = out.alarms / out.stream.map(ref.alarms)
    out["dacc_pp"] = 100 * (out.acc - out.stream.map(ref.acc))
    return out


def report(g: pd.DataFrame, rel: pd.DataFrame) -> None:
    counts: dict[str, int] = {}
    for stream, sub in g.groupby("stream"):
        front = pareto_front(sub)
        for d in front:
            counts[d] = counts.get(d, 0) + 1
        f = sub[sub.detector == "FFWIN"].iloc[0]
        print("{:<26} FFWIN acc {:.3f} (best {:.3f}, gap {:+.1f} pp), alarms {:.1f}, front {}".format(
            stream, f.acc, sub.acc.max(), 100 * (f.acc - sub.acc.max()), f.alarms, front))
    print("on the Pareto front, streams out of {}: {}".format(
        g.stream.nunique(), dict(sorted(counts.items(), key=lambda kv: -kv[1]))))
    summary = rel.groupby("detector").agg(
        median_ratio=("ratio", "median"), median_dacc_pp=("dacc_pp", "median"),
        mean_dacc_pp=("dacc_pp", "mean"),
    )
    print(summary.round(2).to_string())


def plot(rel: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    ax.axhline(0, color=INK_2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axvline(1, color=INK_2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.grid(True, which="major", color="#e4e3df", lw=0.5, zorder=0)

    # Keep log-scale points finite: a stream where a baseline never fired.
    rel = rel.assign(ratio=rel.ratio.clip(lower=0.02))

    ax.scatter(rel.ratio, rel.dacc_pp, s=6, color=CONTEXT, alpha=0.35, lw=0, zorder=2)

    offsets = {  # label placement, points relative to the median marker
        "BNDM": (-4, 7), "CSDDM": (-5, 6), "D3": (-10, 12), "IBDD": (6, -7),
        "OCDD": (7, -1), "SPLL": (6, -7), "UDetect": (6, 7),
    }
    for det, sub in rel.groupby("detector"):
        x, y = sub.ratio.median(), sub.dacc_pp.median()
        hollow = det in PERIODIC
        ax.scatter([x], [y], s=26, marker="o", zorder=4,
                   facecolor="white" if hollow else CONTEXT,
                   edgecolor=CONTEXT if hollow else "white", lw=1.0 if hollow else 0.8)
        dx, dy = offsets.get(det, (5, 4))
        ax.annotate(det + (" (periodic)" if hollow else ""), (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.8, color=INK,
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center",
                    # D3's median sits under the FFWIN star; a leader keeps it findable.
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5, shrinkA=0, shrinkB=3)
                    if det == "D3" else None)

    ax.scatter([1], [0], s=90, marker="*", color=FFWIN_BLUE, edgecolor="white",
               lw=0.6, zorder=5)
    ax.annotate("FFWIN", (1, 0), xytext=(-6, -8), textcoords="offset points",
                fontsize=7, color=INK, ha="right", va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Alarms (retrainings) relative to FFWIN, same stream")
    ax.set_ylabel("Accuracy $-$ FFWIN accuracy (pp)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--experiment", default="revision_main")
    parser.add_argument("--out", default=str(ROOT / "tables" / "pareto_accuracy_alarms.pdf"))
    args = parser.parse_args()

    g = per_stream(args.results, args.experiment)
    rel = relative(g)
    report(g, rel)
    plot(rel, Path(args.out))


if __name__ == "__main__":
    main()
