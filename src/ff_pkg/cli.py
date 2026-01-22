import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from .sliding_window import SlidingWindowFasanoFranceschini


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="ffwin_results.csv")
    ap.add_argument("--win", type=int, default=1000)
    ap.add_argument("--step", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--method", choices=["b", "r"], default="b")
    ap.add_argument("--pairing", choices=["lag_step", "adjacent"], default="lag_step")
    ap.add_argument("--pca-threshold", type=int, default=None)
    ap.add_argument("--pca-target-dim", type=int, default=None)
    ap.add_argument("--pca-mode", choices=["concat", "ref"], default="concat")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(f"csv not found: {csv}")
    # lê tudo como float (apenas features por linha)
    X = np.genfromtxt(str(csv), delimiter=",", dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    ff = SlidingWindowFasanoFranceschini(
        window_size=args.win,
        step=args.step,
        alpha=args.alpha,
        n_permutations=args.perms,
        random_state=None,
        method=args.method,
        pairing=args.pairing,
        pca_threshold=args.pca_threshold,
        pca_target_dim=args.pca_target_dim,
        pca_mode=args.pca_mode,
        verbose=False,
    )
    res = ff.run_parallel(X)
    df = pd.DataFrame(res)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"[WRITE] {out}")


if __name__ == "__main__":
    main()
