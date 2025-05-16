import pandas as pd
import numpy as np
from ff_pkg.ff_core import ff_test_statistic, permutation_test

def generate_csv_files(
    n_samples: int = 50,
    n_features: int = 6,
    shift: float = 2.0,
    seed: int = 42
):
    np.random.seed(seed)
    half = n_samples // 2

    # CSV with equal distributions
    g1 = np.random.randn(half, n_features)
    g2 = np.random.randn(half, n_features)
    df_equal = pd.DataFrame(
        np.vstack([g1, g2]),
        columns=[f"feature_{i+1}" for i in range(n_features)]
    )
    df_equal["group"] = [0]*half + [1]*half
    df_equal.to_csv("equal_distributions.csv", index=False)

    # CSV with different distributions
    g1 = np.random.randn(half, n_features)
    g2 = np.random.randn(half, n_features) + shift
    df_diff = pd.DataFrame(
        np.vstack([g1, g2]),
        columns=[f"feature_{i+1}" for i in range(n_features)]
    )
    df_diff["group"] = [0]*half + [1]*half
    df_diff.to_csv("different_distributions.csv", index=False)

def run_fasano_test(csv_path, n_permutations=1000, method='r', seed=42):
  
    df = pd.read_csv(csv_path)
    feat = [c for c in df.columns if c.startswith("feature_")]
    X = df[df.group == 0][feat].values
    Y = df[df.group == 1][feat].values

    # D statistic
    D = ff_test_statistic(X, Y, method=method)

    # P-value by permutation test
    z_greater, z_equal, p_value = permutation_test(
        X, Y,
        n_permutations=n_permutations,
        method=method,
        seed=seed,
        verbose=True
    )
    return D, p_value

if __name__ == "__main__":
    # Gera os CSVs
    generate_csv_files()

    # Same distributions test
    D_eq, p_eq = run_fasano_test("equal_distributions.csv")
    print(f"[Equal]     D = {D_eq}, p-value ≈ {p_eq:.4f}")

    # Diffetent distributions test
    D_diff, p_diff = run_fasano_test("different_distributions.csv")
    print(f"[Different] D = {D_diff}, p-value ≈ {p_diff:.4f}")
