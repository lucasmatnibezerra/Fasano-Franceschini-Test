Fasano–Franceschini Test Documentation

This repository provides a Python implementation of the multivariate Kolmogorov–Smirnov test, known as the **Fasano & Franceschini (FF)** test, and a sliding-window drift detector (FFWIN).

<img src="src/figures/fasano-logo.png" alt="Fasano & Franceschini Logo" width="200" align="right"/>

---

## 1. Overview

The FF test generalizes the Kolmogorov–Smirnov (KS) test to multiple dimensions by comparing two independent samples (of arbitrary sizes) and assessing whether they come from the same distribution. It is useful in higher dimensions, where univariate tests fail to capture multivariate differences.

- Code author: Lucas Matni Bezerra
- Original method: Fasano & Franceschini (1987)
- Reference R implementation: Puritz, Ness-Cohn & Braun (2023)

---

## 2. Theoretical Background (summary)

1. Peacock (1983): introduces the D statistic for a multivariate KS test.
2. Fasano & Franceschini (1987): refine the approach via partitioning space into orthants defined by reference points from the two samples; the final statistic is \(D = d_1 + d_2\), where \(d_1\) and \(d_2\) are maxima of normalized differences of orthant counts when taking each sample as the set of origins.
3. Puritz, Ness-Cohn & Braun (2023): efficient R implementation that inspires this version.

---

## 3. Project Structure

```
$PROJECT_ROOT
├── LICENSE
├── README.md
├── environment.yml          # Conda environment with required dependencies
├── pyproject.toml           # Metadata and dependencies (all required)
├── ffwin/
│   ├── __init__.py          # Public API
│   ├── cli.py               # Simple CLI for CSV
│   ├── dim_reduction.py     # Optional PCA (scikit-learn)
│   ├── distance.py          # brute_distance (Numba) and range_distance
│   ├── ff_core.py           # FF test core (D, permutations)
│   ├── matrix_util.py       # Matrix utilities (rbind)
│   ├── range_tree.py        # Rtree/Numba backend for range counting
│   └── sliding_window.py    # FFWIN: sliding windows + PCA + p-values
└── examples/
    └── smoke.py             # Synthetic usage example
```

---

## 4. How to Use

### 4.1 Installation (Conda)

Create the environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate ffwin
pip install -e .
```

Note: On Linux, the `Rtree` backend requires `libspatialindex` (already included in `environment.yml`).

### 4.2 Import the Package

```python
from ffwin.ff_core import ff_test_statistic, permutation_test, permutation_test_parallel
from ffwin.sliding_window import SlidingWindowFasanoFranceschini
```

### 4.3 Example Usage (FF core)

```python
import numpy as np
from ffwin.ff_core import ff_test_statistic, permutation_test

X = np.random.randn(100, 3)
Y = np.random.randn(100, 3) + 0.5

# D statistic (range-tree)
D = ff_test_statistic(X, Y, method='r')
print(f"D = {D}")

# Permutation test (p-value)
zg, ze, p = permutation_test(
    X, Y,
    n_permutations=1000,
    method='r',
    seed=42,
    verbose=True,
)
print(f"p-value ≈ {p:.4f}")
```

### 4.4 Example Usage (FFWIN detector)

```python
import numpy as np
from ffwin.sliding_window import SlidingWindowFasanoFranceschini

T, d = 5000, 8
X = np.random.randn(T, d).astype(np.float32)
X[2500:] += 0.8

ff = SlidingWindowFasanoFranceschini(
    window_size=200,
    step=100,
    alpha=0.05,
    n_permutations=50,
    method='b',
    pairing='lag_step',
)
results = ff.run_parallel(X)
alarms = [r['start_cur'] for r in results if r['drift']]
print("alarms:", alarms[:10])
```

---

## 5. Requirements

- Python >= 3.10
- Required dependencies: numpy, joblib, numba, scikit-learn, pandas, tqdm, Rtree, libspatialindex

---

## 6. Notes

- For high dimension (large \(d\)), use `method='r'` or enable PCA.
- `run_parallel` parallelizes windows; permutations can be parallelized with `permutation_test_parallel`.

---

## 7. License

MIT
