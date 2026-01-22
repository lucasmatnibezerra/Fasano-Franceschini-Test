Fasano-Franceschini Test Documentation

Este repositório apresenta uma implementação Python do teste multivariado de Kolmogorov–Smirnov, conhecido como teste de **Fasano & Franceschini (FF)**, e um detector de drift baseado em janelas deslizantes (FFWIN).

<img src="src\figures\fasano-logo.png" alt="Fasano & Franceschini Logo" width="200" align="right"/>

---

## 1. Overview

O teste FF generaliza o KS para várias dimensões, comparando duas amostras independentes (de tamanhos arbitrários) e avaliando se provêm da mesma distribuição. É útil em alta dimensão, onde testes univariados não capturam diferenças multivariadas.

- Code author: Lucas Matni Bezerra
- Original method: Fasano & Franceschini (1987)
- Reference R implementation: Puritz, Ness-Cohn & Braun (2023)

---

## 2. Theoretical Background (resumo)

1. Peacock (1983): introduz a estatística D para KS multivariado.
2. Fasano & Franceschini (1987): refinam via partição em ortantes definidos por pontos de referência das duas amostras; a estatística final é \(D = d_1 + d_2\), onde \(d_1\) e \(d_2\) são máximos de diferenças normalizadas de contagens por ortante tomando origens em cada amostra.
3. Puritz, Ness-Cohn & Braun (2023): implementação eficiente em R que inspira esta versão.

---

## 3. Project Structure

```
$PROJECT_ROOT
├── LICENSE
├── README.md
├── environment.yml          # Ambiente Conda com dependências obrigatórias
├── pyproject.toml           # Metadados e deps (todas obrigatórias)
├── ffwin/
│   ├── __init__.py          # API pública
│   ├── cli.py               # CLI simples para CSV
│   ├── dim_reduction.py     # PCA condicional (scikit-learn)
│   ├── distance.py          # brute_distance (Numba) e range_distance
│   ├── ff_core.py           # núcleo do teste FF (D, permutações)
│   ├── matrix_util.py       # utilitários de matriz (rbind)
│   ├── range_tree.py        # backend Rtree/Numba para contagem em faixas
│   └── sliding_window.py    # FFWIN: janelas deslizantes + PCA + p-valores
└── examples/
    └── smoke.py             # exemplo sintético de uso
```

---

## 4. How to use

### 4.1 Installation (Conda)

Crie o ambiente e instale o pacote em modo editável:

```bash
conda env create -f environment.yml
conda activate ffwin
pip install -e .
```

Obs.: Em Linux, o backend `Rtree` requer `libspatialindex` (o `environment.yml` já inclui `libspatialindex`).

### 4.2 Import the package

```python
from ffwin.ff_core import ff_test_statistic, permutation_test, permutation_test_parallel
from ffwin.sliding_window import SlidingWindowFasanoFranceschini
```

### 4.3 Example usage (FF core)

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

### 4.4 Example usage (FFWIN detector)

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
- Dependências obrigatórias: numpy, joblib, numba, scikit-learn, pandas, tqdm, Rtree, libspatialindex

---

## 6. Notes

- Para alta dimensão (\(d\) grande), use `method='r'` ou ative PCA.
- `run_parallel` paraleliza janelas; permutações podem ser paralelizadas com `permutation_test_parallel`.

---

## 7. License

MIT
