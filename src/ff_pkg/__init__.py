# src/ff_pkg/__init__.py

__version__ = "0.1.0"

# Expondo utilitários de matriz
from .matrix_util import rbind, get_row

# Expondo estruturas de árvore
from .range_tree import RangeTree, NaiveRangeCounter

# Expondo cálculos de distância
from .distance import build_range_trees, brute_distance, range_distance

# Expondo estatísticas e testes
from .ff_core import test_statistic, ff_test_statistic

# Controle de exportação em import *
__all__ = [
    "rbind", "get_row",
    "RangeTree", "NaiveRangeCounter",
    "build_range_trees", "brute_distance", "range_distance",
    "test_statistic", "ff_test_statistic",
    "permutation_test", "permutation_test_parallel",
    "__version__",
]
