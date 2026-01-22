import numpy as np


def rbind(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    return np.vstack((M1, M2))


def get_row(M: np.ndarray, row: int) -> np.ndarray:
    return M[row].copy()
