# matrix_util.py

import numpy as np

def rbind(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Row bind two matrices (stack vertically).

    Parameters
    ----------
    M1 : np.ndarray of shape (r1, c)
    M2 : np.ndarray of shape (r2, c)

    Returns
    -------
    np.ndarray of shape (r1+r2, c)
    """
    return np.vstack((M1, M2))

def get_row(M: np.ndarray, row: int) -> np.ndarray:
    """
    Extract a row from a matrix.

    Parameters
    ----------
    M : np.ndarray of shape (r, c)
    row : int

    Returns
    -------
    np.ndarray of shape (c,)
    """
    return M[row].copy()
