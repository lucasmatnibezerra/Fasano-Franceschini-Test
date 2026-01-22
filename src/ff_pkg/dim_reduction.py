from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np


@dataclass
class PCAReducer:
    threshold: Optional[int] = None
    target_dim: Optional[int] = None
    random_state: Optional[int] = None
    fit_mode: str = "concat"

    def should_reduce(self, d: int) -> bool:
        if not self.threshold or self.threshold <= 0:
            return False
        if d < self.threshold:
            return False
        if self.target_dim is None:
            return False
        if self.target_dim >= d:
            return False
        return True

    def apply(
        self, ref: np.ndarray, cur: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
        d = ref.shape[1]
        if not self.should_reduce(d):
            return ref, cur, None
        k = int(self.target_dim)
        data_fit = ref if self.fit_mode == "ref" else np.vstack([ref, cur])
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except Exception:
            return ref, cur, None
        pca = PCA(n_components=k, random_state=self.random_state)
        pca.fit(data_fit)
        return pca.transform(ref), pca.transform(cur), pca
