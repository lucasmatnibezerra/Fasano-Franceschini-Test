import numpy as np
from joblib import Parallel, delayed
from .ff_core import compute_D, permutation_test
from .dim_reduction import PCAReducer


class SlidingWindowFasanoFranceschini:
    def __init__(
        self,
        window_size: int,
        step: int = 1,
        alpha: float = 0.05,
        n_permutations: int = 1000,
        random_state: int | None = None,
        method: str = "b",
        verbose: bool = False,
        pairing: str = "lag_step",
        pca_threshold: int | None = None,
        pca_target_dim: int | None = None,
        pca_mode: str = "concat",
    ):
        self.window_size = int(window_size)
        self.step = int(step)
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.random_state = random_state
        self.method = str(method)
        self.verbose = verbose
        self.pairing = pairing
        if random_state is not None:
            np.random.seed(random_state)
        self._pca_reducer = PCAReducer(
            threshold=pca_threshold,
            target_dim=pca_target_dim,
            random_state=random_state,
            fit_mode=pca_mode,
        )

    def _compute_window_stat(
        self, ref: np.ndarray, cur: np.ndarray, window_id: int | None = None
    ):
        import time

        t0 = time.time()
        D = compute_D(ref, cur, method=self.method)
        t1 = time.time()
        zg, ze, p_value = permutation_test(
            ref,
            cur,
            self.n_permutations,
            verbose=self.verbose,
            method=self.method,
            seed=self.random_state,
        )
        t2 = time.time()
        return D, p_value, (t1 - t0), (t2 - t1)

    def _iter_starts(self, n_samples: int):
        W, s = self.window_size, self.step
        if self.pairing == "adjacent":
            end = n_samples - 2 * W
        else:
            end = n_samples - (W + s)
        if end < 0:
            return range(0)
        return range(0, end + 1, s)

    def run(self, data: np.ndarray):
        n_samples = int(data.shape[0])
        W, s = self.window_size, self.step
        starts = list(self._iter_starts(n_samples))
        results = []
        for i, start in enumerate(starts):
            start_cur = start + (W if self.pairing == "adjacent" else s)
            ref_window = data[start : start + W]
            cur_window = data[start_cur : start_cur + W]
            ref_proc, cur_proc, pca_obj = self._pca_reducer.apply(
                ref_window, cur_window
            )
            D, p, tD, tP = self._compute_window_stat(
                ref_proc, cur_proc, window_id=i + 1
            )
            results.append(
                {
                    "start_ref": start,
                    "start_cur": start_cur,
                    "D": D,
                    "p": p,
                    "drift": p < self.alpha,
                    "pca_applied": pca_obj is not None,
                    "time_D": tD,
                    "time_perms": tP,
                }
            )
        return results

    def run_parallel(self, data: np.ndarray, n_jobs: int = -1):
        n_samples = int(data.shape[0])
        W, s = self.window_size, self.step
        starts = list(self._iter_starts(n_samples))

        def process(i, start):
            start_cur = start + (W if self.pairing == "adjacent" else s)
            ref_window = data[start : start + W]
            cur_window = data[start_cur : start_cur + W]
            ref_proc, cur_proc, pca_obj = self._pca_reducer.apply(
                ref_window, cur_window
            )
            D, p, tD, tP = self._compute_window_stat(
                ref_proc, cur_proc, window_id=i + 1
            )
            return {
                "start_ref": start,
                "start_cur": start_cur,
                "D": D,
                "p": p,
                "drift": p < self.alpha,
                "pca_applied": pca_obj is not None,
                "time_D": tD,
                "time_perms": tP,
            }

        return Parallel(n_jobs=n_jobs)(
            delayed(process)(i, s0) for i, s0 in enumerate(starts)
        )
