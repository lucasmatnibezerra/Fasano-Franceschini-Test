"""FFWIN adapter for the DFKI-NI unsupervised-concept-drift-detection benchmark.

Drop this file into ``detectors/ffwin.py`` of the benchmark repository and add

    from .ffwin import FasanoFranceschiniWindow

to ``detectors/__init__.py``.

The benchmark drives detectors through a per-sample streaming interface
(``update(features) -> bool``), while the published FFWIN implementation
(``ff_pkg.SlidingWindowFasanoFranceschini``) operates on a whole array. This
adapter bridges the two: it buffers the incoming stream and, every ``step``
samples, compares a reference window against a current window using the
Fasano-Franceschini statistic from ``ff_pkg``, exactly as in Algorithm 1 of the
paper.

The statistic, the range-tree backend and the PCA projection are imported from
``ff_pkg`` rather than reimplemented, so results stay tied to the published
package.
"""

from collections import deque
from typing import List, Optional

import numpy as np

from .base import UnsupervisedDriftDetector

from ff_pkg.dim_reduction import PCAReducer
from ff_pkg.ff_core import test_statistic
from ff_pkg.matrix_util import rbind


class FasanoFranceschiniWindow(UnsupervisedDriftDetector):
    """
    FFWIN detects concept drift by testing whether a reference window and a
    current window are drawn from the same multivariate distribution, using the
    Fasano-Franceschini two-sample test with permutation-calibrated p-values.

    Because the test evaluates the joint distribution over all 2^d orthants, it
    responds to changes in the dependence structure of P(X) even when every
    univariate marginal is unchanged.

    Source: Bezerra, L. M.; Klautau, A.; Silva, A.; Goncalves, G. E. FFWIN: An
        Unsupervised Concept Drift Detector for Data Streams.
    """

    def __init__(
        self,
        n_samples: int = 250,
        step: int = 125,
        alpha: float = 0.05,
        n_permutations: int = 100,
        method: str = "b",
        pca_threshold: Optional[int] = 9,
        pca_target_dim: Optional[int] = 8,
        pca_mode: str = "concat",
        standardize: bool = True,
        hard_reset: bool = True,
        p_value_rule: str = "paper",
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Init a new FFWIN instance.

        :param n_samples: the window length W, i.e. the number of samples held by
            both the reference window and the current window
        :param step: the stride s between consecutive window comparisons; the
            current window starts s samples after the reference window
        :param alpha: the significance level; a drift is signalled when p < alpha
        :param n_permutations: the number of permutations B used to calibrate the
            null distribution of the test statistic
        :param method: the backend computing the statistic, 'b' for brute force or
            'r' for the range tree
        :param pca_threshold: the dimensionality d_0 at or above which PCA is
            applied, or None to never apply it
        :param pca_target_dim: the number k of principal components retained
        :param pca_mode: 'ref' fits PCA on the reference window only, 'concat'
            fits it on both windows stacked
        :param standardize: True to z-score both windows before projection, using
            statistics fitted on the same data as the PCA (see pca_mode)
        :param hard_reset: True to clear the buffer after an alarm, so that the
            windows spanning a detected change are not tested again
        :param p_value_rule: 'paper' for p = (1 + #{D_b >= D}) / (B + 1) as
            published, or 'randomized' for the tie-broken variant used by
            ff_pkg.ff_core.permutation_test
        :param n_jobs: the number of workers used for the permutation loop; 1
            runs it serially
        :param seed: the random seed
        """
        super().__init__(seed)
        if step < 1:
            raise ValueError("step must be >= 1")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if p_value_rule not in ("paper", "randomized"):
            raise ValueError("p_value_rule must be 'paper' or 'randomized'")

        self.n_samples = int(n_samples)
        self.step = int(step)
        self.alpha = float(alpha)
        self.n_permutations = int(n_permutations)
        self.method = str(method)
        self.pca_mode = str(pca_mode)
        self.standardize = bool(standardize)
        self.hard_reset = bool(hard_reset)
        self.p_value_rule = str(p_value_rule)
        self.n_jobs = int(n_jobs)

        self._reducer = PCAReducer(
            threshold=pca_threshold,
            target_dim=pca_target_dim,
            random_state=self.seed,
            fit_mode=self.pca_mode,
        )

        # A comparison needs the reference window (W samples starting at the
        # oldest buffered sample) and the current window (W samples starting s
        # later), so the buffer spans s + W samples.
        self._span = self.step + self.n_samples
        self._buffer = deque(maxlen=self._span)
        self._since_last_test = 0
        self._rng = np.random.default_rng(self.seed)
        self.time_step = 0

        # Per-comparison diagnostics, kept for the efficiency and calibration
        # analyses. The benchmark logger does not read these.
        self.history: List[dict] = []

    def update(self, features: dict) -> bool:
        """
        Update the detector with the given features.

        :param features: the features
        :return: True if a drift occurred, else False
        """
        self._buffer.append(np.fromiter(features.values(), dtype=float))
        self.time_step += 1
        self._since_last_test += 1

        if len(self._buffer) < self._span or self._since_last_test < self.step:
            return False

        self._since_last_test = 0
        window = np.asarray(self._buffer, dtype=float)
        reference = window[: self.n_samples]
        current = window[self.step : self.step + self.n_samples]

        drift = self._test(reference, current)
        if drift and self.hard_reset:
            self._buffer.clear()
            self._since_last_test = 0
        return drift

    def _test(self, reference: np.ndarray, current: np.ndarray) -> bool:
        """
        Test whether the two windows come from the same distribution.

        :param reference: the reference window
        :param current: the current window
        :return: True if the null hypothesis is rejected at level alpha
        """
        reference, current = self._project(reference, current)

        pooled = rbind(reference, current)
        n1, n2 = reference.shape[0], current.shape[0]
        observed = test_statistic(pooled, n1, n2, shuffle=False, method=self.method)
        p_value = self._permutation_p_value(pooled, n1, n2, observed)

        drift = bool(p_value < self.alpha)
        self.history.append(
            {
                "time_step": self.time_step,
                "D": observed,
                "p": p_value,
                "drift": drift,
            }
        )
        return drift

    def _project(self, reference: np.ndarray, current: np.ndarray):
        """
        Standardize and optionally project both windows onto k principal
        components. Standardization statistics are fitted on the same data as the
        PCA, so that the two windows always share one transformation.

        :param reference: the reference window
        :param current: the current window
        :return: the transformed reference and current windows
        """
        if self.standardize:
            basis = reference if self.pca_mode == "ref" else np.vstack(
                [reference, current]
            )
            mean = basis.mean(axis=0)
            scale = basis.std(axis=0)
            scale[scale == 0] = 1.0
            reference = (reference - mean) / scale
            current = (current - mean) / scale

        reference, current, _ = self._reducer.apply(reference, current)
        return reference, current

    def _permutation_p_value(
        self, pooled: np.ndarray, n1: int, n2: int, observed: int
    ) -> float:
        """
        Calibrate the observed statistic against its permutation null distribution.

        :param pooled: the two windows stacked
        :param n1: the size of the reference window
        :param n2: the size of the current window
        :param observed: the observed test statistic
        :return: the p-value
        """
        if self.n_permutations <= 0:
            return 1.0

        total = n1 + n2
        permutations = [
            self._rng.permutation(total).tolist() for _ in range(self.n_permutations)
        ]

        def statistic(order):
            return test_statistic(
                pooled, n1, n2, shuffle=False, method=self.method, s=order
            )

        if self.n_jobs == 1:
            values = [statistic(order) for order in permutations]
        else:
            from joblib import Parallel, delayed

            values = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(statistic)(order) for order in permutations
            )

        values = np.asarray(values)
        if self.p_value_rule == "paper":
            # p = (1 + #{D_b >= D}) / (B + 1), as published.
            return float(
                (1 + np.sum(values >= observed)) / (self.n_permutations + 1)
            )
        # Tie-broken variant: ties contribute a uniform fraction.
        n_greater = int(np.sum(values > observed))
        n_equal = int(np.sum(values == observed))
        return float(
            (n_greater + (1 + n_equal) * self._rng.random())
            / (1 + self.n_permutations)
        )

    def __str__(self):
        return "FasanoFranceschiniWindow"
