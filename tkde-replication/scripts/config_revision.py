"""Configuration for the FFWIN TKDE revision run.

Replaces ``config.py`` in the DFKI-NI benchmark repository. Unlike the original
configuration, this one does NOT perform a hyperparameter search: the values
were already selected and published, so every detector runs at its chosen
configuration across a fixed set of seeds. That turns a multi-month sweep into
a single pass.

Three experiments are defined. Select one by editing ``Configuration`` at the
bottom, or by importing the corresponding class from a driver script.

    EXPERIMENT_MAIN      all detectors, all streams, 10 seeds
    EXPERIMENT_PCA       FFWIN only, PCA ablation (pca_mode x k)
    EXPERIMENT_BACKEND   FFWIN only, brute force vs range tree

Run with:  python main.py revision_main
"""

from datasets import (
    Electricity,
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalReoccurringBalanced,
    NOAAWeather,
    OutdoorObjects,
    PokerHand,
    Powersupply,
    RialtoBridgeTimelapse,
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


# Fixed seeds, recorded in every results row. Without these the run cannot be
# reproduced, which is what forced this rerun in the first place.
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

ALL_STREAMS = [
    Electricity(),
    InsectsAbruptBalanced(),
    InsectsGradualBalanced(),
    InsectsIncrementalAbruptBalanced(),
    InsectsIncrementalBalanced(),
    InsectsIncrementalReoccurringBalanced(),
    NOAAWeather(),
    OutdoorObjects(),
    PokerHand(),
    Powersupply(),
    RialtoBridgeTimelapse(),
]


def _ffwin(**overrides):
    """Build an FFWIN optimizer at the published configuration, with overrides."""
    parameters = {
        "n_samples": Parameter("n_samples", value=250),
        "step": Parameter("step", value=125),
        "alpha": Parameter("alpha", value=0.05),
        "n_permutations": Parameter("n_permutations", value=100),
        "method": Parameter("method", value="b"),
        "pca_threshold": Parameter("pca_threshold", value=9),
        "pca_target_dim": Parameter("pca_target_dim", value=8),
        "pca_mode": Parameter("pca_mode", value="concat"),
    }
    parameters.update(overrides)
    return ModelOptimizer(
        base_model=FasanoFranceschiniWindow,
        parameters=list(parameters.values()),
        seeds=SEEDS,
        n_runs=1,
    )


# ---------------------------------------------------------------------------
# Main experiment: every detector at its real-stream configuration.
#
# Each baseline value is a point of the grid of Lukats et al., the same on every
# stream, with a 250-sample window as for FFWIN. Only SPLL and UDetect coincide
# with a configuration Lukats et al. report as best on some stream (their
# Tables 22-23); the baselines are not tuned here.
# ---------------------------------------------------------------------------
class EXPERIMENT_MAIN:
    streams = ALL_STREAMS
    n_training_samples = 1000
    models = [
        ModelOptimizer(
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", value=250),
                Parameter("const", value=1.0),
                Parameter("max_depth", value=2),
                Parameter("threshold", value=0.5),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=ClusteredStatisticalTestDriftDetectionMethod,
            parameters=[
                Parameter("n_samples", value=250),
                Parameter("confidence", value=0.01),
                Parameter("feature_proportion", value=0.1),
                Parameter("n_clusters", value=2),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", value=250),
                Parameter("recent_samples_proportion", value=0.5),
                Parameter("threshold", value=0.7),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        # EDFS, NN-DVI and UCDD appear in the paper's comparison table but were
        # not evaluated. Reviewer 3 flagged the EDFS omission specifically. All
        # three are already implemented here, so including them is nearly free.
        ModelOptimizer(
            base_model=EDFS,
            parameters=[
                Parameter("n_samples", value=250),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=ImageBasedDriftDetector,
            parameters=[
                Parameter("n_samples", value=250),
                Parameter("n_permutations", value=20),
                Parameter("update_interval", value=100),
                Parameter("n_consecutive_deviations", value=1),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=NNDVI,
            parameters=[
                Parameter("n_samples", value=250),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=OneClassDriftDetector,
            parameters=[
                Parameter("n_samples", value=250),
                Parameter("threshold", value=0.3),
                Parameter(
                    "outlier_detector_kwargs",
                    value={"nu": 0.5, "kernel": "rbf", "gamma": "auto"},
                ),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=SemiParametricLogLikelihood,
            parameters=[
                Parameter("n_samples", value=250),
                Parameter("n_clusters", value=2),
                Parameter("threshold", value=0.05),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=UCDD,
            parameters=[
                Parameter("n_samples", value=250),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        ModelOptimizer(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", value=50),
                Parameter("n_samples", value=250),
                # False matches the shipped real-stream results; True never fires.
                Parameter("disjoint_training_windows", value=False),
            ],
            seeds=SEEDS,
            n_runs=1,
        ),
        _ffwin(),
    ]


# ---------------------------------------------------------------------------
# PCA ablation. Reviewers 1 and 2 both asked what the projection does to
# correlation-drift sensitivity: fitting on the reference window may hide a
# change in the current window's covariance, while fitting on both windows lets
# the basis absorb the drift. The published default is 'concat', i.e. the second
# case, so this ablation tests the shipped configuration.
#
# INSECTS abrupt (d=33) is the only real stream with exact drift timestamps, so
# ground-truth metrics are available for every cell of the sweep.
# ---------------------------------------------------------------------------
class EXPERIMENT_PCA:
    streams = [InsectsAbruptBalanced(), RialtoBridgeTimelapse()]
    n_training_samples = 1000
    models = [
        _ffwin(
            pca_mode=Parameter("pca_mode", values=["ref", "concat"]),
            pca_target_dim=Parameter("pca_target_dim", values=[2, 4, 6, 8, 12]),
        )
    ]


# ---------------------------------------------------------------------------
# Backend comparison. The range-tree backend is implemented but was never used:
# the paper reports method='b' only, and both Reviewer 3 and the AE asked for a
# concrete answer on computational cost. Both backends compute the same
# statistic, so detection results must match and only runtime should differ.
# ---------------------------------------------------------------------------
class EXPERIMENT_BACKEND:
    streams = [InsectsAbruptBalanced()]
    n_training_samples = 1000
    models = [_ffwin(method=Parameter("method", values=["b", "r"]))]


# The benchmark's runner imports this name.
Configuration = EXPERIMENT_MAIN
