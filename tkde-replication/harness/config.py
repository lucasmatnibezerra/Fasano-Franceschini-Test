"""Shard-aware configuration for the FFWIN TKDE revision run.

The benchmark's runner walks streams and models serially inside one process. To
use more than one core we shard the work by (stream, seed) and launch one
process per shard, each selecting its slice through the environment:

    FFWIN_STREAMS   comma-separated stream class names   (default: all eleven)
    FFWIN_SEEDS     comma-separated integer seeds        (default: all ten)
    FFWIN_MODELS    comma-separated detector keys        (default: all)

No hyperparameter search is performed. Every detector runs at the configuration
selected and published for it, so this is a single pass rather than a sweep.
"""

import os

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

ALL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]

STREAMS = {
    "Electricity": Electricity,
    "InsectsAbruptBalanced": InsectsAbruptBalanced,
    "InsectsGradualBalanced": InsectsGradualBalanced,
    "InsectsIncrementalAbruptBalanced": InsectsIncrementalAbruptBalanced,
    "InsectsIncrementalBalanced": InsectsIncrementalBalanced,
    "InsectsIncrementalReoccurringBalanced": InsectsIncrementalReoccurringBalanced,
    "NOAAWeather": NOAAWeather,
    "OutdoorObjects": OutdoorObjects,
    "PokerHand": PokerHand,
    "Powersupply": Powersupply,
    "RialtoBridgeTimelapse": RialtoBridgeTimelapse,
}


def _selected_seeds():
    raw = os.environ.get("FFWIN_SEEDS", "")
    return [int(s) for s in raw.split(",") if s.strip()] or ALL_SEEDS


def _selected_streams():
    raw = os.environ.get("FFWIN_STREAMS", "")
    names = [s.strip() for s in raw.split(",") if s.strip()] or list(STREAMS)
    return [STREAMS[name]() for name in names]


SEEDS = _selected_seeds()


# Four of the detectors carry no randomness: across ten seeds on Poker Hands
# (829k samples) BNDM, OCDD and UDetect returned byte-identical rows, and so did
# FFWIN. Repeating a deterministic detector ten times produces ten copies of one
# number, not ten samples, so they run once and their interval is zero by
# construction. FFWIN is kept at the full seed set anyway: its permutation test
# is stochastic by design, and the reviewers asked specifically about its
# variability, so we report it rather than assume stability generalises.
DETERMINISTIC = {"bndm", "ocdd", "udetect"}


def _model(base, key=None, **params):
    seeds = SEEDS[:1] if key in DETERMINISTIC else SEEDS
    # A value may be a bare scalar or an already-built Parameter, so that an
    # ablation can sweep one axis while every other axis stays pinned.
    return ModelOptimizer(
        base_model=base,
        parameters=[v if isinstance(v, Parameter) else Parameter(k, value=v)
                    for k, v in params.items()],
        seeds=seeds,
        n_runs=1,
    )


# Real-stream configuration for each detector. Every baseline value is a point
# of the grid of Lukats et al., the same on every stream, with a 250-sample
# window as for FFWIN; the baselines are not tuned here. Only the SPLL and
# UDetect values (with disjoint_training_windows=False) coincide with a
# configuration Lukats et al. report as best on some stream (their Tables 22-23).
MODELS = {
    "ffwin": lambda: _model(
        FasanoFranceschiniWindow, key="ffwin",
        n_samples=250, step=125, alpha=0.05, n_permutations=100, method="b",
        pca_threshold=9, pca_target_dim=8, pca_mode="concat",
        # Parallelism is at the process level, so keep the permutation loop
        # serial to avoid oversubscribing the machine.
        n_jobs=1,
    ),
    # Revised configuration: stride s = W. With s = W/2 the current window shares
    # half its samples with the reference, which makes the permutation test
    # conservative and halves the contrast at a change point. Every other axis
    # at the published value. Run as experiment ``revision_stride``.
    "ffwin_stride": lambda: _model(
        FasanoFranceschiniWindow, key="ffwin_stride",
        n_samples=250, step=250, alpha=0.05, n_permutations=100, method="b",
        pca_threshold=9, pca_target_dim=8, pca_mode="concat", n_jobs=1,
    ),
    # PCA ablation (Reviewers 1 and 2). Fitting the subspace on the reference
    # window alone risks missing drift that lives outside it; fitting on both
    # windows stacked lets the basis absorb the very drift it should expose.
    # Every other axis is pinned to the published configuration.
    "ffwin_pca": lambda: _model(
        FasanoFranceschiniWindow, key="ffwin_pca",
        n_samples=250, step=125, alpha=0.05, n_permutations=100, method="b",
        pca_threshold=9, n_jobs=1,
        pca_mode=Parameter("pca_mode", values=["ref", "concat"]),
        pca_target_dim=Parameter("pca_target_dim", values=[2, 4, 6, 8, 12]),
    ),
    "spll": lambda: _model(
        SemiParametricLogLikelihood, key="spll", n_samples=250, n_clusters=2, threshold=0.05,
    ),
    "d3": lambda: _model(
        DiscriminativeDriftDetector2019,
        n_reference_samples=250, recent_samples_proportion=0.5, threshold=0.7,
    ),
    "ibdd": lambda: _model(
        ImageBasedDriftDetector, key="ibdd",
        n_samples=250, n_permutations=20, update_interval=100,
        n_consecutive_deviations=1,
    ),
    "csddm": lambda: _model(
        ClusteredStatisticalTestDriftDetectionMethod, key="csddm",
        n_samples=250, confidence=0.01, feature_proportion=0.1, n_clusters=2,
    ),
    "ocdd": lambda: _model(
        OneClassDriftDetector, key="ocdd",
        n_samples=250, threshold=0.3,
        outlier_detector_kwargs={"nu": 0.5, "kernel": "rbf", "gamma": "auto"},
    ),
    "udetect": lambda: _model(
        UDetect, key="udetect", n_windows=50, n_samples=250, disjoint_training_windows=True,
    ),
    # The entry above keeps disjoint_training_windows=True, the implementation's
    # default, only to document that it silences UDetect: it raises no alarm on
    # any of our streams. Lukats et al. report the same (all 126 of their UDetect
    # runs without a detection used disjoint windows; Sec. 6.1, Table 7), and the
    # configuration they report as most accurate on Elec2, n_samples=250 and
    # n_windows=50, uses False (their Table 23). The paper reports this entry.
    "udetect_joint": lambda: _model(
        UDetect, key="udetect_joint", n_windows=50, n_samples=250,
        disjoint_training_windows=False,
    ),
    # EDFS was excluded from the submitted experiments without a cost argument.
    # It has no published optimum in the benchmark, so we use the defaults of
    # Korycki and Krawczyk. Random subspace selection makes it stochastic.
    "edfs": lambda: _model(
        EDFS, key="edfs", n_subspaces=10, feature_percentage=0.1,
        alpha=0.005, window_size=100,
    ),
    "bndm": lambda: _model(
        BayesianNonparametricDetectionMethod, key="bndm",
        n_samples=250, const=1.0, max_depth=2, threshold=0.5,
    ),
    # EDFS, NN-DVI and UCDD are excluded from the run on measured cost grounds. On
    # Outdoor Objects (4,000 samples, 2 seeds) NN-DVI took 683 s against ~13 s
    # for every other baseline, which extrapolates to roughly 13 days over the
    # full corpus at ten seeds. This matches the characterisation already in the
    # paper's related work, now backed by measurement. UCDD had not finished the
    # same smoke test. Both can be re-enabled here if the budget allows.

}


def _selected_models():
    raw = os.environ.get("FFWIN_MODELS", "")
    keys = [s.strip() for s in raw.split(",") if s.strip()] or list(MODELS)
    return [MODELS[key]() for key in keys]


class Configuration:
    streams = _selected_streams()
    n_training_samples = 1000
    models = _selected_models()
