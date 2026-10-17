# FFWIN — replication package

Replication material for *FFWIN: An Unsupervised Concept Drift Detector for Data
Streams* (TKDE-2026-03-0755, revised manuscript).

Every table in the manuscript regenerates from this package, and every number
behind it is a stored alarm position, not a summary. Where something does
**not** reproduce, that is stated rather than implied (last section).

This package is published as the `tkde-replication/` directory of the repository
the manuscript cites, <https://github.com/lucasmatnibezerra/Fasano-Franceschini-Test>,
so the footnote in Sec. IV keeps pointing at a live URL.

## One change to the detector in this revision

The first submission ran FFWIN with a step half the window (`W=250, s=125`).
With `s < W` the reference and current windows share `W - s` samples: they are
more alike than independent draws, so the permutation test is conservative, and
at a change point at most a fraction `s/W` of the current window comes from the
new regime. On drift-free streams the overlapping detector raises no false alarm
where `alpha` allows 15.9; with disjoint windows (`s = W`) the count matches
`alpha` times the number of comparisons (Table V). The revised detector
therefore uses `s = W`. Both configurations are shipped and both sets of tables
regenerate: the manuscript reports the revised one, and `tables/first_submission/`
holds the other, so the effect of the change can be checked directly.

## Contents

| Path | What it is |
|---|---|
| `detector/ff_pkg/` | The detector: the Fasano–Franceschini statistic, the sliding-window pipeline, the permutation test, both evaluation backends. |
| `harness/` | The DFKI unsupervised-drift benchmark with our patches applied, the baseline detectors as evaluated, and the shard-aware `config.py`. |
| `scripts/` | Launchers, table generators, the synthetic experiments, the backend benchmark, dataset preparation, adapter tests, and the patch sets. |
| `results/` | Real-stream outputs, one CSV per (stream, detector, experiment); every row is one seed and carries the full alarm list. `*_revision_main.csv`: the eight detectors at their published configuration (FFWIN with `s=125`). `FasanoFranceschiniWindow_revision_stride.csv`: FFWIN with `s=W`. `*_revision_pca.csv`: the PCA ablation. |
| `synthetic/generate_datasets.py`, `synthetic/drifts.csv` | The generator of the four scenarios S1–S4 and their prescribed drift times. |
| `synthetic/results*/` | The eight detectors at their published configurations on S1–S4, ten seeds (see "Synthetic: published configurations"). |
| `synthetic/null/` | Every detector on drift-free streams, ten seeds (Table V). |
| `synthetic/tuning/` | The hyperparameter search: every configuration on the tuning seeds, the selections, and the held-out evaluations (Table VI and the per-scenario ceiling). |
| `tables/` | Every table of the manuscript as generated from the above, plus the Friedman/Nemenyi report. |
| `tables/first_submission/` | The real-stream tables with FFWIN at the first-submission step `s=125`. |
| `datasets/MANIFEST.sha256` | SHA-256 of each prepared real stream. |
| `logs/` | Selection logs of the last tuning passes, as written on the machine that ran them. |
| `synthetic/scored/`, `synthetic/alarms/` | Legacy outputs of the synthetic campaign behind the first submission; not used by the revised manuscript (last section). |

## Environment

Python 3.12. The versions the reported runs used:

    river 0.26.1, numpy 2.5.3, pandas 3.0.5, scipy, scikit-learn, joblib,
    rtree, numba, tqdm

    python3.12 -m venv venv
    ./venv/bin/pip install river numpy pandas scipy scikit-learn joblib rtree numba tqdm

`rtree` needs libspatialindex (`apt install libspatialindex-dev`), and is only
required by the range-query backend, which Sec. VI-G shows should not be used.

Set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL/Numba equivalents) whenever you run
many workers. The synthetic scripts and the launcher set them for their own
workers; without them 60 workers on 64 cores ran at a load of 260.

## The real streams

The eleven streams are third party and are not redistributed here. Rebuild them
from the USP DS Repository and verify against the manifest:

    python scripts/prepare_datasets.py \
        --archive "USP DS Repository.zip" \
        --out harness/datasets/files \
        --normalize-insects
    sha256sum -c datasets/MANIFEST.sha256

## Regenerating every table, with no computation

All commands run from the package root.

| Manuscript | File | Command |
|---|---|---|
| Table V (drift-free streams) | `tables/table_null.tex` | `python scripts/run_null.py report` |
| Table VI (one configuration per detector) | `tables/table_synth_single.tex` | `python scripts/select_single_config.py table --offset` |
| Per-scenario ceiling quoted in Sec. VI-A | `tables/table_synth_tuned.tex` | `python scripts/make_tuning_table.py --offset` |
| Sec. VI-A, first-submission configuration | `tables/table_synth.tex` | `python scripts/make_synth_table.py --results synthetic/results` |
| Tables VII–XI, Friedman/Nemenyi | `tables/table_{insects_gt,accuracy,lpd,alarm_counts,classifier}.tex`, `tables/stats_friedman.txt` | see below |
| Table XII (PCA ablation) | `tables/table_pca_ablation.tex` | see below |

The real-stream tables take the baselines from `revision_main` and FFWIN from
`revision_stride`:

    python scripts/make_tables.py --results results --experiment revision_main \
        --override FasanoFranceschiniWindow=revision_stride \
        --only insects accuracy lpd alarms classifier friedman --out tables
    python scripts/make_tables.py --results results --experiment revision_pca \
        --only pca --out tables

Drop `--override` and write to `tables/first_submission` to get the same tables
with FFWIN at `s=125`. Keep `--only` on the first command: without it the script
also rebuilds the PCA table from the wrong experiment.

`make_tuning_table.py` and `select_single_config.py table` also write the
selected configurations (`*_configs.csv`, `synthetic/tuning/selected_*.json`)
and F1 at tolerances of 500, 1000, 1500 and 2000 samples
(`tables/table_synth_tuned_delta_sensitivity.csv`).

## Re-running the real-stream evaluation

    python scripts/launch.py --status                       # what is already done
    python scripts/launch.py --dry-run                      # the shard plan
    python scripts/launch.py --workers N                    # all eight detectors, revision_main
    python scripts/launch.py --experiment revision_stride --models ffwin_stride --workers N

The launcher shards by (stream, seed group), runs `harness/main.py` once per
shard, and is resumable: it skips what is already in `harness/results/`. Its
outputs land in `harness/results/`; point `make_tables.py --results` there, or
copy them into `results/`. Configurations are frozen in `harness/config.py`;
`--models` takes its keys (`ffwin_stride` is the revised FFWIN). Seeds are
`11, 22, ..., 110`; BNDM, OCDD and UDetect have no stochastic component and run
one seed by construction.

Cost: the full `revision_main` run takes about 6 h wall-clock at 40 workers on
a 64-core machine, dominated by FFWIN on Poker Hands (829k samples); the
`revision_stride` run of FFWIN alone took 2.3 h at 22 workers.

    python scripts/bench_backends.py --data harness/datasets/files/INSECTS-abrupt_balanced_norm.csv

reproduces the cost comparison of Table XIII. Set `FF_RANGE_IMPL=numba` to time
the compiled scan instead of the spatial index.

## Synthetic: the scenarios

`synthetic/generate_datasets.py` builds the four scenarios of Sec. V-B, d = 8,
N = 40,000: S1 `copula` (gradual), S2 `graph_precision` (abrupt conditional
dependence), S3 `dcc_dynamic` (incremental), S4 `hmm_regimes` (recurring). In
S1 and S4 every marginal is N(0, 1) in every regime, and in S3 every marginal
variance stays near 0.45. S2 samples from the inverse of its precision matrix
without rescaling, so its marginal variances (0.38-0.66) change with the regime,
by up to 1.47x at a drift; the manuscript states this (Sec. V-B). The seed draws the stream, so ten
seeds are ten distinct realisations. Ground truth is the generator's
`drift_start_indices` minus the leading `0`, which marks the start of the first
concept. S3 annotates both the start and the end of each of its five ramps, so
it has ten drift times; the end of a ramp is not a change, which caps any
detector's recall there.

Detection quality is scored two ways. The benchmark's MDR, MTD, MTFA and MTR
(`harness/metrics/drift.py`) credit the first alarm after a drift at any delay,
so a detector that fires periodically detects everything. Detection F1
(`tune_synthetic.detection_f1`) counts the first alarm within `delta` samples of
a drift as a true positive and every other alarm as a false positive; `delta` was
fixed at 1,000 before any evaluation ran.

## Synthetic: published configurations

    python scripts/run_synthetic.py --workers 6        # writes synthetic/results/
    python scripts/make_synth_table.py --results synthetic/results

Every detector at the configuration it uses on the real streams, ten seeds per
scenario, FFWIN at `W=250, s=125`. The shipped `synthetic/results/` ran UDetect
with `disjoint_training_windows=True`, which silences it on every stream;
`synthetic/results_udetect_joint/` is the same run with the flag cleared, as on
the real streams, and is what Sec. VI-A quotes for UDetect. `run_synthetic.py`
now defaults to the cleared flag. `synthetic/results_w500/` and
`results_w1000/` hold an early FFWIN window sweep at `s = W/2` on three seeds.

## Synthetic: drift-free streams (Table V)

    python scripts/run_null.py run --workers 30
    python scripts/run_null.py report

One stationary stream per seed from the same generator (a single regime, block
correlation 0.35), every alarm false. For FFWIN the script records the number of
window comparisons, so the observed false alarms are set against `alpha` times
that number.

## Synthetic: tuning (Table VI)

Every detector runs its whole grid on the tuning seeds (11, 22, 33) of all four
scenarios: the upstream benchmark grid for each baseline, with both settings of
UDetect's flag and OCDD extended with `nu` in {0.1, 0.05} (the benchmark fixes
0.5, at which OCDD fires once per window on any stream); for FFWIN, `W` in
{250, 500, 1000} x `s` in {W, W/2, W/4} x `alpha` in {0.05, 0.01}. Selected
configurations are scored on the held-out seeds 44–110, which play no part in
selection.

    python scripts/tune_synthetic.py tune --workers 60           # all detectors
    python scripts/tune_synthetic.py select --delta 1000         # best per scenario
    python scripts/tune_synthetic.py eval --delta 1000 --workers 60

**The FFWIN grid control.** With `s = W` FFWIN compares windows on a fixed grid
(multiples of `W` from the start of the stream), and the drift times of S2–S4
fall on multiples of 1,000. At `W = 1000` every comparison then pits a pure
old-regime window against a pure new-regime one, every detection lands at
exactly `t + 999`, and F1 on S2 and S4 reads 0.97 and 0.95. No real stream lines
its drifts up with a detector's grid, so FFWIN is tuned and scored on streams
shifted by a seeded random offset in [1, 999] per (scenario, seed); the
baselines react sample by sample (35–63 distinct detection delays) and keep
their unshifted runs.

    python scripts/tune_synthetic.py tune --offset --detectors FasanoFranceschiniWindow --workers 40
    python scripts/tune_synthetic.py select --offset --delta 1000
    python scripts/tune_synthetic.py eval --offset --delta 1000 --workers 40
    python scripts/select_single_config.py eval --offset --delta 1000 --workers 40
    python scripts/make_tuning_table.py --offset

`select_single_config.py` picks one configuration per detector, the best mean
F1 over the four scenarios on the tuning seeds, which is Table VI;
`make_tuning_table.py` is the per-scenario oracle quoted as a ceiling. Every run
is stored as JSON with its alarm positions under `synthetic/tuning/<phase>/_raw/`,
and every command resumes from what is there.

Order of events, disclosed in the manuscript: a first pass covered the upstream
grids and FFWIN at `s = W/2` only; the strides `W` and `W/4` (both in the
manuscript's declared grid) and OCDD's smaller `nu` were added in a second pass
after the first had been evaluated; the offset control was added after the
second pass showed the grid alignment. `synthetic/tuning/*_pass1.*` keep the
first pass. Cost: the first pass (3,000 runs) took 40 h on 60 workers, slowed by
missing thread caps; the offset pass (216 FFWIN runs) took 3.5 h on 40.

## Synthetic: diagnostics

    python scripts/power_curve.py        # oracle power: disjoint windows placed around each drift
    python scripts/offset_control.py OUT.jsonl --scenarios S1 S2 S3 S4 --alpha 0.01

`power_curve.py` separates the power of the test from the windowing (Sec. VI-A,
"Limits"). `offset_control.py` scores one FFWIN configuration on shifted
streams with the same offsets as `--offset`. `make_pareto.py` draws accuracy
against number of alarms on the real streams; it is not in the manuscript.

## Reproducibility, as measured

Seeds fix the results exactly. Part of the real-stream run was executed on
Windows and part on Linux, with different NumPy builds, and for every detector
except three the two platforms returned **byte-identical rows, including the
full alarm timestamp lists**. The exceptions are CSDDM, SPLL and UDetect on
Poker Hands, where floating-point differences move the alarm count by under 1%
(for example 3049 against 3055); those three were re-run so that the shipped
corpus comes from a single platform. Every table under `tables/` was regenerated
from the shipped outputs with the commands above and matches the manuscript row
for row.

The synthetic streams are the exception. `synthetic/generate_datasets.py`
factors each correlation matrix with `numpy.linalg.eigh`, whose eigenvector
signs, and the basis it picks inside a repeated eigenvalue (the block
correlation matrices have several), depend on the LAPACK build. A seed
therefore fixes the distribution of a synthetic stream on every platform, but
its realization only on a given build. Every shipped synthetic result (Tables V
and VI, the tuning runs) was produced on one Linux server and regenerates
exactly from the stored alarm lists; re-running a detector elsewhere gives
statistically equivalent, not identical, alarms. For FFWIN at W = 250, s = W on
S3, the seven held-out seeds give mean F1 0.25 on a Windows workstation against
0.16 on the server, with a standard deviation of 0.12 to 0.14 on each. A
Cholesky factor would remove the dependence; we left the generator unchanged
because that would change every stream behind the reported tables.

## Corrections to the released implementation

Three defects were found while rebuilding the pipeline. All are fixed here, and
none affects the reported results, which come from the corrected serial path.

1. The permutation p-value now follows the equation printed in the paper,
   `(1 + #{D_b >= D}) / (B + 1)`, rather than a randomized tie-broken variant.
2. `permutation_test_parallel` passed a local function to `multiprocessing.Pool`
   and could not execute; it now runs.
3. `rtree` and `numba` were imported but not declared as dependencies, so a
   fresh install left the package unimportable.

The change of step (`s = W`) is a change of configuration, not a defect fix; it
is described at the top of this file.

## What this package does not reproduce

- **The synthetic table of the first submission.** Its numbers came from an
  earlier implementation and scoring protocol; `synthetic/scored/` and
  `synthetic/alarms/` hold those legacy outputs, and they do not reproduce that
  table. The revised manuscript replaces it with Tables V and VI, which do
  regenerate.
- **Absolute runtimes** (Table XIII, the cost figures above) depend on the
  hardware; the ratios between backends are what `bench_backends.py` reproduces.
