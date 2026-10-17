# Scripts

Reference for each script in this directory. The package `README.md` one level
up gives the commands that regenerate every table of the manuscript; this file
says what each script does and what it assumes. Run everything from the package
root.

## Real streams

**`launch.py`** — runs the benchmark in parallel shards, one process per
(stream, seed group), and skips shards whose rows already exist. Poker Hands
(829k samples, about 63% of the compute) is split one seed per shard.
`--experiment` names the run (the suffix of the results CSVs) and `--models`
restricts it to keys of `harness/config.py`, e.g.
`--experiment revision_stride --models ffwin_stride` for the revised FFWIN.
Outputs go to `harness/results/`; logs to `scripts/logs/`.

**`make_tables.py`** — every real-stream table and the Friedman/Nemenyi report
from the results CSVs. `--override DETECTOR=EXPERIMENT` takes one detector's
rows from another experiment (the manuscript takes FFWIN from
`revision_stride`). Confidence intervals are 95% normal approximations over
seeds; deterministic detectors have one seed and print no interval.

**`bench_backends.py`** — times one window comparison for the orthant
histogram, the compiled range scan and the spatial index over a grid of `k`
(Table XIII). `FF_RANGE_IMPL=numba` selects the compiled scan.

**`prepare_datasets.py`** — turns the USP DS Repository archive into the files
the benchmark reads: adds the header each dataset class expects, renames to the
expected filename, repairs one malformed row in `Powersupply.csv`, and checks
every row count against the declared `n_samples`. Use it instead of the
benchmark's `convert_datasets.py`, which converts ARFF while the archive is
already (headerless) CSV. Read what it prints before accepting
`--normalize-insects`.

**`config_revision.py`** — a class-based form of the same configurations
(`EXPERIMENT_MAIN`, `EXPERIMENT_PCA`, `EXPERIMENT_BACKEND`) for running the
benchmark's `main.py` serially, without the launcher. The shard-aware
`harness/config.py` is the one the reported runs used.

**`ffwin.py`** — the adapter that exposes FFWIN through the benchmark's
per-sample `update(features) -> bool` interface; identical to
`harness/detectors/ffwin.py`. It buffers `s + W` samples, compares the reference
window with the current one every `s` samples using `ff_pkg`, and clears the
buffer after an alarm (`hard_reset`).

**`test_adapter.py`** — checks the adapter against streams whose answer is
known: a pure correlation drift with stable marginals is detected and nothing
fires before it, a stationary stream stays quiet, `hard_reset` prevents
re-alarming on the same change, both p-value rules agree, and the two backends
return identical statistics. Run it where `ff_pkg` and `detectors.base` are
importable (e.g. from `harness/`).

## Synthetic scenarios

**`run_synthetic.py`** — the eight detectors at their published configurations
on S1–S4, ten realisations each; writes `synthetic/results/`. `--window` and
`--step` move FFWIN only, so a sweep never retunes the baselines.

**`make_synth_table.py`** — renders a results directory of `run_synthetic.py` as
MDR/MTD/MTFA/MTR per scenario.

**`run_null.py`** — every detector on drift-free streams (`run`, then `report`
for Table V); records FFWIN's number of window comparisons so its false alarms
can be set against `alpha` times that number.

**`tune_synthetic.py`** — the hyperparameter search: `tune` runs every
configuration on the tuning seeds, `select` keeps the best per (detector,
scenario) by detection F1, `eval` scores it on the held-out seeds, `probe`
estimates the cost of the grid first. `--offset` shifts each realisation by a
seeded random offset (FFWIN only; see the package README). Every run is stored
as JSON with its alarm positions, so F1 at any tolerance can be recomputed
without rerunning.

**`select_single_config.py`** — one configuration per detector, the best mean F1
over all four scenarios (`select`, `eval`, `table`; Table VI). With `--offset`
FFWIN's runs come from the shifted streams and the baselines keep theirs.

**`make_tuning_table.py`** — the per-scenario selection as a table (the oracle
ceiling), the selected configurations, and F1 at tolerances 500–2000.

**`offset_control.py`** — scores one FFWIN configuration on shifted streams,
with the same offsets as `--offset`; the diagnostic that exposed the
grid alignment.

**`power_curve.py`** — oracle power of the FF test: two disjoint windows placed
on either side of each prescribed drift, per scenario and window size, with a
same-regime pair as the null control.

**`make_pareto.py`** — accuracy against number of alarms on the real streams,
with the Pareto front per stream. Not in the manuscript.

**`run_all_remote.sh`** — the unattended driver used for the first synthetic
programme on the lab server (published-configuration campaign, window sweep,
power curve). Kept for the record; the commands above supersede it.

## Patches

**`benchmark_patch/`** — required changes to the DFKI-NI benchmark, already
applied in `harness/`. Without the seed patch the benchmark runs only the first
seed of each configuration and discards the rest, so no confidence interval can
be computed. See `benchmark_patch/README.md`.

**`ff_pkg_patch/`** — the three corrections to the released `ff_pkg`, already
applied in `detector/ff_pkg/`: the p-value as printed in the paper, a parallel
permutation routine that could not run, and the missing `rtree`/`numba`
dependencies. See `ff_pkg_patch/README.md`.

## Environment notes

The benchmark pins `river==0.11.1`, `numpy==1.23.1`, `pandas==1.4.3`, a 2022
combination that does not build on Python 3.12. The runs used river 0.26.1 on
Python 3.12; every API the benchmark uses still exists, and the one behavioural
change (`learn_one` returns `None` instead of `self`) does not affect it.

`ff_pkg` imports `rtree` and `numba` at module level even for the brute-force
backend, so both must be installed; `rtree` also needs the libspatialindex C
library.
