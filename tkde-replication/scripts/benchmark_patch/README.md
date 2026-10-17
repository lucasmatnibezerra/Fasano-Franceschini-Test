# Patches for the DFKI-NI benchmark

Two changes are needed before the revision run. Both were found by running the
pipeline end to end on Outdoor Objects, not by reading the code.

```bash
cd unsupervised-concept-drift-detection
git apply /path/to/0001-seed-iteration-and-electricity-csv.patch
```

## 1. `optimization/config_generator.py` — seeds were being discarded

`ConfigGenerator.__iter__` assigned `config["seed"] = self.seeds[i]`, where `i`
indexes the **parameter combination**, not the seed. Two consequences:

- With a fixed-configuration grid — which is exactly what the revision run uses,
  since the hyperparameters are already selected — there is a single combination,
  so only `seeds[0]` was ever used and the remaining nine were silently dropped.
  Every confidence interval in the revision depends on this being fixed.
- With more combinations than seeds, the same line raises `IndexError`.

The fix runs every configuration under every seed:

```python
seeds = self.seeds if self.seeds is not None else [None]
for combination in itertools.product(*all_parameters):
    config = {...}
    for seed in seeds:
        seeded = dict(config)
        seeded["seed"] = int(time.time()) if seed is None else seed
        yield seeded
```

Behaviour with `seeds=None` is unchanged, so the repository's original
`config.py` still works as before.

Verified: with `seeds=[11, 22]` and a single-valued grid, the run now logs two
rows per detector instead of one.

## 2. `datasets/electricity.py` — reads ARFF, but the archive ships CSV

`Electricity` is the only stream class that uses `stream.iter_arff` and expects
`elec.arff`. Every other class reads CSV. The USP archive contains
`Electricity.csv`, so `prepare_datasets.py` writes `elec.csv` and this class is
switched to `iter_csv` with explicit converters, matching the pattern used by
the other ten streams.

If you obtain the original `elec.arff` instead, revert this hunk and drop the
`Electricity.csv` entry from `SCHEMA` in `prepare_datasets.py`.

## Not patched, but worth knowing

**`convert_datasets.py` does not apply here.** It converts ARFF to CSV, and the
archive on hand is already CSV — but headerless, which is why
`prepare_datasets.py` exists. Do not run `convert_datasets.py`; it will find no
ARFF files and report failures for all eleven entries.

**The pinned requirements do not install on modern Python.** The repository pins
`river==0.11.1`, `numpy==1.23.1`, `pandas==1.4.3` — a 2022 stack that fails to
build on Python 3.12 and later. We verified the benchmark against **river
0.26.1** on Python 3.12: every API it uses is still present, and the one
behavioural change (`learn_one` now returns `None` rather than `self`) does not
affect this code, which discards the return value. Install a current stack
rather than hunting for Python 3.10.
