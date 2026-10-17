"""Turn the USP DS Repository archive into the files the benchmark expects.

The archive ships headerless CSVs. The DFKI-NI benchmark reads its streams with
``river.stream.iter_csv`` and a per-dataset ``converters`` map keyed by column
name, so it needs CSVs *with* headers, under its own filenames. The repository's
own ``convert_datasets.py`` bridges that gap by converting ARFF files, whose
attribute names become the CSV header — but the archive on hand contains CSV,
not ARFF, so that path does not apply.

This script does the equivalent directly: it extracts each stream, writes the
header the corresponding dataset class expects, and saves it under the expected
filename in ``datasets/files/``.

Usage:
    python scripts/prepare_datasets.py \
        --archive "USP DS Repository.zip" \
        --out harness/datasets/files

Add ``--normalize-insects`` to min-max scale the INSECTS features; read the note
under INSECTS_NOTE below before you do.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

INSECTS_NOTE = """
The benchmark expects INSECTS files named '..._balanced_norm.csv', i.e. the
NORMALIZED variant published by the USP DS Repository. The archive on hand holds
the raw balanced CSVs: feature values span roughly 0.004 to 8002, so they are
clearly not scaled.

Two options, in order of preference:

  1. Obtain the normalized ARFF/CSV variants from the USP DS Repository and use
     them directly. This reproduces the preprocessing behind the published
     baseline hyperparameters of Lukats et al., which the revision reuses.

  2. Run this script with --normalize-insects, which applies per-feature min-max
     scaling to [0, 1] over the whole stream. This is the conventional scaling
     for these files, but it is an assumption: it is computed over the full
     stream rather than online, and it may not match the published variant
     exactly. If you take this route, say so in the manuscript, because baseline
     detectors tuned on differently scaled data may not behave identically.

Whichever you choose, record it in the replication package.
"""


def insects_columns() -> list[str]:
    return [f"Att{i}" for i in range(1, 34)] + ["class"]


# source name in the archive -> (output filename, column names)
SCHEMA: dict[str, tuple[str, list[str]]] = {
    "Electricity.csv": (
        "elec.csv",
        [f"att{i}" for i in range(1, 9)] + ["class"],
    ),
    "NOAA.csv": (
        "NOAA.csv",
        [f"attribute{i}" for i in range(1, 9)] + ["class"],
    ),
    "Outdoor.csv": (
        "outdoor.csv",
        [f"att{i}" for i in range(1, 22)] + ["class"],
    ),
    "PokerHand.csv": (
        "poker-lsn.csv",
        ["s1", "r1", "s2", "r2", "s3", "r3", "s4", "r4", "s5", "r5", "class"],
    ),
    "Powersupply.csv": (
        "powersupply.csv",
        ["attribute0", "attribute1", "class"],
    ),
    "Rialto.csv": (
        "rialto.csv",
        [f"att{i}" for i in range(1, 28)] + ["class"],
    ),
    "INSECTS abrupt_balanced.csv": (
        "INSECTS-abrupt_balanced_norm.csv", insects_columns(),
    ),
    "INSECTS gradual_balanced.csv": (
        "INSECTS-gradual_balanced_norm.csv", insects_columns(),
    ),
    "INSECTS incremental_balanced.csv": (
        "INSECTS-incremental_balanced_norm.csv", insects_columns(),
    ),
    "INSECTS incremental-abrupt_balanced.csv": (
        "INSECTS-incremental-abrupt_balanced_norm.csv", insects_columns(),
    ),
    "INSECTS incremental-reoccurring_balanced.csv": (
        "INSECTS-incremental-reoccurring_balanced_norm.csv", insects_columns(),
    ),
}

# The benchmark reads Electricity with iter_arff. Since the archive provides CSV,
# datasets/electricity.py needs this one-line change.
ELECTRICITY_PATCH = '''
Patch datasets/electricity.py:

    -            filename="elec.arff",
    +            filename="elec.csv",

    -        return stream.iter_arff(
    -            self.full_path,
    -            target="class",
    -        )
    +        converters = {f"att{i}": float for i in range(1, 9)}
    +        converters["class"] = int
    +        return stream.iter_csv(
    +            self.full_path,
    +            converters=converters,
    +            target="class",
    +        )
'''


def find_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    """Locate a file inside the archive by basename, ignoring its directory."""
    for name in archive.namelist():
        if name.endswith("/") or "__MACOSX" in name:
            continue
        if Path(name).name == basename:
            return name
    return None


def read_ragged(handle, n_columns: int, source: str) -> pd.DataFrame:
    """
    Parse a headerless CSV whose rows may not all have the same field count.

    At least one file in the archive is malformed: line 22202 of
    ``Powersupply.csv`` carries a trailing comma and therefore parses as four
    fields instead of three, which aborts the C parser outright. Rather than
    silently coercing, this reports every row it repairs so the change can be
    recorded alongside the results.

    :param handle: an open binary file object
    :param n_columns: the number of columns the schema expects
    :param source: the file name, for reporting
    :return: the parsed frame
    """
    text = handle.read().decode("utf-8", "replace")
    rows, repaired = [], []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        # A trailing separator yields one empty field beyond the schema.
        while len(fields) > n_columns and fields[-1] == "":
            fields.pop()
            repaired.append(number)
        if len(fields) != n_columns:
            raise SystemExit(
                f"{source}: line {number} has {len(fields)} fields, expected "
                f"{n_columns}, and it is not a trailing-separator artifact:\n"
                f"  {line[:160]}"
            )
        rows.append(fields)

    if repaired:
        shown = ", ".join(str(n) for n in repaired[:5])
        more = f" (and {len(repaired) - 5} more)" if len(repaired) > 5 else ""
        print(f"    note: stripped a trailing separator on {len(repaired)} "
              f"line(s) of {source}: {shown}{more}")

    frame = pd.DataFrame(rows)
    for column in frame.columns:
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (ValueError, TypeError):
            pass  # genuinely non-numeric, e.g. a string class label
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="USP DS Repository.zip")
    parser.add_argument("--out", required=True, help="benchmark datasets/files")
    parser.add_argument(
        "--normalize-insects",
        action="store_true",
        help="min-max scale INSECTS features to [0,1]; read INSECTS_NOTE first",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, missing = [], []
    with zipfile.ZipFile(args.archive) as archive:
        for source, (target, columns) in SCHEMA.items():
            member = find_member(archive, source)
            if member is None:
                missing.append(source)
                continue

            with archive.open(member) as handle:
                frame = read_ragged(handle, len(columns), source)

            if frame.shape[1] != len(columns):
                raise SystemExit(
                    f"{source}: found {frame.shape[1]} columns, "
                    f"expected {len(columns)}. The schema map is wrong for this "
                    f"file; fix SCHEMA rather than forcing the header."
                )
            frame.columns = columns

            if args.normalize_insects and source.startswith("INSECTS"):
                features = frame.columns[:-1]
                low = frame[features].min()
                span = (frame[features].max() - low).replace(0, 1)
                frame[features] = (frame[features] - low) / span

            destination = out_dir / target
            frame.to_csv(destination, index=False)
            written.append((target, frame.shape))
            print(f"  {source}  ->  {target}  {frame.shape}")

    print(f"\nwrote {len(written)} files to {out_dir.resolve()}")
    if missing:
        print("MISSING from the archive:")
        for name in missing:
            print(f"  {name}")

    if not args.normalize_insects:
        print(INSECTS_NOTE)
    else:
        print("\nINSECTS features were min-max scaled to [0,1]. "
              "Record this in the replication package.")

    print(ELECTRICITY_PATCH)


if __name__ == "__main__":
    main()
