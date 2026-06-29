import pandas as pd
import os
from pathlib import Path


def combine_csv_files(inputs: list | str, output_file: str) -> None:
    """Combine multiple CSV files into one by concatenating rows.

    Args:
        inputs:      List of file paths, or a single folder path (all *.csv in that folder).
        output_file: Path to the combined output CSV.
    """
    # Resolve inputs to a list of file paths
    if isinstance(inputs, str) and os.path.isdir(inputs):
        files = sorted(Path(inputs).glob("*.csv"))
    else:
        files = [Path(f) for f in inputs]

    if not files:
        print("Error: no CSV files found.")
        return

    dfs = []
    for f in files:
        if not f.is_file():
            print(f"Warning: file not found, skipping -> {f}")
            continue
        dfs.append(pd.read_csv(f, sep=";", engine="python"))
        print(f"Loaded {f.name} ({len(dfs[-1])} rows)")

    if not dfs:
        print("Error: no valid files loaded. Aborting.")
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    combined.to_csv(output_file, sep=";", index=False, encoding="utf-8")

    total = sum(len(d) for d in dfs)
    print(f"Combined {len(dfs)} files | {total} rows -> {output_file}")


if __name__ == "__main__":
    OUTPUT = r".\4_ngrams\combined.csv"

    # Option A: explicit file list
    #combine_csv_files(
    #    inputs=[
    #        r".\1_selfmade_ai.csv",
    #        r".\1_selfmade_apple.csv",
    #    ],
    #    output_file=OUTPUT,
    #)

    # Option B: entire folder
    combine_csv_files(
        inputs=r".\4_ngrams",
        output_file=OUTPUT,
    )