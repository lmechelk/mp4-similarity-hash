import pandas as pd
import os
from pathlib import Path


def split_dataset(
    input_files: list,
    train_output: str,
    test_output: str,
    train_selector: str | list = "first",
) -> None:
    """Split n-gram CSVs into training references and test queries.

    Args:
        train_selector: single strategy applied to all files, or a list
                        with one entry per input file.
                        "first"        – first file per folder becomes training reference
                        "every_second" – every 2nd file per folder (index 0,2,4,...) becomes training
                        any other str  – files whose stem ends with that suffix (e.g. "_0001")
    """
    # Normalise selector to a list matching input_files length
    if isinstance(train_selector, str):
        selectors = [train_selector] * len(input_files)
    else:
        if len(train_selector) != len(input_files):
            print("Error: train_selector length must match input_files length. Aborting.")
            return
        selectors = train_selector

    train_frames, test_frames = [], []

    for f, selector in zip(input_files, selectors):
        if not os.path.isfile(f):
            print(f"Warning: file not found, skipping -> {f}")
            continue

        df = pd.read_csv(f, sep=";", engine="python")
        print(f"Loaded {len(df)} rows from {f} (selector={selector!r})")

        # Apply selector strategy per file
        if selector == "first":
            df["_folder"] = df["path"].apply(lambda p: str(Path(str(p)).parent))
            mask = ~df.duplicated(subset="_folder", keep="first")
            df = df.drop(columns="_folder")

        elif selector == "every_second":
            # Per folder: even indices (0,2,4,...) -> train, odd indices (1,3,5,...) -> test
            df["_folder"] = df["path"].apply(lambda p: str(Path(str(p)).parent))
            df["_rank"]   = df.groupby("_folder").cumcount()
            mask = df["_rank"] % 2 == 0
            df = df.drop(columns=["_folder", "_rank"])

        else:
            mask = df["path"].apply(lambda p: Path(str(p)).stem.endswith(selector))

        train_frames.append(df[mask])
        test_frames.append(df[~mask])

    if not train_frames:
        print("Error: no valid input files found. Aborting.")
        return

    df_train = pd.concat(train_frames, ignore_index=True)
    df_test  = pd.concat(test_frames,  ignore_index=True)

    # Export
    os.makedirs(os.path.dirname(train_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(test_output)  or ".", exist_ok=True)

    df_train.to_csv(train_output, sep=";", index=False)
    df_test.to_csv(test_output,  sep=";", index=False)

    print(f"Training (reference): {len(df_train)} rows -> {train_output}")
    print(f"Test (query):         {len(df_test)} rows -> {test_output}")


if __name__ == "__main__":
    TRAIN_OUTPUT = r".\5_split\5_train.csv"
    TEST_OUTPUT  = r".\5_split\5_test.csv"

    split_dataset(
        input_files=[
            r".\4_ngrams\4_vision.csv",
            r".\4_ngrams\4_ai.csv",
            r".\4_ngrams\4_eva.csv",
        ],
        train_output=TRAIN_OUTPUT,
        test_output=TEST_OUTPUT,
        train_selector=["first", "every_second", "_0001"],
    )