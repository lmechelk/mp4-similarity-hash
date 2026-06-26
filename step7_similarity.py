import pandas as pd
import numpy as np
import ast
import os

# Tversky penalty weights
PENALTY_MISSING_IN_TEST = 0.0  # no penalty for boxes missing in test
PENALTY_EXTRA_IN_TEST   = 1.0  # strict penalty for extra boxes in test


def _split_digest(digest: str, ngram_size: int) -> set:
    """Split a concatenated digest string back into a set of n-grams."""
    if pd.isna(digest) or not digest:
        return set()
    chunk_size = ngram_size * 4
    return {digest[i:i + chunk_size] for i in range(0, len(digest), chunk_size)}


def _calculate_similarity(reference_set: set, test_set: set, mode: str) -> float:
    """Calculate similarity between a reference and a test n-gram set.

    Returns np.nan on ZeroDivisionError.
    """
    intersection = len(reference_set & test_set)
    try:
        if mode.lower() == "jaccard":
            union = len(reference_set | test_set)
            return intersection / union

        elif mode.lower() == "tversky":
            missing_in_test = len(reference_set - test_set)
            extra_in_test   = len(test_set - reference_set)
            denominator = (
                intersection
                + (PENALTY_MISSING_IN_TEST * missing_in_test)
                + (PENALTY_EXTRA_IN_TEST   * extra_in_test)
            )  # BUG FIX: fehlende schließende Klammer ergänzt
            return intersection / denominator

        else:
            raise ValueError(f"Unknown similarity mode: {mode}")

    except ZeroDivisionError:
        return np.nan


def _make_output_path(output_dir: str, source_label: str, test_filters: dict) -> str:
    """Auto-generate output filename: {source_label}_vs_{filter_string}.csv"""
    filter_str = "_".join(f"{k}_{v}" for k, v in test_filters.items()) if test_filters else "all"
    return os.path.join(output_dir, f"{source_label}_vs_{filter_str}.csv")


def _parse_filter_keys(filter_str: str) -> list:
    """Extract column keys from a stored filter string, e.g. "{'brand': 'Apple'}" -> ['brand']."""
    try:
        return list(ast.literal_eval(filter_str).keys())
    except Exception:
        return []


def compute_similarity(
    source_file: str,
    source_label: str,
    test_file: str,
    output_dir: str = r".\7_similarity",
    output_file: str = None,
    test_filters: dict = None,
    mode: str = "tversky",
) -> None:
    """Compare one source digest against all matching test rows.

    Args:
        source_file:   Path to 6_source_digests.csv.
        source_label:  Label of the source digest row to use.
        test_file:     Path to 5_test.csv.
        output_dir:    Directory for auto-generated output (used if output_file is None).
        output_file:   Explicit output path. Auto-generated if omitted.
        test_filters:  Optional dict to filter test rows case-insensitively.
        mode:          Similarity metric: "tversky" (default) or "jaccard".
    """
    # Load source digest row by label
    df_source = pd.read_csv(source_file, sep=";", engine="python")
    df_source.columns = [c.lower() for c in df_source.columns]
    source_row = df_source[df_source["source_label"].str.lower() == source_label.lower()]

    if source_row.empty:
        print(f"Error: source_label {source_label!r} not found in {source_file}.")
        return

    source_row = source_row.iloc[0]

    # Extract filter keys from stored filter string to build ground_truth_mp4
    filter_keys = _parse_filter_keys(source_row.get("filter", "{}"))

    # Load and optionally filter test data
    df_test = pd.read_csv(test_file, sep=";", engine="python")
    df_test.columns = [c.lower() for c in df_test.columns]

    if test_filters:
        for col, val in test_filters.items():
            col_lower = col.lower()
            if col_lower not in df_test.columns:
                print(f"Warning: column {col!r} not found, skipping filter.")
                continue
            if isinstance(val, list):
                # OR-Filter
                val_lower = [str(v).lower() for v in val]
                df_test = df_test[df_test[col_lower].astype(str).str.lower().isin(val_lower)]
            else:
                df_test = df_test[df_test[col_lower].astype(str).str.lower() == str(val).lower()]

    if df_test.empty:
        print(f"Warning: no test rows matched filters {test_filters}. Nothing written.")
        return

    # Build ground_truth_mp4 per row from the same filter keys used in source digest
    def make_mp4_label(row: pd.Series) -> str:
        parts = [str(row[k]).lower() for k in filter_keys if k in row.index]
        return "_".join(parts) if parts else "unknown"

    df_test = df_test.copy()
    df_test["ground_truth_mp4"] = df_test.apply(make_mp4_label, axis=1)

    # Detect ngram columns and pre-parse source sets
    ngram_cols = [
        c for c in df_test.columns
        if c.startswith("ngram_") and c in source_row.index
    ]

    source_sets = {
        col: _split_digest(source_row[col], int(col.split("_")[1]))
        for col in ngram_cols
    }

    # Build result dataframe
    results = pd.DataFrame({
        "ground_truth_source":  [source_label] * len(df_test),
        "ground_truth_mp4":     df_test["ground_truth_mp4"].values,
        "filename": df_test["path"].apply(os.path.basename).values if "path" in df_test.columns else [""] * len(df_test),
    })

    results["true_positive"] = results["ground_truth_mp4"] == source_label.lower()

    for col in ngram_cols:
        ngram_size = int(col.split("_")[1])
        sim_col    = col.replace("ngram_", "similarity_")
        ref_set    = source_sets[col]

        scores = []
        for cell in df_test[col]:
            test_set = _split_digest(cell, ngram_size)
            score    = _calculate_similarity(ref_set, test_set, mode)
            if score == 0.0:
                print(f"  -> WARNING: Score 0.0 | source={source_label} vs cell={str(cell)[:20]}...")
            scores.append(round(score, 6) if pd.notna(score) else np.nan)

        results[sim_col] = scores

    if output_file is None:
        output_file = _make_output_path(output_dir, source_label, test_filters or {})

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, sep=";", index=False)

    print(f"Finished! {len(results)} rows -> {output_file}")


if __name__ == "__main__":
    input7 = [
        {
            "source_label": "youtube",
            "test_filters": {"processing": "youtube"},
        },
    ] 

    SOURCE_FILE = r".\6_sourcedigest\6_source_digests.csv"
    TEST_FILE   = r".\5_testtrainsplit\5_test.csv"

    for i in input7:
        compute_similarity(
            source_file=SOURCE_FILE,
            source_label=i["source_label"],
            test_file=TEST_FILE,
            test_filters=i["test_filters"],
        )