import pandas as pd
import os


# 4CC codes are always 4 characters, so chunk size scales with n
NGRAM_CHUNK_SIZES = {f"ngram_{n}": n * 4 for n in range(2, 10)}


def _split_ngrams(s: str, chunk_size: int) -> set:
    """Split a concatenated n-gram string into a set of chunks."""
    s = str(s).strip()
    return {s[i:i + chunk_size] for i in range(0, len(s), chunk_size) if len(s[i:i + chunk_size]) == chunk_size}


def _aggregate_ngrams(series: pd.Series, chunk_size: int) -> str:
    """Union all n-gram sets across rows, deduplicate, sort, concatenate."""
    combined = set()
    for cell in series.dropna():
        combined.update(_split_ngrams(cell, chunk_size))
    return "".join(sorted(combined))


def create_source_digest(
    input_file: str,
    output_file: str,
    filters: dict,
    label: str = None,
) -> None:
    """Aggregate n-grams of all rows matching the filter into one source digest.

    Args:
        input_file:  Path to 5_train.csv.
        output_file: Path to output CSV (appended if exists, created otherwise).
        filters:     Column/value pairs to filter rows, e.g. {"brand": "Apple"}.
                     Both column names and values are matched case-insensitively.
        label:       Name for this digest. Auto-generated from filters if omitted.
    """
    df = pd.read_csv(input_file, sep=";", engine="python")

    # Normalise column names to lowercase for case-insensitive key matching
    df.columns = [c.lower() for c in df.columns]

    # Apply each filter case-insensitively
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        col_lower = col.lower()
        if col_lower not in df.columns:
            print(f"Warning: column {col!r} not found, skipping filter.")
            continue
        if isinstance(val, list):
            val_lower = [str(v).lower() for v in val]
            mask &= df[col_lower].astype(str).str.lower().isin(val_lower)
        else:
            mask &= df[col_lower].astype(str).str.lower() == str(val).lower()

    df_filtered = df[mask]

    if df_filtered.empty:
        print(f"Warning: no rows matched filters {filters}. Nothing written.")
        return

    # Aggregate each n-gram column using its correct chunk size
    ngram_cols = [c for c in df_filtered.columns if c.startswith("ngram_")]
    aggregated = {
        col: _aggregate_ngrams(df_filtered[col], NGRAM_CHUNK_SIZES[col])
        for col in ngram_cols
        if col in NGRAM_CHUNK_SIZES
    }

    # Auto-generate label from filter values if not provided
    if label is None:
        label = "_".join(str(v) for v in filters.values())

    row = {
        "source_label": label,
        "filter": str(filters),
        "file_count": len(df_filtered),
        **aggregated,
    }

    out_df = pd.DataFrame([row])

    # Append to existing file or create new one
    file_exists = os.path.isfile(output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_df.to_csv(
        output_file,
        sep=";",
        index=False,
        mode="a" if file_exists else "w",
        header=not file_exists,
    )

    print(f"Source digest {label!r}: {len(df_filtered)} files aggregated -> {output_file}")


if __name__ == "__main__":
    INPUT  = r".\5_testtrainsplit\5_train.csv"
    OUTPUT = r".\6_sourcedigest\6_source_digests.csv"

    # Examples — extend as needed
    create_source_digest(INPUT, OUTPUT, filters={"brand": "Apple"})
    create_source_digest(INPUT, OUTPUT, filters={"brand": "Apple", "processing": "ORIGINAL"})
    create_source_digest(INPUT, OUTPUT, filters={"processing": "youtube"})

