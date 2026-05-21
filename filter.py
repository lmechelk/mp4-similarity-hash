import pandas as pd
import os
import sys

# ==========================================
# MAIN PROCESSING
# ==========================================
def filter_dataset(input_csv: str, column: str, value: str, output_csv: str, mode: str):
    """
    Filters a dataset by a specific column value and saves the result to a single output file.
    Mode 'include': Keeps ONLY rows matching the value.
    Mode 'exclude': Keeps ALL rows EXCEPT those matching the value.
    """
    if not os.path.isfile(input_csv):
        print(f"Error: Input file not found -> {input_csv}")
        sys.exit(1)
        
    print(f"Loading dataset: {input_csv}")
    # Read file using auto-detect for separator safety
    df = pd.read_csv(input_csv, sep=None, engine='python')
    
    # Check if the requested filter column exists
    if column not in df.columns:
        print(f"Error: Column '{column}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)
        
    print(f"Applying filter (Mode: {mode.upper()}) where '{column}' matches '{value}'...")
    
    # Perform case-insensitive string comparison for robust filtering
    is_match = df[column].astype(str).str.lower() == str(value).lower()
    
    # Apply the include/exclude logic
    if mode.lower() == "include":
        df_filtered = df[is_match]
    elif mode.lower() == "exclude":
        df_filtered = df[~is_match]
    else:
        print(f"Error: Unknown mode '{mode}'. Please use 'include' or 'exclude'.")
        sys.exit(1)
    
    # Create the requested output folder if it does not exist yet
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    
    # Save the filtered dataframe enforcing the established semicolon separator
    df_filtered.to_csv(output_csv, sep=';', index=False, encoding='utf-8')
    
    print(f"Successfully processed dataset:")
    print(f" -> Result: {len(df_filtered)} rows exported to -> {output_csv}")


if __name__ == "__main__":
    
    # ==========================================
    # TESTDATA
    # ==========================================
    
    INPUT_FILE = r".\4_ngrams\4_testdaten.csv"
    FILTER_COLUMN = "processing"
    FILTER_VALUE = "original"
    
    # Logic switch: "include" or "exclude"
    MODE = "include"
    
    OUTPUT_FILE = r".\4_ngrams\original.csv"
    
    filter_dataset(INPUT_FILE, FILTER_COLUMN, FILTER_VALUE, OUTPUT_FILE, MODE)