import pandas as pd
import os
import sys

def split_dataset_by_value(input_csv: str, column: str, value: str, output_motherload: str, output_rest: str):
    """
    Filters a dataset by a specific column value, creating one 'motherload' file 
    with matching rows and one 'rest' file with the remaining rows.
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
        
    print(f"Filtering rows where '{column}' matches '{value}'...")
    
    # Perform case-insensitive string comparison for robust filtering
    is_match = df[column].astype(str).str.lower() == str(value).lower()
    
    # Split dataframe into motherload (matches) and rest (non-matches)
    df_motherload = df[is_match]
    df_rest = df[~is_match]
    
    # Save both dataframes enforcing the established semicolon separator
    df_motherload.to_csv(output_motherload, sep=';', index=False, encoding='utf-8')
    df_rest.to_csv(output_rest, sep=';', index=False, encoding='utf-8')
    
    print(f"Successfully split dataset:")
    print(f" -> Motherload ({len(df_motherload)} rows) -> {output_motherload}")
    print(f" -> Remaining Test Data ({len(df_rest)} rows) -> {output_rest}")

if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    INPUT_FILE = r".\4_ngrams\4_testdaten.csv"
    FILTER_COLUMN = "brand"
    FILTER_VALUE = "Grok"
    
    # Target output paths specified by you
    OUTPUT_MOTHERLOAD = r".\4_ngrams\grok.csv"
    OUTPUT_REST = r".\4_ngrams\testdata_without_grok.csv"
    
    split_dataset_by_value(INPUT_FILE, FILTER_COLUMN, FILTER_VALUE, OUTPUT_MOTHERLOAD, OUTPUT_REST)