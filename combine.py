import pandas as pd
import sys
import os

def combine_csv_files(file1, file2, output_file):
    """Combines two CSV files by merging their rows and aligning headers."""
    
    # Check if files exist
    if not os.path.isfile(file1) or not os.path.isfile(file2):
        print(f"Error: One or both input files not found.")
        sys.exit(1)

    print(f"Loading {file1} and {file2}...")
    
    # Read files using semicolon separator
    df1 = pd.read_csv(file1, sep=';')
    df2 = pd.read_csv(file2, sep=';')

    # Concatenate dataframes: rows are appended, headers are unified
    combined_df = pd.concat([df1, df2], ignore_index=True, sort=False)

    # Export combined data to new CSV
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    
    print(f"Successfully combined {len(df1)} + {len(df2)} = {len(combined_df)} rows.")
    print(f"Saved result to: {output_file}")

if __name__ == "__main__":
    ########
    # HERE #
    ########
    
    # Files to combine
    FILE_1 = "1_selfmade_ai.csv"
    FILE_2 = "1_selfmade_apple.csv"
    OUTPUT = "1_selfmade.csv"

    combine_csv_files(FILE_1, FILE_2, OUTPUT)