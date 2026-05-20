import pandas as pd
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Define which n-gram sizes to extract (e.g., 2 for bigrams, 3 for trigrams)
NGRAM_SIZES = [2, 3, 4]

# ==========================================
# HELPER METHODS
# ==========================================
def _flatten_tree(nodes: list) -> list:
    """
    Recursively extracts the 'type' string from the nested JSON dictionary 
    and returns a flattened sequential list of atoms.
    """
    flat_list = []
    for node in nodes:
        flat_list.append(node.get("type", ""))
        # Recurse if the node has children
        if "children" in node:
            flat_list.extend(_flatten_tree(node["children"]))
    return flat_list

def _generate_ngrams(sequence: list, n: int) -> list:
    """
    Generates unique, alphabetically sorted n-grams from a flat sequence.
    Example: ['moov', 'mvhd', 'moov', 'mvhd'] (n=2) -> ['moov_mvhd']
    """
    if len(sequence) < n:
        return []
        
    # 1. Create sliding window of size n
    raw_ngrams = ["_".join(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    # 2. Remove duplicates (via set) and sort alphabetically
    return sorted(list(set(raw_ngrams)))

# ==========================================
# MAIN PROCESSING
# ==========================================
def process_ngrams(input_csv: str, output_csv: str, ngram_sizes: list):
    """
    Reads the parsed structure, flattens the JSON tree, calculates the 
    unique/sorted n-grams, and appends them as new columns to the CSV.
    """
    if not os.path.isfile(input_csv):
        print(f"Error: Input file not found -> {input_csv}")
        return
        
    print(f"Loading dataset: {input_csv}")
    
    # Auto-detect separator to prevent reading errors, but force ';' later
    df = pd.read_csv(input_csv, sep=None, engine='python')
    
    if 'structure_json' not in df.columns:
        print("Error: Critical column 'structure_json' is missing in the dataset.")
        return

    # Pre-allocate dictionary arrays for the new n-gram columns dynamically
    ngram_columns = {f"ngram_{n}": [] for n in ngram_sizes}
    total_files = len(df)
    
    for index, row in df.iterrows():
        # Print simple progress tracking to the console
        if (index + 1) % 500 == 0 or (index + 1) == total_files:
            print(f"Processing n-grams... {index + 1}/{total_files}")
            
        json_str = row.get('structure_json', '[]')
        
        # Safely parse JSON strings (fallback to empty list on error/NaN)
        try:
            tree = json.loads(json_str) if pd.notna(json_str) and json_str != "" else []
        except json.JSONDecodeError:
            tree = []
            
        # Flatten the deep structure into a sequential list
        flat_seq = _flatten_tree(tree)
        
        # Generate and store n-grams for every requested size
        for n in ngram_sizes:
            ngrams = _generate_ngrams(flat_seq, n)
            # Store as the exact string representation of a Python list (e.g., "['a', 'b']")
            ngram_columns[f"ngram_{n}"].append(str(ngrams))
            
    # Append the dynamically created n-gram columns to the original DataFrame
    for col_name, col_data in ngram_columns.items():
        df[col_name] = col_data
        
    # Export keeping all old columns + new n-gram columns, enforcing semicolon
    df.to_csv(output_csv, sep=';', index=False)
    print(f"Finished! Exported dataset with unique/sorted n-grams to: {output_csv}")

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    
    # Define targets for quick execution
    INPUT_CSV = "3_selfmade_ai.csv" 
    OUTPUT_CSV = "4_selfmade_ai.csv"
    
    process_ngrams(INPUT_CSV, OUTPUT_CSV, NGRAM_SIZES)