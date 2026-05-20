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
    and returns a flattened sequential list of atoms

    input: json_tree
    output: flat_list = ['moov', 'mvhd', 'mvhd', 'trak']
    """
    flat_list = []
    for node in nodes:
        flat_list.append(node.get("type", ""))
        # Recurse if the node has children
        if "children" in node:
            flat_list.extend(_flatten_tree(node["children"]))
    return flat_list

def _generate_similarity_digest(sequence: list, n: int) -> str:
    """
    Generates unique, alphabetically sorted n-grams from a flat sequence
    and concatenates them into a single continuous string (Similarity Digest)
    
    input: sequence = ['moov', 'mvhd', 'mvhd', 'trak'], n = 2
    output: similarity_digest with n-grams -> "moovmvhdmvhdtrak"
    """
    if len(sequence) < n:
        return ""
        
    # Create sliding window of size n to generate n-grams
    raw_ngrams = ["".join(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    # Remove duplicates via set, sort alphabetically, and join into one string
    return "".join(sorted(list(set(raw_ngrams))))

# ==========================================
# MAIN PROCESSING
# ==========================================
def process_ngrams(input_csv: str, output_csv: str, ngram_sizes: list):
    """
    Reads the parsed structure, flattens the JSON tree, calculates the 
    Similarity Digests, and appends them as new columns to the CSV

    input: csv with 'structure_json' column, list of n-gram sizes
    output: csv with new columns for each n-gram size containing the Similarity Digest
    """

    # Basic validation to ensure the input file exists before processing
    if not os.path.isfile(input_csv):
        print(f"Error: Input file not found -> {input_csv}")
        return
    print(f"Loading dataset: {input_csv}")
    
    # Auto-detect separator to prevent reading errors, but force ';' later
    df = pd.read_csv(input_csv, sep=None, engine='python')
    
    # Validate that the critical 'structure_json' column exists before processing
    if 'structure_json' not in df.columns:
        print("Error: Critical column 'structure_json' is missing in the dataset.")
        return

    # Initialize empty lists for each n-gram size to store the generated digests
    ngram_columns = {f"ngram_{n}": [] for n in ngram_sizes}
    total_files = len(df)
    
    # Iterate through each row, process the JSON structure, and generate Similarity Digests
    for index, row in df.iterrows():
        # Simple console progress tracking
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
        
        # Generate and store the Similarity Digest for every requested size
        for n in ngram_sizes:
            digest = _generate_similarity_digest(flat_seq, n)
            ngram_columns[f"ngram_{n}"].append(digest)
            
    # Append the dynamically created digest columns to the DataFrame
    for col_name, col_data in ngram_columns.items():
        df[col_name] = col_data
        
    # Export keeping all old columns + new digest columns, enforcing semicolon
    df.to_csv(output_csv, sep=';', index=False)
    print(f"Finished! Exported dataset with Similarity Digests to: {output_csv}")

if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================
        
    INPUT_CSV = r".\3_parse\3_testdaten.csv"
    OUTPUT_CSV = r".\4_ngrams\4_testdaten.csv"
    
    process_ngrams(INPUT_CSV, OUTPUT_CSV, NGRAM_SIZES)
