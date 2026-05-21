import pandas as pd
import numpy as np
from pathlib import Path
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Parameters for the Tversky Index
# 0.0: We do NOT penalize if the test video lacks atoms from the reference video
PENALTY_MISSING_IN_TEST = 0.0  
# 1.0: We DO penalize strictly if the test video contains foreign/new atoms
PENALTY_EXTRA_IN_TEST = 1.0    

# ==========================================
# HELPER METHODS
# ==========================================
def _split_digest(digest: str, ngram_size: int) -> set:
    """
    Splits the continuous Similarity Digest string back into a set of n-grams.
    """
    if pd.isna(digest) or not digest:
        return set()
        
    chunk_size = ngram_size * 4
    return {digest[i:i + chunk_size] for i in range(0, len(digest), chunk_size)}

def _calculate_similarity(reference_set: set, test_set: set, mode: str) -> float:
    """
    Calculates the similarity between a single Reference file and a single Test file.
    Returns np.nan if a ZeroDivisionError occurs.
    """
    intersection = len(reference_set & test_set)
    
    try:
        if mode.lower() == 'jaccard':
            union = len(reference_set | test_set)
            return intersection / union
            
        elif mode.lower() == 'tversky':
            missing_in_test = len(reference_set - test_set)
            extra_in_test = len(test_set - reference_set)
            
            denominator = (intersection + 
                           (PENALTY_MISSING_IN_TEST * missing_in_test) + 
                           (PENALTY_EXTRA_IN_TEST * extra_in_test))
                           
            return intersection / denominator
        
        else:
            raise ValueError(f"Unknown similarity mode: {mode}")
            
    except ZeroDivisionError:
        return np.nan

# ==========================================
# MAIN PROCESSING
# ==========================================
def compare_datasets(input_file: str, output_file: str, ngram_size: int, mode: str):
    """
    Performs an N-to-N comparison between EVERY file within a SINGLE dataset.
    Exports strictly 3 columns: ground_truth (brand), source (brand), similarity.
    """
    if not os.path.isfile(input_file):
        print(f"Error: Input file not found -> {input_file}")
        return

    print(f"Loading Dataset: {input_file}")
    df = pd.read_csv(input_file, sep=';', engine='python')
    
    target_col = f"ngram_{ngram_size}"
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' missing in the file.")
        return

    # 1. Pre-parse the dataset into memory for performance
    print("Pre-parsing files into memory...")
    parsed_data = []
    for index, row in df.iterrows():
        digest = str(row.get(target_col, ''))
        
        # Keep filename for console warnings, but extract brand for the CSV output
        fname = Path(str(row.get('path', f'row_{index}'))).stem
        brand = row.get('brand', 'UNKNOWN')
        
        parsed_data.append({
            "set": _split_digest(digest, ngram_size),
            "file": fname,
            "brand": brand
        })

    # 2. Perform N-to-N Comparison (Every item against every item)
    total_comparisons = len(parsed_data) * len(parsed_data)
    print(f"Starting {total_comparisons} individual comparisons...")
    
    results = []
    
    # Nested loop using the same parsed list twice
    for t_data in parsed_data:      # t_data acts as the Test video (source)
        for r_data in parsed_data:  # r_data acts as the Reference video (ground_truth)
            
            score = _calculate_similarity(r_data["set"], t_data["set"], mode)
            
            # Warn only on complete structural mismatches (using filename for clarity)
            if score == 0.0:
                print(f"  -> WARNING: Score 0.0 | Ref: {r_data['file']} vs Test: {t_data['file']}")
            
            # Strictly use the brand for the CSV output
            results.append({
                "ground_truth": r_data["brand"],
                "source": t_data["brand"],
                "similarity": round(score, 4) if pd.notna(score) else np.nan
            })
            
    # 3. Generate and Export Output Dataframe
    df_results = pd.DataFrame(results)
    
    # Create the requested output folder if it does not exist yet
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Export using the established semicolon separator
    df_results.to_csv(output_file, sep=';', index=False)
    
    print(f"Finished! Exported {len(df_results)} comparison results to: {output_file}")


if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    INPUT_FILE = r".\4_ngrams\4_original.csv"     
    OUTPUT_FILE = r".\5_similarity\5_original.csv" 
    
    NGRAM_SIZE = 2                              
    SIMILARITY_MODE = "tversky"                 # "tversky" or "jaccard"
    
    compare_datasets(INPUT_FILE, OUTPUT_FILE, NGRAM_SIZE, SIMILARITY_MODE)