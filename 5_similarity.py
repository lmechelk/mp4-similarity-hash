import pandas as pd
import numpy as np
from pathlib import Path
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Parameters for the Tversky Index
# 0.0: We do NOT penalize if the test video lacks atoms from the huge Motherload
PENALTY_MISSING_IN_TEST = 0.0  
# 1.0: We DO penalize strictly if the test video contains foreign/new atoms
PENALTY_EXTRA_IN_TEST = 1.0    

# ==========================================
# HELPER METHODS
# ==========================================
def _split_digest(digest: str, ngram_size: int) -> set:
    """
    Splits the continuous Similarity Digest string back into a set of n-grams
    Since MP4 atoms are strictly 4 characters, chunk size is ngram_size * 4

    input: digest = "moovmvhdmvhdtrak", ngram_size = 2
    output: set of n-grams -> {"moovmvhd", "mvhdmvhd", "mvhdtrak"}
    """
    if pd.isna(digest) or not digest:
        return set()
        
    chunk_size = ngram_size * 4
    return {digest[i:i + chunk_size] for i in range(0, len(digest), chunk_size)}

def _calculate_similarity(motherload: set, test_set: set, mode: str) -> float:
    """
    Calculates the similarity between the Motherload and a single Test Set.
    Returns np.nan if a ZeroDivisionError occurs

    input: motherload = set of n-grams from Ground Truth, 
        test_set = set of n-grams from Test video, 
        mode = "jaccard" or "tversky"
    output: similarity score (float)
    """
    intersection = len(motherload & test_set)
    
    try:
        if mode.lower() == 'jaccard':
            union = len(motherload | test_set)
            return intersection / union
            
        elif mode.lower() == 'tversky':
            # missing_in_test: Atoms present in Motherload but missing in Test video
            missing_in_test = len(motherload - test_set)
            # extra_in_test: Foreign atoms present in Test video but NOT in Motherload
            extra_in_test = len(test_set - motherload)
            
            denominator = (intersection + 
                           (PENALTY_MISSING_IN_TEST * missing_in_test) + 
                           (PENALTY_EXTRA_IN_TEST * extra_in_test))
                           
            return intersection / denominator
        
        else:
            raise ValueError(f"Unknown similarity mode: {mode}")
            
    except ZeroDivisionError:
        # Gracefully handle empty datasets/corrupted files by returning NaN
        print("Warning: ZeroDivisionError encountered during similarity calculation. Returning NaN.")
        return np.nan

# ==========================================
# MAIN PROCESSING
# ==========================================
def compare_datasets(gt_file: str, test_file: str, ngram_size: int, mode: str):
    """
    Reads the Ground Truth file to build a global Motherload set, then 
    evaluates each row of the Test file against it.
    """
    if not os.path.isfile(gt_file) or not os.path.isfile(test_file):
        print("Error: One or both input files not found.")
        return

    print(f"Loading Ground Truth: {gt_file}")
    df_gt = pd.read_csv(gt_file, sep=';', engine='python')
    print(f"Loading Test Source: {test_file}")
    df_test = pd.read_csv(test_file, sep=';', engine='python')
    
    target_col = f"ngram_{ngram_size}"
    
    # Validate datasets
    if target_col not in df_gt.columns or target_col not in df_test.columns:
        print(f"Error: Target column '{target_col}' missing in one of the files.")
        return

    # Build the Motherload (Union of all n-gram sets in Ground Truth)
    print("Building Motherload signature from Ground Truth...")
    motherload_set = set()
    for digest in df_gt[target_col]:
        motherload_set.update(_split_digest(str(digest), ngram_size))
        
    print(f"Motherload built: {len(motherload_set)} unique {ngram_size}-grams.")

    # Compare each row of the Test file against the Motherload
    print(f"Calculating {mode.upper()} similarity for {len(df_test)} test files...")
    
    gt_name = Path(gt_file).stem    # Extract filename without extension
    results = []
    
    for index, row in df_test.iterrows():
        test_digest = str(row.get(target_col, ''))
        test_set = _split_digest(test_digest, ngram_size)
        
        # Calculate score
        score = _calculate_similarity(motherload_set, test_set, mode)

        # Console logging for extreme values
        file_ref = row.get('path', f"Row {index}")
        if score == 0.0:
            print(f"  -> WARNING: Score 0.0 (No overlap) | File: {file_ref}")
        elif score == 1.0:
            print(f"  -> INFO: Score 1.0 (Perfect match) | File: {file_ref}")
        
        # Append to results
        results.append({
            "ground_truth": gt_name,
            "source": row.get('brand', 'UNKNOWN'),
            "similarity": round(score, 4) if pd.notna(score) else np.nan
        })
        
    # Generate Output Dataframe
    df_results = pd.DataFrame(results)
    
    # Export using the manually provided path and the established semicolon separator
    df_results.to_csv(OUTPUT_FILE, sep=';', index=False)
    print(f"Finished! Exported results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================
    
    FILE_GROUND_TRUTH = r".\4_ngrams\grok.csv"     
    FILE_TEST_SOURCE = r".\4_ngrams\testdata_without_grok.csv"
    OUTPUT_FILE = r".\5_similarity\5_grok_testdata.csv"
    
    
    NGRAM_SIZE = 3                              
    SIMILARITY_MODE = "tversky" # "tversky" or "jaccard"
    
    compare_datasets(FILE_GROUND_TRUTH, FILE_TEST_SOURCE, NGRAM_SIZE, SIMILARITY_MODE)