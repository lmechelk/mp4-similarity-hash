import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# ==========================================
# MAIN PROCESSING
# ==========================================
def plot_single_roc(input_csv: str, target_class: str, output_image: str):
    """
    Reads similarity results, binarizes the target class, calculates the ROC curve,
    and plots/saves the result as a high-res image.
    """
    if not os.path.isfile(input_csv):
        print(f"Error: Input file not found -> {input_csv}")
        return

    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv, sep=';', engine='python')

    # 1. Validate dataset
    if 'source' not in df.columns or 'similarity' not in df.columns:
        print("Error: Missing required columns ('source' or 'similarity').")
        return

    # 2. Preprocessing
    print("Preprocessing data...")
    # Fill NaN scores (corrupt/empty files) with 0.0 (no similarity)
    df['similarity'] = df['similarity'].fillna(0.0)

    # --- NEUER FILTER FÜR DIE N-ZU-N MATRIX ---
    # Wir isolieren alle Zeilen, in denen das Referenz-Video unsere Zielmarke ist.
    df_filtered = df[df['ground_truth'] == target_class]
    
    if df_filtered.empty:
        print(f"Error: No reference data found for '{target_class}' in 'ground_truth' column.")
        return
    # ------------------------------------------

    # Create binary ground truth based on the FILTERED dataset: 
    # 1 if the test video (source) is the target class, 0 otherwise
    y_true = np.where(df_filtered['source'] == target_class, 1, 0)
    y_scores = df_filtered['similarity']

    # Check if we actually have both classes (otherwise ROC calculation fails)
    if len(np.unique(y_true)) < 2:
        print(f"Error: Target class '{target_class}' either not found or is the ONLY class in the dataset.")
        return

    # 3. Calculate ROC and AUC
    print("Calculating ROC curve and AUC...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 4. Plotting
    print(f"Generating plot... (AUC = {roc_auc:.4f})")
    plt.figure(figsize=(8, 6))
    
    # Plot the actual ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve ({target_class})\nAUC = {roc_auc:.3f}')
    
    # Plot the random guessing baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Formatting the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    db_name = os.path.basename(input_csv)
    plt.title('Similarity Hash Performance\n'
              f'Ground Truth: {target_class}\n'
              f'Database: {db_name}', 
              pad=15)
              
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
  
    # Save as high-resolution PNG (dpi=300 is standard for print/thesis)
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Finished! Saved high-res ROC plot to: {output_image}")
    
    # Show the interactive plot window
    plt.show()


if __name__ == "__main__":
    # ==========================================
    # TESTDATA
    # ==========================================

    TARGET_BRAND = "Grok"

    INPUT_CSV = r".\5_similarity\5_testdaten.csv"
    
    # Where to save the plot (automatically creates folder if missing)
    OUTPUT_PLOT = r".\6_curves\6_grok_testdaten.png"

    
    plot_single_roc(INPUT_CSV, TARGET_BRAND, OUTPUT_PLOT)