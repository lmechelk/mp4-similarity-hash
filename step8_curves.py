import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


def plot_roc(input_csv: str, output_dir: str = r".\8_curves") -> None:
    """Plot ROC curves for all similarity columns found in the input CSV.

    One curve per similarity_N column, all in a single plot.
    Output filename is auto-generated from the input CSV name.
    """
    df = pd.read_csv(input_csv, sep=";", engine="python")

    # Derive y_true from true_positive column
    y_true = df["true_positive"].astype(int).values

    # Detect all similarity columns
    sim_cols = sorted([c for c in df.columns if c.startswith("similarity_")])

    if not sim_cols:
        print("Error: no similarity_* columns found in input.")
        return

    source_label = df["ground_truth_source"].iloc[0]

    plt.figure(figsize=(8, 6))

    for col in sim_cols:
        y_scores = df[col].fillna(0).values
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        n = col.split("_")[1]
        plt.plot(fpr, tpr, lw=2, label=f"{n}-Gram  AUC = {roc_auc:.3f}")

    # Random baseline
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(
        f"Similarity Hash Performance\n"
        f"Ground Truth: {source_label}\n"
        f"Database: {os.path.basename(input_csv)}",
        pad=15,
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Auto-generate output path from input filename
    base = os.path.splitext(os.path.basename(input_csv))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, f"{base}_roc.png")

    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {output_image}")


if __name__ == "__main__":
    plot_roc(
        input_csv=r".\7_similarity\Samsung_ORIGINAL_vs_processing_ORIGINAL_content_source_DIGITAL_CAMERA.csv",
        output_dir=r".\8_curves",
    )
