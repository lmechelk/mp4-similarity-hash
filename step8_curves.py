import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os

def plot_roc(
    input_csvs:  list | str,
    ngram_size:  int,
    title:       str,
    output_dir:  str  = r".\8_curves",
    output_name: str  = None,
    show_mean:   bool = False,
    export_auc:  bool = False,
) -> None:
    """Plot ROC curves for multiple CSV files in a single plot.

    Args:
        input_csvs:  List of CSV paths, or a single folder path (all *.csv in that folder).
                     Each entry can also be a dict {"csv": path, "label": "My Label"}
                     or a tuple (path, "My Label") to set a custom legend label.
                     Without a custom label, ground_truth_source from the CSV is used.
        ngram_size:  N-gram size to evaluate (e.g. 2, 3, 4).
        title:       Plot title.
        output_dir:  Directory for the output image.
        output_name: Optional filename. Auto-generated if omitted.
        show_mean:   If True, adds a macro-average ROC curve across all input files.
        export_auc:  If True, saves AUC values as a .txt file in output_dir.
    """
    # Resolve inputs to list of (path, custom_label_or_None)
    if isinstance(input_csvs, str) and os.path.isdir(input_csvs):
        entries = [(p, None) for p in sorted(Path(input_csvs).glob("*.csv"))]
    else:
        entries = []
        for item in input_csvs:
            if isinstance(item, dict):
                entries.append((Path(item["csv"]), item.get("label")))
            elif isinstance(item, tuple):
                entries.append((Path(item[0]), item[1] if len(item) > 1 else None))
            else:
                entries.append((Path(item), None))

    sim_col  = f"similarity_{ngram_size}"
    mean_fpr = np.linspace(0, 1, 200)
    tprs         = []
    auc_results  = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for csv_path, custom_label in entries:
        if not csv_path.is_file():
            print(f"Warning: file not found, skipping -> {csv_path}")
            continue

        df = pd.read_csv(csv_path, sep=";", engine="python")

        if sim_col not in df.columns:
            print(f"Warning: column {sim_col!r} not found in {csv_path}, skipping.")
            continue

        if "true_positive" not in df.columns:
            print(f"Warning: column 'true_positive' not found in {csv_path}, skipping.")
            continue

        y_true   = df["true_positive"].astype(int).values
        y_scores = df[sim_col].fillna(0).values

        # Custom label overrides ground_truth_source
        label = custom_label if custom_label is not None else df["ground_truth_source"].iloc[0]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"{label} AUC = {roc_auc:.3f}")
        auc_results.append((label, roc_auc))

        if show_mean:
            tprs.append(np.interp(mean_fpr, fpr, tpr))

    # Macro-average curve
    if show_mean and tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, color="black", lw=2.5, linestyle="-.",
                label=f"Mean AUC = {mean_auc:.3f}")
        auc_results.append(("Mean", mean_auc))

    # Random baseline
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(title, pad=12)
    ax.grid(alpha=0.3)

    # Legend outside plot on the right
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Resolve output path
    if output_name is None:
        base = entries[0][0].stem if entries else "output"
        output_name = f"{base}_{ngram_size}gram_roc.png"

    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, output_name)

    # Export AUC values as txt
    if export_auc and auc_results:
        auc_file = os.path.join(output_dir, os.path.splitext(output_name)[0] + ".txt")
        with open(auc_file, "w", encoding="utf-8") as f:
            f.write(f"AUC - {title}\n")
            f.write("-" * 40 + "\n")
            for lbl, val in auc_results:
                f.write(f"{lbl:<30} {val:.3f}\n")

    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {output_image}")


if __name__ == "__main__":
    # Option A: plain file list (label from CSV)
    plot_roc(
        input_csvs=[
            r".\7_similarity\brands\Apple_ORIGINAL_vs_processing_ORIGINAL_content_source_DIGITAL_CAMERA.csv",
            r".\7_similarity\brands\Asus_ORIGINAL_vs_processing_ORIGINAL_content_source_DIGITAL_CAMERA.csv",
        ],
        ngram_size=2,
        title="ROC – Brands (ORIGINAL)",
        output_dir=r".\8_curves",
        output_name="brands_roc.png",
        show_mean=True,
        export_auc=True,
    )

    # Option B: custom labels via dict
    plot_roc(
        input_csvs=[
            {"csv": r".\7_similarity\brands\Apple_ORIGINAL_vs_....csv", "label": "Apple (nativ)"},
            {"csv": r".\7_similarity\brands\Asus_ORIGINAL_vs_....csv",  "label": "Asus (nativ)"},
        ],
        ngram_size=2,
        title="ROC – Brands (ORIGINAL)",
        output_dir=r".\8_curves",
        output_name="brands_custom_roc.png",
    )

    # Option C: custom labels via tuple
    plot_roc(
        input_csvs=[
            (r".\7_similarity\brands\Apple_ORIGINAL_vs_....csv", "Apple (nativ)"),
            (r".\7_similarity\brands\Asus_ORIGINAL_vs_....csv",  "Asus (nativ)"),
        ],
        ngram_size=2,
        title="ROC – Brands (ORIGINAL)",
        output_dir=r".\8_curves",
        output_name="brands_tuple_roc.png",
    )

    # Option D: entire folder (no custom labels)
    # plot_roc(
    #     input_csvs=r".\7_similarity\brands",
    #     ngram_size=2,
    #     title="ROC – Brands (ORIGINAL)",
    #     output_dir=r".\8_curves",
    #     output_name="brands_roc.png",
    #     show_mean=True,
    #     export_auc=True,
    # )