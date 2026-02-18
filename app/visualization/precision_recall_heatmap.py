import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from app.dataset.utils import find_qg_results_file, get_qg_results
from app.config.config import (
    COLORS,
    COLORMAPS,
    CURRENT_BEST_RUN_FOLDER,
    FIGURE_CONFIG,
    apply_matplotlib_style,
)

# Apply consistent styling for all figures
apply_matplotlib_style()


def plot_precision_recall_heatmap(
    data, out_path="precision_recall_heatmap.png", bins=20, min_positive_threshold=50
):
    """
    Plots a smooth heatmap showing frequency of precision-recall combinations.

    Parameters:
    - data: A DataFrame containing 'pubmed_precision' and 'pubmed_recall' columns.
    - bins: Number of bins for both axes (default 20).
    """
    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()

    # "wspace": 0.05 all triples fromd ata that do not satisify min_positive_threshold
    data = data[data["num_positive"] >= min_positive_threshold]

    # Determine range from data with small padding
    precision_min, precision_max = data["pubmed_precision"].min(), data["pubmed_precision"].max()
    recall_min, recall_max = data["pubmed_recall"].min(), data["pubmed_recall"].max()
    print(f"Precision range: {precision_min:.4f} - {precision_max:.4f}")
    print(f"Recall range: {recall_min:.4f} - {recall_max:.4f}")
    # Add small padding to avoid edge cases
    padding = 0.0
    precision_range = [max(0, precision_min - padding), min(1, precision_max + padding)]
    recall_range = [max(0, recall_min - padding), min(1, recall_max + padding)]

    # Create 2D histogram for smooth heatmap
    heatmap, xedges, yedges = np.histogram2d(
        data["pubmed_precision"],
        data["pubmed_recall"],
        bins=bins,
        range=[precision_range, recall_range],
    )

    # Plot as smooth heatmap
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[
            precision_range[0],
            precision_range[1],
            recall_range[0],
            recall_range[1],
        ],
        aspect="auto",
        cmap=COLORMAPS["heatmap"],
        interpolation="bilinear",
    )

    plt.colorbar(im, ax=ax, label="Frequency")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Frequency Heatmap")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


def plot_precision_recall_scatter(
    data,
    out_path="precision_recall_scatter.png",
    alpha=0.5,
    size=20,
    min_positive_threshold=50,
):
    """
    Plots a scatter plot of precision-recall pairs.

    Parameters:
    - data: A DataFrame containing 'pubmed_precision' and 'pubmed_recall' columns.
    - alpha: Transparency of dots (default 0.5).
    - size: Size of dots (default 20).
    """
    # "wspace": 0.05 all triples fromd ata that do not satisify min_positive_threshold
    data = data[data["num_positive"] >= min_positive_threshold]

    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()

    ax.scatter(
        data["pubmed_precision"], data["pubmed_recall"], alpha=alpha, s=size, c=COLORS["primary"]
    )

    # Determine range from data with small padding
    precision_min, precision_max = data["pubmed_precision"].min(), data["pubmed_precision"].max()
    recall_min, recall_max = data["pubmed_recall"].min(), data["pubmed_recall"].max()

    padding = 0.0001
    ax.set_xlim(max(0, precision_min - padding), min(1, precision_max + padding))
    ax.set_ylim(max(0, recall_min - padding), min(1, recall_max + padding))

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Scatter Plot")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {out_path}")


def get_precision_recall_pairs_from_jsonl(jsonl_path):
    precision_recall_pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            precision = data.get("pubmed_precision")
            recall = data.get("pubmed_recall")
            num_positive = data.get("num_positive", 0)
            if precision is not None and recall is not None:
                precision_recall_pairs.append((precision, recall, num_positive))
    return precision_recall_pairs


def plot_precision_recall_histograms(
    data, out_path="precision_recall_histograms.png", bins=100
):
    """
    Plots two histograms (precision and recall) with stacked bars split by top_k_type.

    Parameters:
    - data: A DataFrame containing 'pubmed_precision', 'pubmed_recall', and 'top_k_type' columns.
    - bins: Number of bins for histograms (default 20).
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_CONFIG["full_width"], FIGURE_CONFIG["full_width"] * 0.45)
    )
    plt.subplots_adjust(wspace=0.05)

    # Split data by top_k_type
    top_k_types = data["top_k_type"].unique()
    split_data = [data[data["top_k_type"] == t] for t in sorted(top_k_types)]
    
    # Color mapping for top_k_type
    color_map = {
        "cosine": COLORS["cosine_k"],
        "fixed": COLORS["fixed_k"],
        "pos_count": COLORS["pos_count_k"],
    }
    colors_split = [color_map.get(t, COLORS["primary"]) for t in sorted(top_k_types)]
    labels_split = [f"top_k_type: {t}" for t in sorted(top_k_types)]

    # Count total values for debugging
    precision_count = sum(len(d["pubmed_precision"].dropna()) for d in split_data)
    recall_count = sum(len(d["pubmed_recall"].dropna()) for d in split_data)
    print(f"Precision subplot: {precision_count} values")
    print(f"Recall subplot: {recall_count} values")

    ax = axes[0]
    ax.hist(
        [d["pubmed_precision"] for d in split_data],
        bins=bins,
        stacked=True,
        color=colors_split,
        label=labels_split,
    )
    ax.set_xlabel("Precision")
    ax.set_ylabel("Frequency")
    ax.set_title("Precision Distribution")

    ax = axes[1]
    ax.hist(
        [d["pubmed_recall"] for d in split_data],
        bins=bins,
        stacked=True,
        color=colors_split,
        label=labels_split,
    )
    ax.set_xlabel("Recall")
    ax.set_title("Recall Distribution")
    ax.legend()
    ax.set_yticklabels([])
    
    # Sync y-axis between both plots
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, y_max)
    axes[1].set_ylim(0, y_max)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms to {out_path}")


def plot_all_heatmaps(top_k_type, betas_key, out_dir):
    dataframes = []
    for top_k_type in ["cosine", "fixed", "pos_count"]:
        path = find_qg_results_file(
            CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key=betas_key
        )
        df = get_qg_results(path, min_positive_threshold=50)
        df["top_k_type"] = top_k_type
        dataframes.append(df)
    data = (
        pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    )
    # Filter to only num_positive >= 50
    data = data[data["num_positive"] >= 50]
    # p_r_pairs = get_precision_recall_pairs_from_jsonl(path)
    # data = pd.DataFrame(p_r_pairs, columns=["Precision", "Recall", "num_positive"])
    plot_precision_recall_heatmap(
        data,
        out_path=os.path.join(out_dir, "precision_recall_heatmap.png"),
        min_positive_threshold=50,
    )
    plot_precision_recall_scatter(
        data,
        out_path=os.path.join(out_dir, "precision_recall_scatter.png"),
        min_positive_threshold=50,
    )
    plot_precision_recall_histograms(
        data,
        out_path=os.path.join(out_dir, "precision_recall_histograms.png"),
        bins=40,
    )


if __name__ == "__main__":
    # Output directory for thesis images
    out_dir = (
        "../master-thesis-writing/writing/thesis/images/graphs/precision_recall_heatmap"
    )
    top_k_type = "cosine"
    betas_key = "50"
    os.makedirs(out_dir, exist_ok=True)
    plot_all_heatmaps(
        betas_key=betas_key,
        out_dir=out_dir,
        top_k_type=top_k_type,
    )
