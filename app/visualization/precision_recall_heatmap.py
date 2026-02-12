import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from app.dataset.utils import find_qg_results_file
from app.config.config import COLORS, COLORMAPS, CURRENT_BEST_RUN_FOLDER, FIGURE_CONFIG, apply_matplotlib_style

# Apply consistent styling for all figures
apply_matplotlib_style()

def plot_precision_recall_heatmap(data, out_path="precision_recall_heatmap.png", bins=20, min_positive_threshold=50):
    """
    Plots a smooth heatmap showing frequency of precision-recall combinations.
    
    Parameters:
    - data: A DataFrame containing 'Precision' and 'Recall' columns.
    - bins: Number of bins for both axes (default 20).
    """
    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()
    
    # remove all triples fromd ata that do not satisify min_positive_threshold
    data = data[data['num_positive'] >= min_positive_threshold]
    
    # Determine range from data with small padding
    precision_min, precision_max = data['Precision'].min(), data['Precision'].max()
    recall_min, recall_max = data['Recall'].min(), data['Recall'].max()
    print(f"Precision range: {precision_min:.4f} - {precision_max:.4f}")
    print(f"Recall range: {recall_min:.4f} - {recall_max:.4f}")
    # Add small padding to avoid edge cases
    padding = 0.0
    precision_range = [max(0, precision_min - padding), min(1, precision_max + padding)]
    recall_range = [max(0, recall_min - padding), min(1, recall_max + padding)]
    
    # Create 2D histogram for smooth heatmap
    heatmap, xedges, yedges = np.histogram2d(
        data['Precision'], data['Recall'], 
        bins=bins, 
        range=[precision_range, recall_range]
    )
    
    # Plot as smooth heatmap
    im = ax.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[precision_range[0], precision_range[1], recall_range[0], recall_range[1]],
        aspect='auto',
        cmap=COLORMAPS['heatmap'],
        interpolation='bilinear'
    )
    
    plt.colorbar(im, ax=ax, label='Frequency')
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Frequency Heatmap")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {out_path}")


def plot_precision_recall_scatter(data, out_path="precision_recall_scatter.png", alpha=0.5, size=20, min_positive_threshold=50):
    """
    Plots a scatter plot of precision-recall pairs.
    
    Parameters:
    - data: A DataFrame containing 'Precision' and 'Recall' columns.
    - alpha: Transparency of dots (default 0.5).
    - size: Size of dots (default 20).
    """
    # remove all triples fromd ata that do not satisify min_positive_threshold
    data = data[data['num_positive'] >= min_positive_threshold]
    
    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()
    
    ax.scatter(data['Precision'], data['Recall'], alpha=alpha, s=size, c=COLORS['primary'])
    
    # Determine range from data with small padding
    precision_min, precision_max = data['Precision'].min(), data['Precision'].max()
    recall_min, recall_max = data['Recall'].min(), data['Recall'].max()
    
    padding = 0.0001
    ax.set_xlim(max(0, precision_min - padding), min(1, precision_max + padding))
    ax.set_ylim(max(0, recall_min - padding), min(1, recall_max + padding))
    
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision-Recall Scatter Plot")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
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


def plot_precision_recall_histograms(data, out_path="precision_recall_histograms.png", bins=100, threshold=50):
    """
    Plots two histograms (precision and recall) with stacked bars based on number of positives.
    
    Parameters:
    - data: A DataFrame containing 'Precision', 'Recall', and 'num_positive' columns.
    - bins: Number of bins for histograms (default 20).
    - threshold: Threshold for splitting data (default 50).
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_CONFIG["full_width"], 
                                             FIGURE_CONFIG["full_width"] * 0.45))
    
    # Split data by threshold
    data_high = data[data['num_positive'] >= threshold]
    data_low = data[data['num_positive'] < threshold]
    
    # Precision histogram - use matplotlib's native stacking
    ax = axes[0]
    ax.hist(
        [data_high['Precision'], data_low['Precision']], 
        bins=bins, 
        stacked=True,
        color=[COLORS['precision'], COLORS['precision_light']], 
        label=[f'≥ {threshold} positives', f'< {threshold} positives']
    )
    ax.set_xlabel("Precision")
    ax.set_ylabel("Frequency")
    ax.set_title("Precision Distribution")
    ax.legend()
    
    # Recall histogram - use matplotlib's native stacking
    ax = axes[1]
    ax.hist(
        [data_high['Recall'], data_low['Recall']], 
        bins=bins, 
        stacked=True,
        color=[COLORS['recall'], COLORS['recall_light']], 
        label=[f'≥ {threshold} positives', f'< {threshold} positives']
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Frequency")
    ax.set_title("Recall Distribution")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histograms to {out_path}")


if __name__ == "__main__":
    # Output directory for thesis images
    out_dir = "../master-thesis-writing/writing/thesis/images/graphs"
    top_k_type="cosine"
    betas_key="50"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Example data with more points for smooth visualization
    path = find_qg_results_file(CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key=betas_key)
    if path is None:
        print("No matching qg_results.jsonl found with cosine top_k_type")
        exit(1)
    print(f"Found: {path}")
    p_r_pairs = get_precision_recall_pairs_from_jsonl(path)
    data = pd.DataFrame(p_r_pairs, columns=["Precision", "Recall", "num_positive"])
    plot_precision_recall_heatmap(data, out_path=os.path.join(out_dir, "precision_recall_heatmap.png"), min_positive_threshold=50)
    plot_precision_recall_scatter(data, out_path=os.path.join(out_dir, "precision_recall_scatter.png"), min_positive_threshold=50)
    plot_precision_recall_histograms(data, out_path=os.path.join(out_dir, "precision_recall_histograms.png"), bins=40, threshold=50)