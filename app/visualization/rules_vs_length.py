import os

import matplotlib.pyplot as plt

from app.config.config import COLORS, CURRENT_BEST_RUN_FOLDER, FIGURE_CONFIG, apply_matplotlib_style
from app.dataset.utils import get_qg_results
from app.tree_learning.query_generation import query_size
from app.visualization.helper import pretty_print_param
from app.visualization.size_impact import compute_avg_term_len

apply_matplotlib_style()

def plot_scatter(
    x_values,
    y_values,
    x_label,
    y_label,
    out_path,
    alpha=0.5,
    size=None,
    
):
    """
    Plots a scatter plot of x-recall pairs.

    Parameters:
    - data: A DataFrame containing 'pubmed_x' and 'pubmed_recall' columns.
    - alpha: Transparency of dots (default 0.5).
    - size: Size of dots (default 20).
    """

    fig, ax = plt.subplots(
            figsize=(
                FIGURE_CONFIG["half_width"],
                FIGURE_CONFIG["half_width"] #* FIGURE_CONFIG["aspect_ratio"],
            )
        )
    ax.scatter(
        x_values,
        y_values,
        alpha=alpha,
        s=size if size is not None else 20,
        c=COLORS["primary"],
    )

    # Determine range from data with small padding
    x_min, x_max = (
        min(x_values),
        max(x_values),
    )
    y_min, y_max = min(y_values), max(y_values)

    # padding = 0.0001
    # ax.set_xlim(max(0, x_min - padding), min(1, x_max + padding))
    # ax.set_ylim(max(0, y_min - padding), min(1, y_max + padding))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.savefig(out_path)
    plt.close()
    print(f"Saved scatter plot to {out_path}")
    
if __name__ == "__main__":
    
    out_dir = "../master-thesis-writing/writing/thesis/images/graphs/rule_size_scatter"
    os.makedirs(out_dir, exist_ok=True)
    betas_key = "50"
    top_k_types = ["cosine"]
    datasets = None
    dataframe = get_qg_results(CURRENT_BEST_RUN_FOLDER, min_positive_threshold=50, datasets=datasets, restrict_betas=[betas_key], top_k_types=top_k_types)
    x_value = "paths"
    pairs = [("query_size_paths", "avg_term_len"), ("avg_path_len", "avg_term_len")]
    for x_value, y_value in pairs:
        print(len(dataframe), "total samples in results dataframe")
        dataframe["avg_term_len"] = dataframe["rules"].apply(compute_avg_term_len)
        dataframe["avg_path_len"] = dataframe["rules"].apply(lambda x: query_size(x)["avg_path_len"])
        x_values = dataframe[x_value].to_list()
        y_values = dataframe[y_value].to_list()
        
        # Scale dot sizes by count of (x, y) pairs
        from collections import Counter
        import numpy as np

        points = list(zip(x_values, y_values))
        counts = Counter(points)
        unique_points = list(counts.keys())
        sizes = np.array([counts[pt] for pt in unique_points]) * 3.0  # scale factor
        x_unique, y_unique = zip(*unique_points)

        plot_scatter(
            x_values=x_unique,
            y_values=y_unique,
            x_label=pretty_print_param(x_value.replace("query_size_", "")),
            y_label=pretty_print_param(y_value.replace("query_size_", "")),
            out_path=f"{out_dir}/{x_value}_vs_{y_value}.png",
            size=sizes
        )
    