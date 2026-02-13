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
from app.helper.helper import f_beta
from app.visualization.helper import pretty_print_param

# Apply consistent styling for all figures
apply_matplotlib_style()


def plot_precision_recall_heatmap(
    data, out_path="precision_recall_heatmap.png", bins=20, min_positive_threshold=50
):
    """
    Plots a smooth heatmap showing frequency of precision-recall combinations.

    Parameters:
    - data: A DataFrame containing 'Precision' and 'Recall' columns.
    - bins: Number of bins for both axes (default 20).
    """
    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()

    # "wspace": 0.05 all triples fromd ata that do not satisify min_positive_threshold
    data = data[data["num_positive"] >= min_positive_threshold]

    # Determine range from data with small padding
    precision_min, precision_max = data["Precision"].min(), data["Precision"].max()
    recall_min, recall_max = data["Recall"].min(), data["Recall"].max()
    print(f"Precision range: {precision_min:.4f} - {precision_max:.4f}")
    print(f"Recall range: {recall_min:.4f} - {recall_max:.4f}")
    # Add small padding to avoid edge cases
    padding = 0.0
    precision_range = [max(0, precision_min - padding), min(1, precision_max + padding)]
    recall_range = [max(0, recall_min - padding), min(1, recall_max + padding)]

    # Create 2D histogram for smooth heatmap
    heatmap, xedges, yedges = np.histogram2d(
        data["Precision"],
        data["Recall"],
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
    - data: A DataFrame containing 'Precision' and 'Recall' columns.
    - alpha: Transparency of dots (default 0.5).
    - size: Size of dots (default 20).
    """
    # "wspace": 0.05 all triples fromd ata that do not satisify min_positive_threshold
    data = data[data["num_positive"] >= min_positive_threshold]

    fig, ax = plt.subplots()  # uses default figsize from apply_matplotlib_style()

    ax.scatter(
        data["Precision"], data["Recall"], alpha=alpha, s=size, c=COLORS["primary"]
    )

    # Determine range from data with small padding
    precision_min, precision_max = data["Precision"].min(), data["Precision"].max()
    recall_min, recall_max = data["Recall"].min(), data["Recall"].max()

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
    data, out_path="precision_recall_histograms.png", bins=100, threshold=50
):
    """
    Plots two histograms (precision and recall) with stacked bars based on number of positives.

    Parameters:
    - data: A DataFrame containing 'Precision', 'Recall', and 'num_positive' columns.
    - bins: Number of bins for histograms (default 20).
    - threshold: Threshold for splitting data (default 50).
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_CONFIG["full_width"], FIGURE_CONFIG["full_width"] * 0.45)
    )

    # Split data by threshold
    data_high = data[data["num_positive"] >= threshold]
    data_low = data[data["num_positive"] < threshold]

    # Precision histogram - use matplotlib's native stacking
    ax = axes[0]
    ax.hist(
        [data_high["Precision"], data_low["Precision"]],
        bins=bins,
        stacked=True,
        color=[COLORS["precision"], COLORS["precision_light"]],
        label=[f"≥ {threshold} positives", f"< {threshold} positives"],
    )
    ax.set_xlabel("Precision")
    ax.set_ylabel("Frequency")
    ax.set_title("Precision Distribution")
    ax.legend()

    # Recall histogram - use matplotlib's native stacking
    ax = axes[1]
    ax.hist(
        [data_high["Recall"], data_low["Recall"]],
        bins=bins,
        stacked=True,
        color=[COLORS["recall"], COLORS["recall_light"]],
        label=[f"≥ {threshold} positives", f"< {threshold} positives"],
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Frequency")
    ax.set_title("Recall Distribution")
    ax.legend()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histograms to {out_path}")


def plot_performance_by_query_size_multi(
    df: pd.DataFrame,
    size_keys: list[tuple[str, int | None]],
    out_path: str,
    precision_col: str = "pubmed_precision",
    recall_col: str = "pubmed_recall",
    query_size_col: str = "query_size",
    min_positive_threshold: int | None = 0,
    y_break: tuple[tuple[float, float], tuple[float, float]] | None = (
        (0.0, 0.04),
        (0.7, 1.0),
    ),
    min_points_in_bucket: int = 1,
) -> None:
    """
    Plot mean precision, recall, and F50 score for multiple size metrics side-by-side.

    Args:
        df: DataFrame with precision/recall and a query_size dict column.
        size_keys: List of (size_key, bin_size) tuples.
        out_path: Output path for the figure.
        precision_col: Column name for precision.
        recall_col: Column name for recall.
        query_size_col: Column name containing query_size dict.
        min_positive_threshold: Optional filter on num_positive.
        min_points_in_bucket: Minimum number of points required in a bucket to include it.
    """
    n_plots = len(size_keys)
    if n_plots == 0:
        raise ValueError("size_keys must contain at least one entry")

    if y_break is None:
        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(FIGURE_CONFIG["full_width"], FIGURE_CONFIG["full_width"] * 0.4),
            gridspec_kw={"wspace": 0.1},
        )
        if n_plots == 1:
            axes = [axes]

        for idx, (size_key, bin_size) in enumerate(size_keys):
            ax = axes[idx]
            data = _prepare_size_data(
                df,
                size_key,
                bin_size,
                precision_col,
                recall_col,
                query_size_col,
                min_positive_threshold,
                min_points_in_bucket,
            )
            if data is None:
                continue

            x, counts, grouped, size_label = data

            ax.plot(
                x,
                grouped[precision_col],
                marker="o",
                label="Precision",
                color=COLORS["precision"],
            )
            ax.plot(
                x,
                grouped[recall_col],
                marker="s",
                label="Recall",
                color=COLORS["recall"],
            )
            ax.plot(x, grouped["f50"], marker="D", label="F50", color=COLORS["f_score"])

            ax.set_xlabel(pretty_print_param(size_label))
            ax.set_title(f"{pretty_print_param(size_label)}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if idx == n_plots - 1:
                ax.legend(loc="best")
            ax.locator_params(axis="y", nbins=4)

            y_min, y_max = ax.get_ylim()
            y_text = y_min + (y_max - y_min) * 0.02
            for xi, c in zip(x, counts):
                ax.text(
                    xi,
                    y_text,
                    str(int(c)),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        (low_min, low_max), (high_min, high_max) = y_break
        fig, axes = plt.subplots(
            2,
            n_plots,
            sharex="col",
            figsize=(FIGURE_CONFIG["full_width"], FIGURE_CONFIG["full_width"] * 0.6),
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15, "wspace": 0.07},
        )

        for idx, (size_key, bin_size) in enumerate(size_keys):
            ax_high = axes[0, idx]
            ax_low = axes[1, idx]

            data = _prepare_size_data(
                df,
                size_key,
                bin_size,
                precision_col,
                recall_col,
                query_size_col,
                min_positive_threshold,
                min_points_in_bucket,
            )
            if data is None:
                continue

            x, counts, grouped, size_label = data

            for ax in (ax_high, ax_low):
                ax.plot(
                    x,
                    grouped[precision_col],
                    marker="o",
                    label="Precision",
                    color=COLORS["precision"],
                )
                ax.plot(
                    x,
                    grouped[recall_col],
                    marker="s",
                    label="Recall",
                    color=COLORS["recall"],
                )
                ax.plot(
                    x, grouped["f50"], marker="D", label="F50", color=COLORS["f_score"]
                )
                ax.grid(True, linestyle="--", alpha=0.6)

            ax_high.set_ylim(high_min, high_max)
            ax_low.set_ylim(low_min, low_max)
            ax_high.locator_params(axis="y", nbins=4)
            ax_low.locator_params(axis="y", nbins=4)

            ax_high.spines["bottom"].set_visible(False)
            ax_low.spines["top"].set_visible(False)
            ax_high.tick_params(labeltop=False, bottom=False, labelbottom=False)
            ax_low.xaxis.tick_bottom()

            d = 0.008
            kwargs = dict(
                transform=ax_high.transAxes, color="k", clip_on=False, linewidth=0.8
            )
            ax_high.plot((-d, +d), (-d, +d), **kwargs)
            ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            kwargs.update(transform=ax_low.transAxes)
            ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            ax_low.set_xlabel(pretty_print_param(size_label))
            if idx != 0:
                # remove y ticks values. but keep ticks
                ax_high.set_yticklabels([])
                ax_low.set_yticklabels([])
                # now remove also the ticks
                # ax_high.tick_params(left=False)
                # ax_low.tick_params(left=False)

            # ax_high.set_title(f"{pretty_print_param(size_label)}")
            if idx == n_plots - 2:
                ax_low.legend(loc="best")

            y_min, y_max = ax_low.get_ylim()
            y_text = y_min + (y_max - y_min) * 0.02
            for xi, c in zip(x, counts):
                ax_low.text(
                    xi,
                    y_text,
                    str(int(c)),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Saved multi-plot to {out_path}")


def _prepare_size_data(
    df: pd.DataFrame,
    size_key: str,
    bin_size: int | None,
    precision_col: str,
    recall_col: str,
    query_size_col: str,
    min_positive_threshold: int | None,
    min_points_in_bucket: int,
):
    """Helper to prepare data for a single size metric."""
    working = df.copy()
    if min_positive_threshold is not None and "num_positive" in working.columns:
        threshold_value = int(min_positive_threshold)
        working = working[working["num_positive"] >= threshold_value]

    if query_size_col not in working.columns:
        return None

    if size_key == "ops_count":
        keys = ["ORs", "ANDs", "NOTs", "added_ORs", "synonym_ORs"]
        working["size_value"] = working[query_size_col].apply(
            lambda x: sum(x.get(k, 0) for k in keys) if isinstance(x, dict) else None
        )
        size_label = "ops_count"
    elif size_key == "all_ORs":
        keys = ["ORs", "added_ORs", "synonym_ORs"]
        working["size_value"] = working[query_size_col].apply(
            lambda x: sum(x.get(k, 0) for k in keys) if isinstance(x, dict) else None
        )
        size_label = "all_ORs"
    elif size_key == "avg_term_len":
        if "avg_term_len" in working.columns:
            working["size_value"] = working["avg_term_len"]
        else:
            return None
        size_label = "avg_term_len"
    elif isinstance(size_key, (list, tuple)):
        keys = list(size_key)
        working["size_value"] = working[query_size_col].apply(
            lambda x: sum(x.get(k, 0) for k in keys) if isinstance(x, dict) else None
        )
        size_label = "+".join(keys)
    else:
        working["size_value"] = working[query_size_col].apply(
            lambda x: x.get(size_key) if isinstance(x, dict) else None
        )
        size_label = size_key

    working = working.dropna(subset=["size_value", precision_col, recall_col])
    if working.empty:
        return None

    if bin_size is not None:
        max_val = working["size_value"].max()
        bins = np.arange(0, max_val + bin_size, bin_size)
        working["size_bin"] = pd.cut(
            working["size_value"], bins=bins, include_lowest=True, right=False
        )
        grouped = (
            working.groupby("size_bin", observed=False)
            .agg(
                **{
                    precision_col: (precision_col, "mean"),
                    recall_col: (recall_col, "mean"),
                    "size_value_mean": ("size_value", "mean"),
                    "count": ("size_value", "count"),
                }
            )
            .reset_index()
            .sort_values("size_value_mean")
        )
        x = grouped["size_value_mean"].values
        counts = grouped["count"].values
    else:
        grouped = (
            working.groupby("size_value")
            .agg(
                **{
                    precision_col: (precision_col, "mean"),
                    recall_col: (recall_col, "mean"),
                    "count": ("size_value", "count"),
                }
            )
            .reset_index()
            .sort_values("size_value")
        )
        x = grouped["size_value"].values
        counts = grouped["count"].values

    grouped = grouped[grouped["count"] >= min_points_in_bucket].reset_index(drop=True)
    if grouped.empty:
        return None

    if bin_size is not None:
        x = grouped["size_value_mean"].values
    else:
        x = grouped["size_value"].values
    counts = grouped["count"].values

    grouped["f50"] = grouped.apply(
        lambda row: f_beta(row[precision_col], row[recall_col], beta=50), axis=1
    )

    return x, counts, grouped, size_label

def plot_all_heatmaps(top_k_type, betas_key, out_dir):
    # Example data with more points for smooth visualization
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key=betas_key
    )
    if path is None:
        print(f"No matching qg_results.jsonl found with top_k_type={top_k_type}, betas_key={betas_key}")
        exit(1)
    print(f"Found: {path}")
    p_r_pairs = get_precision_recall_pairs_from_jsonl(path)
    data = pd.DataFrame(p_r_pairs, columns=["Precision", "Recall", "num_positive"])
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
        threshold=50,
    )

def plot_size_impact(top_k_types, betas_key, min_points_in_bucket, out_dir):
    dataframes = []
    for top_k_type in top_k_types:
        path = find_qg_results_file(
            CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key=betas_key
        )
        if path is not None:
            df = get_qg_results(path, min_positive_threshold=50)
            dataframes.append(df)

    dataframe = (
        pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    )

    def compute_avg_term_len(rules):
        if not isinstance(rules, list):
            return None
        term_lens = []
        for rule in rules:
            for term in rule:
                term_lens.append(len(term[0]))
        return sum(term_lens) / len(term_lens)

    dataframe["avg_term_len"] = dataframe["rules"].apply(compute_avg_term_len)

    os.makedirs(out_dir, exist_ok=True)

    # Generate grouped plots (3 next to each other)
    groups = [
        ("group1", [("paths", None), ("avg_path_len", 0.5), ("avg_term_len", 0.5)]),
        ("group2", [("all_ORs", 5), ("ANDs", 2), ("NOTs", None)]),
        ("group3", [("ops_count", 10), ("added_ORs", 5), ("synonym_ORs", 5)]),
    ]

    for group_name, size_configs in groups:
        plot_performance_by_query_size_multi(
            dataframe,
            size_keys=size_configs,
            out_path=os.path.join(out_dir, f"performance_{group_name}.png"),
            min_positive_threshold=50,
            y_break=((0.0, 0.03), (0.7, 1.0)),
            min_points_in_bucket=min_points_in_bucket,
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

    # size impact to performance
    # WRITINGTD: took all three (cosine, fixed, pos_count) together to have more points for size-based analysis.
    # took F50 and "wspace": 0.05d buckets with only one datapoint (min_points_in_bucket=2) to reduce noise
    min_points_in_bucket = 2
    top_k_types = ["cosine", "fixed", "pos_count"]
    out_dir = "../master-thesis-writing/writing/thesis/images/graphs/size_imapact"
    plot_size_impact(
        top_k_types=top_k_types,
        betas_key=betas_key,
        min_points_in_bucket=min_points_in_bucket,
        out_dir=out_dir,
    )
