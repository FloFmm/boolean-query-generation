import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from app.dataset.utils import find_qg_results_file, get_qg_results, load_vectors
from app.config.config import (
    BOW_PARAMS,
    COLORS,
    CURRENT_BEST_RUN_FOLDER,
    FIGURE_CONFIG,
    apply_matplotlib_style,
)
from app.helper.helper import f_beta
from app.statistics.duplicate_features import calculate_duplicate_features_percentage
from app.visualization.helper import pretty_print_param

# Apply consistent styling for all figures
apply_matplotlib_style()


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
    pi_interval: int | None = 80,
    pi_show_f_score: bool = True,
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
        pi_interval: Percentile interval width for error bands (e.g. 80 for 10th-90th).
            Set to None to disable error bars.
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
                pi_interval,
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

            if pi_interval is not None:
                ax.fill_between(x, grouped[f"{precision_col}_plo"], grouped[f"{precision_col}_phi"], alpha=0.15, color=COLORS["precision"])
                ax.fill_between(x, grouped[f"{recall_col}_plo"], grouped[f"{recall_col}_phi"], alpha=0.15, color=COLORS["recall"])
                if pi_show_f_score:
                    ax.fill_between(x, grouped["f50_plo"], grouped["f50_phi"], alpha=0.15, color=COLORS["f_score"])

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
        if n_plots == 1:
            axes = np.asarray(axes).reshape(2, 1)

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
                pi_interval,
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
                if pi_interval is not None:
                    ax.fill_between(x, grouped[f"{precision_col}_plo"], grouped[f"{precision_col}_phi"], alpha=0.15, color=COLORS["precision"])
                    ax.fill_between(x, grouped[f"{recall_col}_plo"], grouped[f"{recall_col}_phi"], alpha=0.15, color=COLORS["recall"])
                    if pi_show_f_score:
                        ax.fill_between(x, grouped["f50_plo"], grouped["f50_phi"], alpha=0.15, color=COLORS["f_score"])
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
    pi_interval: int | None = 80,
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
    elif size_key == "avg_term_len" or size_key == "avg_df" or size_key == "duplicate_pct_exact" or size_key == "duplicate_pct_substring":
        if size_key in working.columns:
            working["size_value"] = working[size_key]
        size_label = size_key
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

    working["f50"] = working.apply(
        lambda row: f_beta(row[precision_col], row[recall_col], beta=50), axis=1
    )

    # Compute percentile bounds from pi_interval
    if pi_interval is not None:
        p_lo = (100 - pi_interval) / 2
        p_hi = 100 - p_lo

    if bin_size is not None:
        max_val = working["size_value"].max()
        bins = np.arange(0, max_val + bin_size, bin_size)
        working["size_bin"] = pd.cut(
            working["size_value"], bins=bins, include_lowest=True, right=False
        )
        agg_dict = {
            precision_col: (precision_col, "mean"),
            recall_col: (recall_col, "mean"),
            "f50": ("f50", "mean"),
            "size_value_mean": ("size_value", "mean"),
            "count": ("size_value", "count"),
        }
        if pi_interval is not None:
            agg_dict.update({
                f"{precision_col}_plo": (precision_col, lambda x: np.percentile(x, p_lo) if len(x) > 0 else np.nan),
                f"{precision_col}_phi": (precision_col, lambda x: np.percentile(x, p_hi) if len(x) > 0 else np.nan),
                f"{recall_col}_plo": (recall_col, lambda x: np.percentile(x, p_lo) if len(x) > 0 else np.nan),
                f"{recall_col}_phi": (recall_col, lambda x: np.percentile(x, p_hi) if len(x) > 0 else np.nan),
                "f50_plo": ("f50", lambda x: np.percentile(x, p_lo) if len(x) > 0 else np.nan),
                "f50_phi": ("f50", lambda x: np.percentile(x, p_hi) if len(x) > 0 else np.nan),
            })
        grouped = (
            working.groupby("size_bin", observed=False)
            .agg(**agg_dict)
            .reset_index()
            .sort_values("size_value_mean")
        )
        x = grouped["size_value_mean"].values
        counts = grouped["count"].values
    else:
        agg_dict = {
            precision_col: (precision_col, "mean"),
            recall_col: (recall_col, "mean"),
            "f50": ("f50", "mean"),
            "count": ("size_value", "count"),
        }
        if pi_interval is not None:
            agg_dict.update({
                f"{precision_col}_plo": (precision_col, lambda x: np.percentile(x, p_lo)),
                f"{precision_col}_phi": (precision_col, lambda x: np.percentile(x, p_hi)),
                f"{recall_col}_plo": (recall_col, lambda x: np.percentile(x, p_lo)),
                f"{recall_col}_phi": (recall_col, lambda x: np.percentile(x, p_hi)),
                "f50_plo": ("f50", lambda x: np.percentile(x, p_lo)),
                "f50_phi": ("f50", lambda x: np.percentile(x, p_hi)),
            })
        grouped = (
            working.groupby("size_value")
            .agg(**agg_dict)
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

    return x, counts, grouped, size_label

def compute_avg_term_len(rules):
    if not isinstance(rules, list):
        return None
    term_lens = []
    for rule in rules:
        for term in rule:
            term_lens.append(len(term[0]))
    return sum(term_lens) / len(term_lens)

def plot_size_impact(top_k_types, betas_key, min_points_in_bucket, out_dir, y_break=None, pi_interval=None, pi_show_f_score=True):
    dataframes = []
    for top_k_type in top_k_types:
        path = find_qg_results_file(
            CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key=betas_key
        )
        df = get_qg_results(path, min_positive_threshold=50)
        dataframes.append(df)

    dataframe = (
        pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
    )

    dataframe["avg_term_len"] = dataframe["rules"].apply(compute_avg_term_len)
    
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    feature_doc_freq = np.asarray(X.sum(axis=0)).ravel()

    def compute_avg_df(rules):
        f_ids = [f_id for rule in rules for term in rule for f_id in term[0]]
        return float(np.median(feature_doc_freq[f_ids]))

    dataframe["avg_df"] = dataframe["rules"].apply(compute_avg_df)
    dataframe["duplicate_pct_exact"] = dataframe["rules"].apply(lambda rules: calculate_duplicate_features_percentage(rules, feature_names, exact_match=True))
    dataframe["duplicate_pct_substring"] = dataframe["rules"].apply(lambda rules: calculate_duplicate_features_percentage(rules, feature_names, exact_match=False))

    os.makedirs(out_dir, exist_ok=True)

    # Generate grouped plots (3 next to each other)
    groups = [
        ("group1", [("paths", None), ("avg_path_len", 0.5), ("avg_term_len", 0.5)]),
        ("group2", [("all_ORs", 5), ("ANDs", 2), ("NOTs", None)]),
        ("group3", [("avg_df", 3000), ("duplicate_pct_exact", 7), ("duplicate_pct_substring", 7)]),
        ("group4", [("ops_count", 10), ("added_ORs", 5), ("synonym_ORs", 5)]),
    ]

    for group_name, size_configs in groups:
        plot_performance_by_query_size_multi(
            dataframe,
            size_keys=size_configs,
            out_path=os.path.join(out_dir, f"performance_{group_name}.png"),
            min_positive_threshold=50,
            y_break=y_break,
            min_points_in_bucket=min_points_in_bucket,
            pi_interval=pi_interval,
            pi_show_f_score=pi_show_f_score,
        )

if __name__ == "__main__":
    # size impact to performance
    # WRITINGTD: took all three (cosine, fixed, pos_count) together to have more points for size-based analysis.
    # took F50 and "wspace": 0.05d buckets with only one datapoint (min_points_in_bucket=2) to reduce noise
    min_points_in_bucket = 3
    betas_key = "50"
    top_k_types = ["cosine", "fixed", "pos_count"]
    out_dir = "../master-thesis-writing/writing/thesis/images/graphs/size_impact"
    plot_size_impact(
        top_k_types=top_k_types,
        betas_key=betas_key,
        min_points_in_bucket=min_points_in_bucket,
        out_dir=out_dir,
        y_break=((0.0, 0.05), (0.5, 1.0)),
        pi_interval=80,
        pi_show_f_score=False,
    )
