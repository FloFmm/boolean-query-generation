import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from app.dataset.utils import load_statistics_data, statistics_base_path

def analyze_dataframe_results(df):
    """
    Analyze performance metrics from a preloaded DataFrame.
    Plots Precision, Recall, F1, Time, and counts of OR/if separately for each variable.

    Args:
        df (pd.DataFrame): DataFrame containing experiment metrics with columns:
            - precision, recall, f1, time_seconds, or_count, if_count
            - and hyperparameters: max_depth, min_samples_split, etc.
    """
    
    if df.empty:
        print("DataFrame is empty.")
        return
    
    variables = [
        "max_depth",
        "min_samples_split",
        "min_impurity_decrease_start",
        "min_impurity_decrease_end",
        "top_k_or_candidates",
        "class_weight",
        "n_docs",
        "min_df",
        "max_df",
        "mesh",
    ]
    metrics = ["precision", "recall", "f1", "time_seconds", "ORs", "IFs", "threshold"]

    for var in variables:
        if var not in df.columns:
            continue
        grouped = df.groupby(var)[metrics].mean().reset_index()

        # --- Dual y-axis plot ---
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = ax1.twinx()

        # Left y-axis (precision, recall, f1)
        ax1.plot(
            grouped[var],
            grouped["precision"],
            marker="o",
            label="Precision",
            color="tab:blue",
        )
        ax1.plot(
            grouped[var],
            grouped["recall"],
            marker="s",
            label="Recall",
            color="tab:orange",
        )
        ax1.plot(
            grouped[var], grouped["f1"], marker="D", label="F1", color="tab:purple"
        )
        ax1.set_xlabel(var)
        ax1.set_ylabel("Precision / Recall / F1")
        ax1.tick_params(axis="y", labelcolor="black")

        # Right y-axis (time, OR count, IF count)
        ax2.plot(
            grouped[var],
            grouped["time_seconds"],
            marker="^",
            label="Time (s)",
            color="tab:green",
            linestyle="--",
        )
        ax2.plot(
            grouped[var],
            grouped["ORs"],
            marker="x",
            label="ORst",
            color="tab:red",
            linestyle="-.",
        )
        ax2.plot(
            grouped[var],
            grouped["IFs"],
            marker="*",
            label="IFs",
            color="tab:brown",
            linestyle=":",
        )
        ax1.plot(
            grouped[var],
            grouped["threshold"],
            marker="v",
            label="Threshold",
            color="tab:pink",
        )
        ax2.set_ylabel("Time / OR count / IF count")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(f"Effect of {var} on Precision, Recall, F1, Time, and OR/IF counts")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # plt.show()
        out_path = statistics_base_path() / "images" / f"effects_of_{var}.png"
        os.makedirs(statistics_base_path() / "images", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    return df

def analyze_and_plot_best_files_from_df(df, top_n=10):
    """
    Plot the top-performing configurations from an already prepared DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with averaged metrics per file (from load_experiment_data).
        top_n (int): Number of top configurations to plot.
    """
    if df.empty:
        print("DataFrame is empty.")
        return

    # Sort by F1 and take top N
    df = df.sort_values("f1", ascending=False)
    top_df = df.head(top_n)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    x = range(len(top_df))
    i = 1
    for file in list(top_df["file"]):
        print(i, file)
        i += 1

    # Left axis: precision/recall/F1
    ax1.plot(x, top_df["precision"], marker="o", label="Precision", color="tab:blue")
    ax1.plot(x, top_df["recall"], marker="s", label="Recall", color="tab:orange")
    ax1.plot(x, top_df["f1"], marker="D", label="F1", color="tab:purple")
    ax1.plot(x, top_df["threshold"], marker="v", label="Threshold", color="tab:pink")
    ax1.set_ylabel("Precision / Recall / F1")
    ax1.tick_params(axis="y", labelcolor="black")

    # Right axis: time and size
    ax2.plot(
        x,
        top_df["time_seconds"],
        marker="^",
        label="Time (s)",
        color="tab:green",
        linestyle="--",
    )
    ax2.plot(x, top_df["ORs"], marker="x", label="ORs", color="tab:red", linestyle="-.")
    ax2.plot(
        x, top_df["IFs"], marker="*", label="IFs", color="tab:brown", linestyle=":"
    )
    ax2.set_ylabel("Time (seconds) / ORs / IFs")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # X-axis: filenames or file identifiers
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [
            "\n".join(
                f.split(',')
            )
            for f in top_df["file"]
        ],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    ax1.set_xlabel(f"Top {top_n} Files (Configurations)")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title(f"Top {top_n} Best Files (Average F1)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    # plt.show()
    out_path = statistics_base_path() / "images" / f"analyze_and_plot_best_files_from_df.png"
    os.makedirs(statistics_base_path() / "images", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")
    
    return top_df

def visualize_results(
    model="GreedyORDecisionTree",
    filter_vars={"total_docs": 433660},
):
    df = load_statistics_data(
        filter_vars=filter_vars,
    )

    analyze_and_plot_best_files_from_df(
        df,
        top_n=10,
    )

    df = analyze_dataframe_results(df)
    

if __name__ == "__main__":
    visualize_results()
