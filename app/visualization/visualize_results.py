import json
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from app.dataset.utils import load_statistics_data, statistics_base_path

def analyze_dataframe_results(df, variables, metrics):
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
    
    # variables = [
    #     "max_depth",
    #     "min_samples_split",
    #     "min_impurity_decrease_start",
    #     "min_impurity_decrease_end",
    #     "top_k_or_candidates",
    #     "class_weight",
    #     "n_docs",
    #     "min_df",
    #     "max_df",
    #     "mesh",
    # ]
    # metrics = ["precision", "recall", "f1", "time_seconds", "ORs", "IFs"]

    for var in variables:
        if var not in df.columns:
            continue
        grouped = df.groupby(var)[[m[0] for m in metrics]].mean().reset_index()

        # --- Dual y-axis plot ---
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = ax1.twinx()

        # Left y-axis (precision, recall, f1)
        for m in metrics:
            if m[4] == "axis1":
                ax1.plot(grouped[var], grouped[m[0]], marker=m[1], label=m[0], color=m[2], linestyle=m[3])
            else:
                ax2.plot(grouped[var], grouped[m[0]], marker=m[1], label=m[0], color=m[2])
    
        ax1.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis1"]))
        ax1.tick_params(axis="y", labelcolor="black")

        # Right axis: time and size
        ax2.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis2"]))
        ax2.tick_params(axis="y", labelcolor="tab:green")
        
        ax1.set_xlabel(var)

        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(f"Effect of {var} on Precision, Recall, F1, Time, and OR/IF counts")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        # plt.show()
        out_path = statistics_base_path() / "../images" / f"effects_of_{var}.png"
        os.makedirs(statistics_base_path() / "../images", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    return df

def analyze_and_plot_best_files_from_df(df, top_n=10, opt_metric="f1_dt", metrics=[]):
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
    df = df.sort_values(opt_metric, ascending=False)
    top_df = df.head(top_n)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    x = range(len(top_df))
    i = 1
    for file in list(top_df["file"]):
        print(i, file)
        i += 1

    for m in metrics:
        if m[4] == "axis1":
            ax1.plot(x, top_df[m[0]], marker=m[1], label=m[0], color=m[2], linestyle=m[3])
        else:
            ax2.plot(x, top_df[m[0]], marker=m[1], label=m[0], color=m[2])
    
    ax1.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis1"]))
    ax1.tick_params(axis="y", labelcolor="black")

    # Right axis: time and size
    ax2.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis2"]))
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
    out_path = statistics_base_path() / "../images" / f"analyze_and_plot_best_files_from_df.png"
    os.makedirs(statistics_base_path() / "../images", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")
    
    return top_df

def visualize_results(
    model="GreedyORDecisionTree",
    filter_vars={"total_docs": 433660},
    qg = False,
):
    if not qg:
        metrics = [("precision_dt", "o", "tab:blue", None, "axis1"), 
                    ("recall_dt", "s", "tab:orange", None, "axis1"), 
                    ("f1_dt", "D", "tab:purple", None, "axis1"), 
                    ("time_seconds_dt", "^", "tab:green", "--", "axis2") 
                    ]
    else:
        metrics = [("precision_qg", "o", "tab:blue", None, "axis1"), 
                  ("recall_dg", "s", "tab:orange", None, "axis1"), 
                  ("f1_qg", "D", "tab:purple", None, "axis1"), 
                  ("time_seconds_qg", "^", "tab:green", "--", "axis2"), 
                  ("size_ORs_qg", "x", "tab:red", "-.", "axis2"), 
                  ("size_IFs_qg", "*", "tab:brown", ":", "axis2"), 
                  ]
    
    
    df, params = load_statistics_data(
        filter_vars=filter_vars,
        qg=qg
    )

    analyze_and_plot_best_files_from_df(
        df,
        top_n=10,
        metrics=metrics,
    )

    df = analyze_dataframe_results(df, variables=params, metrics=metrics)
    

if __name__ == "__main__":
    visualize_results()
