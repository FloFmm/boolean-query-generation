import json
import os
import re
import numpy as np
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
    qg = any([m[0].endswith('_qg') for m in metrics])
    
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
        df = df.replace([np.inf, -np.inf], None)
        if var not in df.columns or var == "file":
            continue
        print(var)
        grouped = df.groupby(var)[[m[0] for m in metrics]].mean().reset_index()
        
        x_vals = grouped[var].astype(str)
        x_pos = np.arange(len(x_vals))
        # --- Dual y-axis plot ---
        fig, ax1 = plt.subplots(figsize=(7, 5))
        # ax1.tick_params(axis="x", labelsize=6)
        ax2 = ax1.twinx()

        # Left y-axis (precision, recall, f1)
        for m in metrics:
            if m[4] == "axis1":
                ax1.plot(x_pos, grouped[m[0]], marker=m[1], label=m[0], color=m[2], linestyle=m[3])
            else:
                ax2.plot(x_pos, grouped[m[0]], marker=m[1], label=m[0], color=m[2])

        # --- X-tick labels with sample counts ---
        counts = df.groupby(var).size().reindex(grouped[var]).tolist()
        xtick_labels = [f"{v}\n({c})" for v, c in zip(x_vals, counts)]

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(xtick_labels, fontsize=6)

        # --- Dual y-axis plot ---
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
        out_path = statistics_base_path() / "../images" / f"effects_of_{var}_{'qg' if qg else'dt'}.png"
        os.makedirs(statistics_base_path() / "../images", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    return df

def analyze_and_plot_best_files_from_df(df, top_n=10, opt_metric="pubmed_f1_dt", metrics=[], worst=False):
    """
    Plot the top-performing configurations from an already prepared DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with averaged metrics per file (from load_experiment_data).
        top_n (int): Number of top configurations to plot.
    """
    if df.empty:
        print("DataFrame is empty.")
        return

    # keep only those with atleast 10 samples (TODO remove)
    df = df[df["samples_dt"] >= 10]
    
    # Sort by F1 and take top N
    df = df.sort_values(opt_metric, ascending=False)
    if worst:
        top_df = df.tail(top_n)
    else:
        top_df = df.head(top_n)

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    x = range(len(top_df))
    i = 1
    for file in list(top_df["file"]):
        print(i, file)
        i += 1

    # --- Build legend labels with Top-1 values ---
    if worst:
        top1 = top_df.iloc[-1]
    else:
        top1 = top_df.iloc[0]
    legend_labels = {}
    for m in metrics:
        metric_name = m[0]
        if metric_name in top1:
            legend_labels[metric_name] = f"{metric_name} ({top1[metric_name]:.3f})"
        else:
            legend_labels[metric_name] = metric_name

    # --- Plot lines ---
    for m in metrics:
        label = legend_labels[m[0]]
        if m[4] == "axis1":
            ax1.plot(
                x, top_df[m[0]],
                marker=m[1], label=label, color=m[2], linestyle=m[3]
            )
        else:
            ax2.plot(
                x, top_df[m[0]],
                marker=m[1], label=label, color=m[2]
            )
    
    ax1.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis1"]))
    ax1.tick_params(axis="y", labelcolor="black")

    # Right axis: time and size
    ax2.set_ylabel(" / ".join([m[0] for m in metrics if m[4] == "axis2"]))
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # X-axis: filenames or file identifiers
    # --- Build X-tick labels including DT and QG sample counts ---
    xtick_labels = []
    for _, row in top_df.iterrows():
        file_label = "\n".join(
            row["file"].split(',')
        ).replace("'", "").replace("GreedyORDecisionTree", "")
        if 'optimization_metric' in row:
          file_label += f"\nom={row['optimization_metric']}"
        if 'constraint' in row: 
            file_label += f"\nc={row['constraint']}"
            
        # two sample counts (DT + QG)
        if "samples_qg" in row:
            samples_label = f"DT:{row['samples_dt']}  QG:{row['samples_qg']}"
        else:
            samples_label = f"DT:{row['samples_dt']}"

        xtick_labels.append(file_label + "\n" + samples_label)
    
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        xtick_labels,
        rotation=0,
        ha="center",
        fontsize=6,
    )

    ax1.set_xlabel(f"Top {top_n} Files (Configurations)")

    # Combined legend
    text = "worst" if worst else "best"
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title(f"{'Bottom' if worst else 'Top'} {top_n} {text} Files (Average {opt_metric})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    # plt.show()
    out_path = statistics_base_path() / "../images" / f"{text}_{opt_metric}.png"
    os.makedirs(statistics_base_path() / "../images", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")
    
    return top_df

def visualize_results(
    model="GreedyORDecisionTree",
    filter_vars={"total_docs": 433660},
    qg = True,
):
    if not qg:
        opt_metric="f3_dt"
        metrics = [("precision_dt", "o", "tab:blue", None, "axis1"), 
                    ("recall_dt", "s", "tab:orange", None, "axis1"), 
                    ("f3_dt", "D", "tab:purple", None, "axis1"), 
                    ("time_seconds_dt", "^", "tab:green", "--", "axis2") 
                    ]
    else:
        opt_metric="pubmed_f3_qg"
        metrics = [("pubmed_precision_qg", "o", "tab:blue", None, "axis1"), 
                  ("pubmed_recall_qg", "s", "tab:orange", None, "axis1"), 
                  ("pubmed_f3_qg", "D", "tab:purple", None, "axis1"), 
                  ("query_size_ANDs_qg", "x", "tab:red", "-.", "axis2"), 
                  ("query_size_added_ORs_qg", "*", "tab:brown", ":", "axis2"), 
                  ("query_size_NOTs_qg", "*", "tab:pink", ":", "axis2"), 
                  ]
    
    
    df, params = load_statistics_data(
        filter_vars=filter_vars,
        qg=qg,
        metrics=metrics
    )

    analyze_and_plot_best_files_from_df(
        df,
        opt_metric=opt_metric,
        top_n=10,
        metrics=metrics,
    )
    analyze_and_plot_best_files_from_df(
        df,
        opt_metric=opt_metric,
        top_n=10,
        metrics=metrics,
        worst=True,
    )

    df = analyze_dataframe_results(df, variables=params, metrics=metrics)
    
if __name__ == "__main__":
    visualize_results(qg=True)
    visualize_results(qg=False)
