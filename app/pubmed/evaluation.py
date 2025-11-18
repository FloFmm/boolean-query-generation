import json
import time
import os
import re
from pathlib import Path
from tqdm import tqdm
from Bio import Entrez
from app.pubmed.retrieval import classify_by_mesh
from app.tree_learning.logical_query_generation import (
    train_text_classifier,
    build_semantic_map,
    map_synonyms,
    build_synonym_map,
)
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
import pandas as pd
import matplotlib.pyplot as plt

# from imodels import SkopeRulesClassifier, DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import List


def load_completed_mesh_terms(jsonl_path: Path):
    """Load already processed MeSH terms from JSONL to skip duplicates."""
    if not jsonl_path.exists():
        return set()
    completed = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed.add(record["mesh_term"])
            except json.JSONDecodeError:
                continue
    return completed


def train_all_mesh_terms_jsonl(
    models,
    baseline_folder: str = "./data/pubmed/baseline",
    output_path: str = "data/pubmed/statistics/classifier_learning/",
    max_mesh_terms=None,
    mesh_terms: List[str] = None,
    skip_existing=False,
    n_docs=1_000_000,
    min_f_occ={0: 20, 1: 3},
):
    """
    Train text classifiers for all MeSH terms and save results incrementally as JSONL.
    Skips terms already in the output file.
    """

    docs_by_pmid, pmids_by_mesh = classify_by_mesh(baseline_folder, n_docs)
    if not mesh_terms:
        mesh_terms = list(pmids_by_mesh.keys())

    if max_mesh_terms:
        mesh_terms = mesh_terms[:max_mesh_terms]

    unique_words = set(
        word for doc in docs_by_pmid.values() for word in doc["bag_of_words"].split()
    )
    synonym_map = build_synonym_map(list(unique_words))
    for doc in docs_by_pmid.values():
        doc["bag_of_words_synonyms"] = map_synonyms(doc["bag_of_words"], synonym_map)
    print("unique words:", len(unique_words))

    for model in models:
        file_path = Path(
            os.path.join(
                output_path,
                f"{model}_n_docs={n_docs / 1_000:.0f}k_min_f_occ_={min_f_occ}.jsonl",
            )
        )
        if skip_existing:
            completed_mesh = load_completed_mesh_terms(file_path)
            print(
                f"Already computed {len(completed_mesh)} MeSH terms, skipping those..."
            )

        with file_path.open("a", encoding="utf-8") as out_f:
            for mesh in mesh_terms:
                if skip_existing and mesh in completed_mesh:
                    continue
                relevant_ids = pmids_by_mesh[mesh]
                if not relevant_ids:
                    print(f"Skipping {mesh}, because it does not occur in any doc")
                    continue

                relevant_records = [
                    docs_by_pmid[pmid]
                    for pmid in relevant_ids
                    if "bag_of_words_synonyms" in docs_by_pmid[pmid]
                ]

                negative_ids = list(set(docs_by_pmid.keys()) - set(relevant_ids))
                negative_records = [
                    docs_by_pmid[pmid]
                    for pmid in negative_ids
                    if "bag_of_words_synonyms" in docs_by_pmid[pmid]
                ]

                start_time = time.time()
                result = train_text_classifier(
                    model,
                    [r["bag_of_words_synonyms"] for r in relevant_records],
                    [r["bag_of_words_synonyms"] for r in negative_records],
                    min_f_occ=min_f_occ,
                )
                duration = time.time() - start_time

                record = {
                    "mesh_term": mesh,
                    "num_positive": len(relevant_records),
                    "num_negative": len(negative_records),
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "threshold": model._optimal_threshold
                    if model._optimal_threshold
                    else "",
                    "boolean_function_set1": result["boolean_function_set1"],
                    "boolean_function_set2": result["boolean_function_set2"],
                    "pretty_print": result["pretty_print"],
                    "time_seconds": duration,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()  # ensures progress is safely written
        print(f"✅ Completed training for {output_path}")


def plot_metrics_from_jsonl(path: str):
    """
    Reads a JSONL file and plots average recall, accuracy, and time_seconds
    vs. max_depth, using separate y-axes for each metric.
    Additionally shows min/max deviation for each metric.
    Only includes data if depths 3, 5, 10, 15, and 100 are all present.
    """

    # Load data
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                data.append(
                    {
                        "max_depth": obj.get("max_depth"),
                        "recall": obj.get("recall"),
                        "accuracy": obj.get("accuracy"),
                        "time_seconds": obj.get("time_seconds"),
                    }
                )
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    valid_depths = [3, 5, 10, 15, 100]  # Use a list for order
    df = df[df["max_depth"].isin(valid_depths)]

    grouped_mean = df.groupby("max_depth").mean(numeric_only=True)
    grouped_min = df.groupby("max_depth").min(numeric_only=True)
    grouped_max = df.groupby("max_depth").max(numeric_only=True)

    if set(grouped_mean.index) != set(valid_depths):
        print(f"Not all required max_depth values {valid_depths} are present.")
        return

    # Treat x-axis as categorical for equal spacing
    x_labels = valid_depths
    x_positions = range(len(x_labels))

    # Plot with three y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    color2 = "tab:orange"
    color3 = "tab:green"

    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Recall", color=color1)
    ax1.plot(
        x_positions,
        grouped_mean.loc[x_labels, "recall"],
        color=color1,
        marker="o",
        label="Recall (avg)",
    )
    ax1.fill_between(
        x_positions,
        grouped_min.loc[x_labels, "recall"],
        grouped_max.loc[x_labels, "recall"],
        color=color1,
        alpha=0.15,
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(
        x_positions,
        grouped_mean.loc[x_labels, "accuracy"],
        color=color2,
        marker="s",
        label="Accuracy (avg)",
    )
    ax2.fill_between(
        x_positions,
        grouped_min.loc[x_labels, "accuracy"],
        grouped_max.loc[x_labels, "accuracy"],
        color=color2,
        alpha=0.15,
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Time (s)", color=color3)
    ax3.plot(
        x_positions,
        grouped_mean.loc[x_labels, "time_seconds"],
        color=color3,
        marker="^",
        label="Time (avg)",
    )
    ax3.fill_between(
        x_positions,
        grouped_min.loc[x_labels, "time_seconds"],
        grouped_max.loc[x_labels, "time_seconds"],
        color=color3,
        alpha=0.15,
    )
    ax3.tick_params(axis="y", labelcolor=color3)

    # Set categorical x-axis
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels)

    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)
    ax3.set_ylim(0, None)

    plt.title(
        "Average Recall, Accuracy, and Time vs. max_depth (with min/max deviation)"
    )
    fig.tight_layout()
    plt.show()


def plot_metrics_vs_class_ratio(path: str):
    """
    Reads a JSONL file and plots recall and accuracy
    vs. the positive class ratio (num_positive / (num_positive + num_negative)),
    including min/max deviation.
    """

    # Load data
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "num_positive" in obj and "num_negative" in obj:
                    pos = obj["num_positive"]
                    neg = obj["num_negative"]
                    ratio = pos / (pos + neg) if (pos + neg) > 0 else None
                    if ratio is not None:
                        data.append(
                            {
                                "class_ratio": ratio,
                                "recall": obj.get("recall"),
                                "accuracy": obj.get("accuracy"),
                            }
                        )
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data with num_positive/num_negative found.")
        return

    # Group by class_ratio (round to avoid too many unique values if needed)
    df["class_ratio_rounded"] = df["class_ratio"].round(3)
    grouped_mean = df.groupby("class_ratio_rounded").mean(numeric_only=True)
    grouped_min = df.groupby("class_ratio_rounded").min(numeric_only=True)
    grouped_max = df.groupby("class_ratio_rounded").max(numeric_only=True)

    x_values = grouped_mean.index

    # Plot recall and accuracy on separate y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.set_xlabel("Positive Class Ratio")
    ax1.set_ylabel("Recall", color=color1)
    ax1.plot(
        x_values, grouped_mean["recall"], color=color1, marker="o", label="Recall (avg)"
    )
    ax1.fill_between(
        x_values, grouped_min["recall"], grouped_max["recall"], color=color1, alpha=0.15
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(
        x_values,
        grouped_mean["accuracy"],
        color=color2,
        marker="s",
        label="Accuracy (avg)",
    )
    ax2.fill_between(
        x_values,
        grouped_min["accuracy"],
        grouped_max["accuracy"],
        color=color2,
        alpha=0.15,
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    plt.title("Recall and Accuracy vs. Positive Class Ratio (with min/max deviation)")
    fig.tight_layout()
    plt.show()


def load_experiment_data(folder, model="GreedyORDecisionTree", filter_vars=None):
    """
    Load and aggregate JSONL experiment results by file.

    Args:
        folder (str): Path containing JSONL result files.
        model (str): Model name to filter filenames, e.g., "GreedyORDecisionTree".
        filter_vars (dict, optional): Example: {'n_docs': '50k'} to filter filenames.

    Returns:
        pd.DataFrame: Averaged metrics per file and associated hyperparameters.
    """
    # Pattern based on the model
    pattern = re.compile(
        rf"{re.escape(model)}\(max_depth=(\d+), min_samples_split=(\d+), "
        r"min_impurity_decrease_range=\[([\d\.]+), ([\d\.]+)\], "
        r"top_k_or_candidates=(\d+), class_weight=('[^']+'|\{[^}]+\})\)_"
        r"n_docs=([^_]+)_min_f_occ_=\{0: (\d+), 1: (\d+)\}"
    )

    records = []
    files = os.listdir(folder)

    for file in files:
        if not file.endswith(".jsonl"):
            continue

        # Optional filter for other parameters in filename
        if filter_vars and not all(f"{k}={v}" in file for k, v in filter_vars.items()):
            continue

        match = pattern.search(file)
        if not match:
            print(file)
            continue

        params = {
            "file": file,
            "max_depth": int(match.group(1)),
            "min_samples_split": int(match.group(2)),
            "min_impurity_decrease_start": float(match.group(3)),
            "min_impurity_decrease_end": float(match.group(4)),
            "top_k_or_candidates": int(match.group(5)),
            "class_weight": match.group(6),
            "n_docs": match.group(7),
            "min_f_occ_0": int(match.group(8)),
            "min_f_occ_1": int(match.group(9)),
        }

        file_records = []
        with open(os.path.join(folder, file), "r") as f:
            for line in f:
                data = json.loads(line)
                file_records.append(data)
                data["f1"] = (
                    2
                    * data["precision"]
                    * data["recall"]
                    / (data["precision"] + data["recall"])
                )
                pretty = data.get("pretty_print", "")
                data["leafs"] = pretty.count("class")
                data["ORs"] = pretty.count("OR")
                data["IFs"] = pretty.count("if")

        if not file_records:
            continue

        df_file = pd.DataFrame(file_records)
        mean_metrics = df_file.mean(numeric_only=True).to_dict()

        records.append({**params, **mean_metrics})

    if not records:
        print("No matching files or records found.")
        return pd.DataFrame()

    return pd.DataFrame(records)


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
        "min_f_occ_0",
        "min_f_occ_1",
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
        plt.show()

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
                [
                    (x if len(x) < 30 else x[:15] + "..." + x[-15:])
                    for x in re.findall(
                        r"(max_depth=\d+|min_samples_split=\d+|min_impurity_decrease_range=\[[\d\.]+, [\d\.]+\]|top_k_or_candidates=\d+|class_weight=[^{\n]+|\{[^}]+\}|n_docs=[^_]+|min_f_occ_=\{0: \d+, 1: \d+\})",
                        f,
                    )
                ]
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
    plt.show()

    return top_df


if __name__ == "__main__":
    df = load_experiment_data(
        folder="data/pubmed/statistics/classifier_learning",
        model="GreedyORDecisionTree",
        filter_vars={"n_docs": "50k"},
    )
    print(df)

    analyze_and_plot_best_files_from_df(
        df,
        top_n=10,
    )
    exit(0)

    # df = analyze_dataframe_results(df)
    # exit(0)
    models = [
        # SkopeRulesClassifier(
        #     n_estimators=10,
        #     precision_min=0.01,
        #     recall_min=0.05,
        # ),
        # DecisionTreeClassifier(
        #     max_depth=5,
        #     random_state=42,
        #     # min_samples_split=min_samples_split,
        #     class_weight="balanced",
        #     # min_samples_leaf=min_samples_leaf,
        # ),
        GreedyORDecisionTree(
            max_depth=4,
            min_samples_split=2,
            min_impurity_decrease_range=[0.01, 0.03],
            top_k_or_candidates=500,
            class_weight={1: 1, 0: 1},  # "balanced",
            verbose=True,
        )
    ]

    models = []
    for min_samples_split in [2, 5]:
        for min_impurity_d_start in [0.01, 0.1, 0.001]:
            for min_impurity_d_end in [0.03, 0.3, 0.003]:
                for top_k_or_candidates in [100, 500, 1000]:
                    for class_weight in ["balanced", {1: 1, 0: 1}, {1: 2, 0: 1}]:
                        models.append(
                            GreedyORDecisionTree(
                                max_depth=4,
                                min_samples_split=min_samples_split,
                                min_impurity_decrease_range=[
                                    min_impurity_d_start,
                                    min_impurity_d_end,
                                ],
                                top_k_or_candidates=top_k_or_candidates,
                                class_weight=class_weight,  # "balanced",
                                verbose=True,
                            )
                        )

    args = {
        "models": models,
        "baseline_folder": "./data/pubmed/baseline",
        "output_path": "data/pubmed/statistics/classifier_learning/",
        "skip_existing": True,
        "n_docs": 5_000_0,
        "min_f_occ": {0: 10, 1: 2},
        "mesh_terms": [
            "Endometriosis",
            "Rectal Neoplasms",
            "Fluorodeoxyglucose F18",
            "Cholelithiasis",
            "Antigens, Helminth",
            "Down Syndrome",
            "Antigens, Protozoan",
            "Urinary Tract Infections",
            "Chromosome Aberrations",
            "Streptococcal Infections",
            "Kidney Transplantation",
            "Cognition Disorders",
            "Alzheimer Disease",
            "Pregnancy",
        ],
    }
    train_all_mesh_terms_jsonl(**args)
    # plot_metrics_from_jsonl("data/pubmed/statistics/mesh_results_330k.jsonl")
    # plot_metrics_vs_class_ratio("data/pubmed/statistics/mesh_results_330k.jsonl")
