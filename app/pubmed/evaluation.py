import json
import time
from pathlib import Path
from tqdm import tqdm
from Bio import Entrez
from app.pubmed.retrieval import classify_by_mesh
from app.tree_learning.logical_query_generation import train_text_classifier
import pandas as pd
import matplotlib.pyplot as plt

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


def train_all_mesh_terms_jsonl(baseline_folder: str, output_path: str = "mesh_results.jsonl", max_mesh_terms=None, skip_existing=False, n_docs=1_000_000, max_depths=[3,5,7,10,15,100]):
    """
    Train text classifiers for all MeSH terms and save results incrementally as JSONL.
    Skips terms already in the output file.
    """
    output_path = Path(output_path)
    if skip_existing:
        completed_mesh = load_completed_mesh_terms(output_path)
        print(f"Already computed {len(completed_mesh)} MeSH terms, skipping those...")

    docs_by_pmid, pmids_by_mesh = classify_by_mesh(baseline_folder, n_docs)
    mesh_terms = list(pmids_by_mesh.keys())
    if max_mesh_terms:
        mesh_terms = mesh_terms[:max_mesh_terms]

    with output_path.open("a", encoding="utf-8") as out_f:
        for mesh in tqdm(mesh_terms, desc="Training classifiers per MeSH term"):
            if skip_existing and mesh in completed_mesh:
                continue

            relevant_ids = pmids_by_mesh[mesh]
            relevant_records = [docs_by_pmid[pmid] for pmid in relevant_ids if "bag_of_words" in docs_by_pmid[pmid]]

            negative_ids = list(set(docs_by_pmid.keys()) - set(relevant_ids))
            negative_records = [docs_by_pmid[pmid] for pmid in negative_ids if "bag_of_words" in docs_by_pmid[pmid]]

            start_time = time.time()
            for max_depth in max_depths:
                result = train_text_classifier(
                    [r["bag_of_words"] for r in relevant_records],
                    [r["bag_of_words"] for r in negative_records],
                    max_depth = max_depth,
                )
                duration = time.time() - start_time

                record = {
                    "mesh_term": mesh,
                    "max_depth": max_depth,
                    "num_positive": len(relevant_records),
                    "num_negative": len(negative_records),
                    "accuracy": result["accuracy"],
                    "recall": result["recall"],
                    "boolean_function_set1": result["boolean_function_set1"],
                    "boolean_function_set2": result["boolean_function_set2"],
                    "decision_tree": result["decision_tree"],
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
                data.append({
                    "max_depth": obj.get("max_depth"),
                    "recall": obj.get("recall"),
                    "accuracy": obj.get("accuracy"),
                    "time_seconds": obj.get("time_seconds")
                })
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
    ax1.plot(x_positions, grouped_mean.loc[x_labels, "recall"], color=color1, marker="o", label="Recall (avg)")
    ax1.fill_between(x_positions,
                     grouped_min.loc[x_labels, "recall"],
                     grouped_max.loc[x_labels, "recall"],
                     color=color1, alpha=0.15)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(x_positions, grouped_mean.loc[x_labels, "accuracy"], color=color2, marker="s", label="Accuracy (avg)")
    ax2.fill_between(x_positions,
                     grouped_min.loc[x_labels, "accuracy"],
                     grouped_max.loc[x_labels, "accuracy"],
                     color=color2, alpha=0.15)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("Time (s)", color=color3)
    ax3.plot(x_positions, grouped_mean.loc[x_labels, "time_seconds"], color=color3, marker="^", label="Time (avg)")
    ax3.fill_between(x_positions,
                     grouped_min.loc[x_labels, "time_seconds"],
                     grouped_max.loc[x_labels, "time_seconds"],
                     color=color3, alpha=0.15)
    ax3.tick_params(axis="y", labelcolor=color3)

    # Set categorical x-axis
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels)

    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)
    ax3.set_ylim(0, None)

    plt.title("Average Recall, Accuracy, and Time vs. max_depth (with min/max deviation)")
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
                        data.append({
                            "class_ratio": ratio,
                            "recall": obj.get("recall"),
                            "accuracy": obj.get("accuracy")
                        })
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
    ax1.plot(x_values, grouped_mean["recall"], color=color1, marker="o", label="Recall (avg)")
    ax1.fill_between(x_values,
                     grouped_min["recall"],
                     grouped_max["recall"],
                     color=color1, alpha=0.15)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color2)
    ax2.plot(x_values, grouped_mean["accuracy"], color=color2, marker="s", label="Accuracy (avg)")
    ax2.fill_between(x_values,
                     grouped_min["accuracy"],
                     grouped_max["accuracy"],
                     color=color2, alpha=0.15)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    plt.title("Recall and Accuracy vs. Positive Class Ratio (with min/max deviation)")
    fig.tight_layout()
    plt.show()
if __name__ == "__main__":
    # train_all_mesh_terms_jsonl("./data/pubmed/baseline", output_path="data/pubmed/statistics/mesh_results_1000k.jsonl", skip_existing=True)
    plot_metrics_from_jsonl("data/pubmed/statistics/mesh_results_330k.jsonl")
    plot_metrics_vs_class_ratio("data/pubmed/statistics/mesh_results_330k.jsonl")