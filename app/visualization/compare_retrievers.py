import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def pretty_file_name(file_name):
    """
    Convert raw retriever JSON filenames into readable names for plots.
    """
    if file_name.endswith(".json"):
        file_name = file_name[:-5]

    # Replace document types
    file_name = file_name.replace("title_abstract", "(Title, Abstract)")
    file_name = file_name.replace("title", "(Title)")
    file_name = file_name.replace("abstract", "(Abstract)")

    # Remove dataset size info
    file_name = file_name.replace("docs=503679", "")

    # Map model codes to readable names
    model_map = {
        "biobert-nli": "BioBERT-NLI",
        "biolinkbert": "BioLinkBERT",
        "MiniLM-512": "MiniLM-512",
        "pubmedbert": "PubMedBERT",
        "roberta": "RoBERTa",
        "MedCPT": "MedCPT",
        "bm25": "BM25",
    }

    for key, val in model_map.items():
        if key in file_name:
            file_name = file_name.replace(key, val)

    # Remove extra underscores and whitespace
    file_name = file_name.replace("_", " ")
    file_name = file_name.replace("__", "_").strip("_ ")

    return file_name


def compare_dense_retrievers(folder_path, metrics, save_path):
    """
    Compare dense retrievers from JSON files in a folder.
    """
    folder_path = Path(folder_path)
    save_path = Path(save_path)

    all_data = []
    file_names = []

    for file_path in folder_path.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            filtered_data = {}
            for metric in metrics:
                value = data.get(metric, None)
                filtered_data[metric] = (
                    value if isinstance(value, (int, float)) else float("nan")
                )
            all_data.append(filtered_data)
            file_names.append(pretty_file_name(file_path.stem))
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    if not all_data:
        print("No valid JSON files found to plot.")
        return

    df = pd.DataFrame(all_data, index=file_names)

    # Sort by retriever name and document type: title -> abstract -> title+abstract
    def sort_key(name):
        if "(Title)" in name:
            doc_order = 0
        elif "(Abstract)" in name:
            doc_order = 1
        elif "(Title, Abstract)" in name:
            doc_order = 2
        else:
            doc_order = 3
        # Sort by model name first, then document type
        model_name = name.split("(")[0].strip()
        return (model_name.lower(), doc_order)

    df = df.reindex(sorted(df.index, key=sort_key))

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Define colors per metric
    colors = {
        "precision@10": "#d73027",  # Red
        "precision@100": "#fd8383",  # Lighter red
        "recall@10": "#4575b4",  # Blue
        "recall@100": "#91bfdb",  # Lighter blue
        "map": "#e41aef",
        "mrr@100": "#5bb229",
    }

    # Match colors to metrics in df
    metric_colors = [colors.get(m, "#999999") for m in df.columns]

    sns.set(style="whitegrid")
    plt.figure(figsize=(max(10, len(df) * 0.6), 6))

    df.plot(kind="bar", figsize=(max(10, len(df) * 0.6), 6), color=metric_colors)
    plt.title("Dense Retriever Comparison")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="Metrics")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    data_folder = Path(
        "/data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/reports/title_and_abstract"
    )
    metrics_to_compare = [
        "map",
        "mrr@100",
        "recall@10",
        "recall@100",
        "precision@10",
        "precision@100",
    ]
    save_folder = Path("data/statistics/images/retriever_comparison")
    save_folder.mkdir(parents=True, exist_ok=True)
    save_file = save_folder / "retriever_comparison.png"

    compare_dense_retrievers(data_folder, metrics_to_compare, save_file)
