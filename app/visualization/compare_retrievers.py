import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from app.config.config import apply_matplotlib_style, COLORS

apply_matplotlib_style()


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


def compare_dense_retrievers(folder_path, metrics, save_path, models=None):
    """
    Compare dense retrievers from JSON files in a folder.
    """
    folder_path = Path(folder_path)
    save_path = Path(save_path)

    all_data = []
    file_names = []

    for file_path in folder_path.glob("*.json"):
        if models and not any(model in file_path.stem for model in models):
            continue
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

    # Insert NaN spacer rows between model groups for visual separation
    model_of = lambda name: name.split("(")[0].strip()
    new_rows, new_index, prev_model = [], [], None
    for name in df.index:
        model = model_of(name)
        if prev_model is not None and model != prev_model:
            new_rows.append([float("nan")] * len(df.columns))
            new_index.append("")
        new_rows.append(df.loc[name].tolist())
        new_index.append(name)
        prev_model = model
    df = pd.DataFrame(new_rows, index=new_index, columns=df.columns)

    # Define colors per metric using centralized config
    colors = {
        "precision@100": COLORS["precision"],
        "precision@1000": COLORS["precision_light"],
        "recall@100": COLORS["recall"],
        "recall@1000": COLORS["recall_light"],
        "map": COLORS["map"],
        "mrr@100": COLORS["mrr"],
    }

    # Match colors to metrics in df
    metric_colors = [colors.get(m, "#999999") for m in df.columns]

    # Create figure with explicit settings, then use pandas plot on the axes
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.6), 6))
    
    df.plot(kind="bar", ax=ax, color=metric_colors, width=0.9)
    ax.set_title("Dense Retriever Comparison")
    ax.set_ylabel("Metric Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    # Update legend labels to readable names
    legend_label_map = {
        "map": "MAP",
        "mrr@100": "MRR@100",
        "precision@100": "Precision@100",
        "precision@1000": "Precision@1000",
        "recall@100": "Recall@100",
        "recall@1000": "Recall@1000",
    }
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [legend_label_map.get(label, label) for label in labels]
    ax.legend(handles, new_labels, ncol=len(new_labels)+1)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    data_folder = Path(
        "data/reports/title_and_abstract"
    )
    metrics_to_compare = [
        "map",
        # "mrr@100",
        # "recall@10",
        "recall@100",
        "recall@1000",
        # "precision@10",
        "precision@100",
        "precision@1000",
    ]
    # save_folder = Path("data/statistics/images/retriever_comparison")
    save_folder = Path("../master-thesis-writing/writing/thesis/images/graphs")
    save_folder.mkdir(parents=True, exist_ok=True)
    save_file = save_folder / "retriever_comparison.png"
    # models = ["biobert-nli", "biolinkbert", "MiniLM-512", "pubmedbert", "roberta", "MedCPT", "bm25"]
    models = ["bm25", "MedCPT", "MiniLM-512", "pubmedbert"]
    compare_dense_retrievers(data_folder, metrics_to_compare, save_file, models=models)
