import pandas as pd
from app.dataset.utils import (
    get_qg_results,
    find_qg_results_file,
    get_dataset_details,
)
from app.config.config import CURRENT_BEST_RUN_FOLDER
from app.visualization.helper import escape_typst
from app.helper.helper import f_beta


def dataframe_to_best_worst_table(
    df: pd.DataFrame,
    output_path: str,
    dataset_details: dict,
    query_id_col: str = "query_id",
    precision_col: str = "pubmed_precision",
    recall_col: str = "pubmed_recall",
    query_col: str = "pubmed_query",
) -> None:
    """
    Generate a Typst table with best/worst query comparisons.

    Rows:
    - highest F50
    - lowest F50
    - highest Recall (precision as tie-breaker)
    - highest Precision (recall as tie-breaker)
    - average precision and recall

    Columns: type, title, query
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    working = df.copy()
    working = working[[query_id_col, precision_col, recall_col, query_col]].dropna(subset=[precision_col, recall_col])
    if working.empty:
        raise ValueError("No rows with precision/recall values found.")

    working["f50"] = working.apply(
        lambda row: f_beta(row[precision_col], row[recall_col], beta=50), axis=1
    )

    highest_f50 = working.sort_values(["f50", precision_col, recall_col], ascending=[False, False, False]).iloc[0]
    lowest_f50 = working.sort_values(["f50", precision_col, recall_col], ascending=[True, True, True]).iloc[0]
    highest_recall = working.sort_values([recall_col, precision_col], ascending=[False, False]).iloc[0]
    highest_precision = working.sort_values([precision_col, recall_col], ascending=[False, False]).iloc[0]

    avg_precision = working[precision_col].mean()
    avg_recall = working[recall_col].mean()
    eps = 1e-12
    precision_weight = 1.0 / max(avg_precision, eps)
    recall_weight = 1.0 / max(avg_recall, eps)
    working["avg_distance"] = (
        (precision_weight * (working[precision_col] - avg_precision)) ** 2
        + (recall_weight * (working[recall_col] - avg_recall)) ** 2
    ) ** 0.5
    closest_to_avg = working.sort_values(["avg_distance", precision_col, recall_col], ascending=[True, False, False]).iloc[0]

    def get_title(row):
        """Get title from dataset_details using query_id."""
        query_id = row[query_id_col]
        return dataset_details.get(query_id, {}).get("title", "")

    rows = [
        ("Highest F50", highest_f50[precision_col], highest_f50[recall_col], get_title(highest_f50), highest_f50[query_col]),
        ("Lowest F50", lowest_f50[precision_col], lowest_f50[recall_col], get_title(lowest_f50), lowest_f50[query_col]),
        ("Highest Recall", highest_recall[precision_col], highest_recall[recall_col], get_title(highest_recall), highest_recall[query_col]),
        ("Highest Precision", highest_precision[precision_col], highest_precision[recall_col], get_title(highest_precision), highest_precision[query_col]),
        ("Closest to Average", closest_to_avg[precision_col], closest_to_avg[recall_col], get_title(closest_to_avg), closest_to_avg[query_col]),
    ]

    if highest_f50[query_id_col] == highest_recall[query_id_col]:
        rows = [row for row in rows if row[0] != "Highest Recall"]

    typst_lines = []
    typst_lines.append('#import "../thesis/assets/assets.typ": *')
    typst_lines.append("#let best_worst_table() = [")
    typst_lines.append("#table(")
    typst_lines.append("  columns: (auto, 1fr, 2fr),")
    typst_lines.append("  table.header([Type], [Title], [Query]),")

    for name, precision, recall, title, query in rows:
        precision_text = f"{precision:.4f}" if pd.notna(precision) else ""
        recall_text = f"{recall:.4f}" if pd.notna(recall) else ""
        title_text = escape_typst(title) if title else ""
        query_text = escape_typst(query) if query else ""
        type_text = f"*{escape_typst(name)}*\\ *Precision:* {precision_text}\\ *Recall:* {recall_text}"
        typst_lines.append(
            f"  [{type_text}], [{title_text}], [{query_text}],"
        )

    typst_lines.append(")")
    typst_lines.append("]")

    with open(output_path, "w") as f:
        f.write("\n".join(typst_lines))

if __name__ == "__main__":
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type="cosine", betas_key="50"
    )
    dataframe = get_qg_results(path, min_positive_threshold=50)

    dataset_details = get_dataset_details()
    
    dataframe_to_best_worst_table(
        df=dataframe,
        output_path="../master-thesis-writing/writing/tables/best_worst_table.typ",
        dataset_details=dataset_details,
        query_id_col="query_id",
        precision_col="pubmed_precision",
        recall_col="pubmed_recall",
        query_col="pubmed_query",
    ) 

    