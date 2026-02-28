import pandas as pd
from app.dataset.utils import (
    get_qg_results,
    get_dataset_details,
)
from app.config.config import (
    COLORS,
    CURRENT_BEST_RUN_FOLDER,
    CURRENT_N_TRIALS_FOLDER,
    HIGHLIGHT_LIGHTNESS,
)
from app.visualization.helper import (
    escape_typst,
    mark_outer_operators,
    split_query_into_words,
    highlight_query_words,
)
from app.helper.helper import f_beta


def dataframe_to_best_worst_table(
    df: pd.DataFrame,
    output_path: str,
    dataset_details: dict,
    query_id_col: str = "query_id",
    precision_col: str = "pubmed_precision",
    recall_col: str = "pubmed_recall",
    query_col: str = "pubmed_query",
    table_name: str = "best_worst_table",
    highlight_unique_terms: bool = False,
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
    working = working[[query_id_col, precision_col, recall_col, query_col]].dropna(
        subset=[precision_col, recall_col]
    )
    if working.empty:
        raise ValueError("No rows with precision/recall values found.")

    working["f50"] = working.apply(
        lambda row: f_beta(row[precision_col], row[recall_col], beta=50), axis=1
    )

    highest_f50 = working.sort_values(
        ["f50", precision_col, recall_col], ascending=[False, False, False]
    ).iloc[0]
    lowest_f50 = working.sort_values(
        ["f50", precision_col, recall_col], ascending=[True, True, True]
    ).iloc[0]
    highest_recall = working.sort_values(
        [recall_col, precision_col], ascending=[False, False]
    ).iloc[0]
    highest_precision = working.sort_values(
        [precision_col, recall_col], ascending=[False, False]
    ).iloc[0]

    working["query_word_count"] = working[query_col].apply(
        lambda query: len(split_query_into_words(query))
    )
    shortest_query = working.sort_values(
        ["query_word_count", precision_col, recall_col], ascending=[True, False, False]
    ).iloc[0]

    avg_precision = working[precision_col].mean()
    avg_recall = working[recall_col].mean()
    eps = 1e-12
    precision_weight = 1.0 / max(avg_precision, eps)
    recall_weight = 1.0 / max(avg_recall, eps)
    working["avg_distance"] = (
        (precision_weight * (working[precision_col] - avg_precision)) ** 2
        + (recall_weight * (working[recall_col] - avg_recall)) ** 2
    ) ** 0.5
    closest_to_avg = working.sort_values(
        ["avg_distance", precision_col, recall_col], ascending=[True, False, False]
    ).iloc[0]

    def get_title(row):
        """Get title from dataset_details using query_id."""
        query_id = row[query_id_col]
        return dataset_details.get(query_id, {}).get("title", "")

    rows = [
        (
            "Highest F50",
            highest_f50[precision_col],
            highest_f50[recall_col],
            get_title(highest_f50),
            highest_f50[query_col],
        ),
        (
            "Lowest F50",
            lowest_f50[precision_col],
            lowest_f50[recall_col],
            get_title(lowest_f50),
            lowest_f50[query_col],
        ),
        (
            "Highest Recall",
            highest_recall[precision_col],
            highest_recall[recall_col],
            get_title(highest_recall),
            highest_recall[query_col],
        ),
        (
            "Highest Precision",
            highest_precision[precision_col],
            highest_precision[recall_col],
            get_title(highest_precision),
            highest_precision[query_col],
        ),
        (
            "Shortest Query",
            shortest_query[precision_col],
            shortest_query[recall_col],
            get_title(shortest_query),
            shortest_query[query_col],
        ),
        (
            "Closest to Average",
            closest_to_avg[precision_col],
            closest_to_avg[recall_col],
            get_title(closest_to_avg),
            closest_to_avg[query_col],
        ),
    ]

    if highest_f50[query_id_col] == highest_recall[query_id_col]:
        rows = [row for row in rows if row[0] != "Highest Recall"]

    highest_f50_words = {
        word.lower() for word in split_query_into_words(highest_f50[query_col])
    }
    lowest_f50_words = {
        word.lower() for word in split_query_into_words(lowest_f50[query_col])
    }
    unique_highest_f50_words = highest_f50_words - lowest_f50_words

    typst_lines = []
    typst_lines.append('#import "../thesis/assets/assets.typ": *')
    typst_lines.append(f"#let {table_name}() = [")
    typst_lines.append("#table(")
    typst_lines.append("  columns: (auto, 1fr, 2fr),")
    typst_lines.append("  table.header([Type], [Title], [Query]),")

    for name, precision, recall, title, query_text in rows:
        
        precision_text = f"{precision:.4f}" if pd.notna(precision) else ""
        recall_text = f"{recall:.4f}" if pd.notna(recall) else ""
        title_text = escape_typst(title) if title else ""
        if highlight_unique_terms and name == "Highest F50":
            query_text = escape_typst(query_text)
            query_text = (
                highlight_query_words(
                    query_text,
                    unique_highest_f50_words,
                    f'rgb("{COLORS["positive_light"]}")',
                    lightness=HIGHLIGHT_LIGHTNESS,
                )
            )
        elif highlight_unique_terms:
            row_words = (
                {word.lower() for word in split_query_into_words(query_text)}
            )
            unique_row_words = row_words - highest_f50_words
            query_text = escape_typst(query_text)
            query_text = (
                highlight_query_words(
                    query_text,
                    unique_row_words,
                    f'rgb("{COLORS["negative_light"]}")',
                    lightness=HIGHLIGHT_LIGHTNESS,
                )
            )
        else:
            query_text = escape_typst(query_text)
        
        query_text = mark_outer_operators(query_text, operator_types=["OR"])
        type_text = f"*{escape_typst(name)}*\\ *Precision:* {precision_text}\\ *Recall:* {recall_text}"
        typst_lines.append(f"  [{type_text}], [{title_text}], [{query_text}],")

    typst_lines.append(")")
    typst_lines.append("]")

    with open(output_path, "w") as f:
        f.write("\n".join(typst_lines))


def filter_to_query_with_max_precision_spread(
    df: pd.DataFrame,
    query_id_col: str,
    precision_col: str,
    recall_col: str,
) -> pd.DataFrame:
    """Keep only rows from the query_id with the largest max-min precision spread."""
    if df.empty:
        return df

    working = df[[query_id_col, precision_col, recall_col]].dropna(
        subset=[precision_col, recall_col]
    )
    if working.empty:
        return df

    spreads = (
        working.groupby(query_id_col)[precision_col]
        .agg(["min", "max"])
        .assign(spread=lambda group: group["max"] - group["min"])
    )
    if spreads.empty:
        return df

    target_query_id = spreads.sort_values(
        ["spread", "max"], ascending=[False, False]
    ).index[0]
    return df[df[query_id_col] == target_query_id]


if __name__ == "__main__":
    dataframe = get_qg_results(CURRENT_BEST_RUN_FOLDER, min_positive_threshold=50, top_k_types=["cosine"], restrict_betas=["50"])

    dataset_details = get_dataset_details()

    dataframe_to_best_worst_table(
        df=dataframe,
        output_path="../master-thesis-writing/writing/tables/best_worst_table.typ",
        dataset_details=dataset_details,
        query_id_col="query_id",
        precision_col="pubmed_precision",
        recall_col="pubmed_recall",
        query_col="pubmed_query",
        table_name="best_worst_table",
        highlight_unique_terms=False,
    )

    # same query
    dataframe = get_qg_results(CURRENT_N_TRIALS_FOLDER, min_positive_threshold=50)

    dataframe = filter_to_query_with_max_precision_spread(
        df=dataframe,
        query_id_col="query_id",
        precision_col="pubmed_precision",
        recall_col="pubmed_recall",
    )

    dataset_details = get_dataset_details()

    dataframe_to_best_worst_table(
        df=dataframe,
        output_path="../master-thesis-writing/writing/tables/best_worst_table_same_query.typ",
        dataset_details=dataset_details,
        query_id_col="query_id",
        precision_col="pubmed_precision",
        recall_col="pubmed_recall",
        query_col="pubmed_query",
        table_name="best_worst_table_same_query",
        highlight_unique_terms=True,
    )
