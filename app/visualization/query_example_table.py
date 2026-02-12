import pandas as pd
from app.dataset.utils import (
    get_qg_results,
    find_qg_results_file,
    get_dataset_details,
    get_paper_query_examples,
    review_id_to_dataset,
)
from app.config.config import CURRENT_BEST_RUN_FOLDER
from app.pubmed.retrieval import evaluate_query
from app.visualization.helper import escape_typst


def dataframe_to_typst_query_table(
    df: pd.DataFrame, output_path: str, review_ids: list = None
) -> None:
    """
    Convert a dataframe with query examples to a Typst table.

    Expected dataframe columns:
    - review_id: query identifier
    - title: query title
    - my_query: generated query string
    - my_p: precision for my query
    - my_r: recall for my query
    - paper_query: baseline/paper query string
    - paper: name of the baseline paper
    - paper_p: precision for paper query
    - paper_r: recall for paper query

    Output format:
    - Row 1: Headers (query_title, my query, baseline query)
    - Rows with same review_id: query_title, my_query cells span across rows
    - Each cell contains query string followed by "precision: p, recall: r"
    - Baseline query cells include paper name before query string
    """

    # Sort by review_id to group same queries together
    df = df.sort_values("review_id").reset_index(drop=True)

    # Build the table content
    typst_lines = []
    typst_lines.append('#import "../thesis/assets/assets.typ": *')
    typst_lines.append("#let query_example_table() = [")
    typst_lines.append("#table(")
    typst_lines.append("  columns: 3,")

    # Header row
    header_cells = ["[Title]", "[#algo-name-short]", "[Baseline]"]
    typst_lines.append(f"  table.header({', '.join(header_cells)}),")

    # Process data rows
    current_review_id = None
    row_buffer = []

    for idx, row in df.iterrows():
        review_id = row["review_id"]
        if review_ids is not None and review_id not in review_ids:
            continue

        # Check if we need to write the buffered row(s) and start a new one
        if current_review_id is not None and review_id != current_review_id:
            # Write the buffered row(s)
            _write_query_table_row(typst_lines, current_review_id, row_buffer)
            row_buffer = []

        current_review_id = review_id
        row_buffer.append(row)

    # Write the last buffered row(s)
    if row_buffer:
        _write_query_table_row(typst_lines, current_review_id, row_buffer)

    typst_lines.append(")")
    typst_lines.append("]")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(typst_lines))


def _write_query_table_row(typst_lines: list, review_id: str, rows_data: list) -> None:
    """
    Helper function to write a single logical row (which may span multiple physical rows
    for the baseline query column).

    For rows with the same review_id:
    - query_title, my_query are combined with rowspan
    - baseline query rows are repeated separately
    """

    num_rows = len(rows_data)
    first_row = rows_data[0]

    title = first_row["title"]
    escaped_title = escape_typst(title)
    for i, s in enumerate(escaped_title):
        if s == " " and i > len(escaped_title) // 2:
            escaped_title = escaped_title[:i-1] + "\\" + escaped_title[i :]
            break
    title_cell = f"rotate(-90deg, reflow:true)[{escaped_title}]"

    # Format my query cell
    my_query = first_row["my_query"]
    my_p = first_row["my_p"]
    my_r = first_row["my_r"]
    my_query_text = f"*#algo-name-short\-F50\-cosine*\\ (*Precision:* {my_p:.4f}, *Recall:* {my_r:.4f})\\ {escape_typst(my_query)}"
    my_query_cell = f"[{my_query_text}]"

    # If multiple rows (multiple baselines), add rowspan to first two columns
    if num_rows > 1:
        title_cell = f"table.cell(rowspan: {num_rows}, {title_cell})"
        my_query_cell = f"table.cell(rowspan: {num_rows}, {my_query_cell})"

    # Add the first baseline query row
    paper = f"*{rows_data[0]['approach']}* @{rows_data[0]['paper']}"
    paper_query = rows_data[0]["paper_query"]
    paper_p = rows_data[0]["paper_p"]
    paper_r = rows_data[0]["paper_r"]
    paper_query_text = f"{paper}\\ (*Precision:* {paper_p:.4f}, *Recall:* {paper_r:.4f})\\ {escape_typst(paper_query)}"
    paper_query_cell = f"[{paper_query_text}]"

    typst_lines.append(f"  {title_cell}, {my_query_cell}, {paper_query_cell},")

    # Add remaining baseline query rows
    for row_data in rows_data[1:]:
        paper = f"*{row_data['approach']}* @{row_data['paper']}"
        paper_query = row_data["paper_query"]
        paper_p = row_data["paper_p"]
        paper_r = row_data["paper_r"]
        paper_query_text = f"{paper}\\ (*Precision:* {paper_p:.4f}, *Recall:* {paper_r:.4f})\\ {escape_typst(paper_query)}"
        paper_query_cell = f"[{paper_query_text}]"

        typst_lines.append(f"{paper_query_cell},")

if __name__ == "__main__":
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type="cosine", betas_key="50"
    )
    dataframe = get_qg_results(path, min_positive_threshold=None)

    dataset_details = get_dataset_details()

    paper_query_examples = get_paper_query_examples()

    rows = []
    for paper, data in paper_query_examples.items():
        print(f"Paper: {data['title']}")
        for example in data["examples"]:
            review_id = example.get("query_id")

            if (
                not review_id
                or "usable_for_stats" in example
                and not example["usable_for_stats"]
            ):
                print(
                    "Skipping example due to missing query_id or marked as not usable for stats."
                )
                continue
            positives = set(dataset_details[review_id]["positives"])
            dataset, _, end_year = review_id_to_dataset(review_id)
            precision, recall, retrieved_count, TP = evaluate_query(
                example["result"],
                positives,
                end_year=end_year,
            )

            # compare to manual
            # manual_example = None
            # for me in paper_query_examples["Manual"]["examples"]:
            #     if me["query_id"] == example["query_id"]:
            #         manual_example = me
            #         break
            # manual_precision, manual_recall, manual_retrieved_count, TP = None, None, None, None
            # if manual_example is not None:
            #     manual_precision, manual_recall, manual_retrieved_count, TP = evaluate_query(
            #         manual_example["result"],
            #         positives,
            #         end_year=end_year,
            #     )

            # comapre to mine
            row = dataframe[dataframe["query_id"] == example["query_id"]]
            pubmed_precision = row["pubmed_precision"].values[0]
            pubmed_recall = row["pubmed_recall"].values[0]
            pubmed_query = row["pubmed_query"].values[0]

            assert dataset_details[review_id]["title"] == example["query"]
            rows.append(
                {
                    "paper": paper,
                    "approach": data["name"],
                    "paper_p": precision,
                    "paper_r": recall,
                    "paper_stated_p": example.get("precision"),
                    "paper_stated_r": example.get("recall"),
                    "paper_stated_manual_p": example.get("manual_precision"),
                    "paper_stated_manual_r": example.get("manual_recall"),
                    "my_p": pubmed_precision,
                    "my_r": pubmed_recall,
                    # "manual_p": manual_precision,
                    # "manual_r": manual_recall,
                    "title": dataset_details[review_id]["title"],
                    "review_id": example["query_id"],
                    "paper_query": example["result"],
                    "my_query": pubmed_query,
                    # "manual_query": manual_example["result"] if manual_example is not None else None,
                }
            )

    if rows:
        table = pd.DataFrame(rows)
        display_cols = [
            "paper",
            "review_id",
            # "title",
            "paper_p",
            "paper_r",
            "paper_stated_p",
            "paper_stated_r",
            "my_p",
            "my_r",
            # "manual_p",
            # "manual_r",
            # "paper_stated_manual_p",
            # "paper_stated_manual_r",
        ]
        print("\nSummary table (queries omitted):")

        # print the table fully no matter if its too big
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(table[display_cols])

        # Generate Typst table
        output_path = "../master-thesis-writing/writing/tables/query_example_table.typ"

        # i only want the following rows from table: CD007394: chatgpt and semantic, CD009579: manual and objective approach
        # remoeve all toher rows fromt he table
        table = table[
            (
                table["review_id"].isin(["CD007394"])
                & table["approach"].isin(["#chatgpt-approach", "#semantic-approach"])
            )
            | (
                table["review_id"].isin(["CD009579"])
                & table["approach"].isin(["#manual-approach", "#objective-approach"])
            )
        ]

        dataframe_to_typst_query_table(table, output_path)
        print(f"\nTypst table written to: {output_path}")
