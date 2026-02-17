import json
import os
import pandas as pd
from app.dataset.utils import (
    get_qg_results,
    find_qg_results_file,
    get_dataset_details,
    get_paper_query_examples,
    review_id_to_dataset,
)
from app.config.config import COLORS, CURRENT_BEST_RUN_FOLDER
from app.pubmed.retrieval import evaluate_query
from app.visualization.helper import escape_typst, mark_outer_operators

# Map (review_id, approach) to the replacement JSON file
REPLACEMENT_FILES = {
    ("CD007394", "#chatgpt-approach"): "data/examples/generated_chatgpt_CD007394.json",
    ("CD007394", "#semantic-approach"): "data/examples/generated_semantic_CD007394.json",
    ("CD009579", "#manual-approach"): "data/examples/generated_manual_CD009579.json",
    ("CD009579", "#objective-approach"): "data/examples/generated_objective_CD009579.json",
}

def load_replacement_pairs(json_path: str, k: int = 2) -> list:
    """Load top k replacement pairs per direction from a JSON file.

    Returns list of dicts with:
    - my_term: term from my query (query1)
    - paper_term: term from paper query (query2)
    - direction: 'improve_mine' or 'improve_paper'
    """
    with open(json_path) as f:
        data = json.load(f)

    pairs = []

    # replacements1: keys = my_query terms, values = [(paper_term, score), ...]
    # Replacing my_term with paper_term improves my_query
    r1_cand = []
    for my_term, replacements in data.get("replacements1", {}).items():
        r1_cand += [(my_term, r) for r in replacements]
    sorted_r1_cand = sorted(r1_cand, key=lambda x: x[1][1], reverse=True)  # sort by score
    pairs += [{
        "my_term": my_term,
        "paper_term": replacement[0],
        "direction": "improve_mine",
    } for my_term, replacement in sorted_r1_cand[:k]]

    # replacements2: keys = paper_query terms, values = [(my_term, score), ...]
    # Replacing paper_term with my_term improves paper_query
    r2_cand = []
    for paper_term, replacements in data.get("replacements2", {}).items():
        r2_cand += [(paper_term, r) for r in replacements]
    sorted_r2_cand = sorted(r2_cand, key=lambda x: x[1][1], reverse=True)  # sort by score
    pairs += [{
        "my_term": replacement[0],
        "paper_term": paper_term,
        "direction": "improve_paper",
    } for paper_term, replacement in sorted_r2_cand[:k]]

    return pairs


def load_all_replacement_data(k: int = 2) -> dict:
    """Load all replacement pairs with colors assigned.

    Returns dict mapping (review_id, approach) -> list of pair dicts (with 'color' added).
    """
    replacement_data = {}
    color_idx = 0
    target_color = {}
    for key, json_path in REPLACEMENT_FILES.items():
        pairs = load_replacement_pairs(json_path, k=k)
        for pair in pairs:
            if pair["direction"] == "improve_mine":
                target = pair["my_term"]
            else:                
                target = pair["paper_term"]
            if target in target_color:
                pair["color"] = target_color[target]
            else:
                pair["color"] = f"rgb(\"{COLORS['category'][color_idx % len(COLORS['category'])]}\")"
                target_color[target] = pair["color"]
                color_idx += 1
        replacement_data[key] = pairs
    return replacement_data


def mark_query_terms(query_text: str, markings: list) -> str:
    """Apply underline/highlight markings to terms in a query string for Typst output.

    markings: list of (term, format_type, color) where:
    - term: exact substring to find in query_text (raw, pre-escape)
    - format_type: 'underline' (good replacement) or 'highlight' (replaceable term)
    - color: Typst color string

    A term is only replaced when bounded on both sides by a boolean operator
    ( OR , AND , NOT ), a parenthesis, or start/end of string.

    Returns Typst-formatted string with marked terms and the rest escaped.
    """
    import re

    query_text = escape_typst(query_text)
    if not markings:
        return query_text

    # Sort by term length (longest first) to avoid partial substring matches
    markings = sorted(markings, key=lambda x: len(x[0]), reverse=True)

    for term, fmt, color in markings:
        escaped_term = escape_typst(term)
        regex_term = re.escape(escaped_term)
        if fmt == "underline":
            replacement = f"#underline(stroke: 1pt + {color})[{escaped_term}]"
        else:
            replacement = f"#highlight(fill: {color}.lighten(70%))[{escaped_term}]"

        # Match term bounded by operators / parens / string boundaries on both sides
        pattern = (
            r'((?:^|\(|\[| (?:OR|AND|NOT) ))'   # preceding: start, "(", or " OP "
            + regex_term
            + r'(?= (?:OR|AND|NOT) |\)|\]|$)'    # following: " OP ", ")", or end
        )
        query_text = re.sub(pattern, lambda m: m.group(1) + replacement, query_text)

    return query_text


def dataframe_to_typst_query_table(
    df: pd.DataFrame, output_path: str, review_ids: list = None, replacement_data: dict = None
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
    typst_lines.append("  columns: 2,")

    # Header row
    header_cells = ["[#algo-name-short]", "[Baseline]"]
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
            _write_query_table_row(typst_lines, current_review_id, row_buffer, replacement_data)
            row_buffer = []

        current_review_id = review_id
        row_buffer.append(row)

    # Write the last buffered row(s)
    if row_buffer:
        # seperator = f"  table.cell(colspan: {2}, inset: (top: 5pt, bottom: 5pt))[],\n"
        # typst_lines.append(seperator)  # add a row separator
        _write_query_table_row(typst_lines, current_review_id, row_buffer, replacement_data)

    typst_lines.append(")")
    typst_lines.append("]")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(typst_lines))


def _write_query_table_row(typst_lines: list, review_id: str, rows_data: list, replacement_data: dict = None) -> None:
    """
    Helper function to write a single logical row (which may span multiple physical rows
    for the baseline query column).

    For rows with the same review_id:
    - query_title, my_query are combined with rowspan
    - baseline query rows are repeated separately
    - replacement word pairs are marked with colored underline/highlight
    """

    num_rows = len(rows_data)
    first_row = rows_data[0]

    title = first_row["title"]
    escaped_title = escape_typst(title)
    # for i, s in enumerate(escaped_title):
    #     if s == " " and i > len(escaped_title) // 2:
    #         escaped_title = escaped_title[:i-1] + "\\" + escaped_title[i :]
    #         break
    title_cell = f"[*{escaped_title}*]"

    # Build my_query markings from ALL baselines (since my_query cell spans all rows)
    my_markings = []
    if replacement_data:
        for row_data in rows_data:
            approach = row_data["approach"]
            pairs = replacement_data.get((review_id, approach), [])
            for pair in pairs:
                if pair["direction"] == "improve_mine":
                    # my_term is replaceable -> underline it
                    my_markings.append((pair["my_term"], "underline", pair["color"]))
                else:  # improve_paper
                    # my_term is the good replacement -> highlight it
                    my_markings.append((pair["my_term"], "highlight", pair["color"]))

    # Format my query cell
    my_query = first_row["my_query"]
    my_p = first_row["my_p"]
    my_r = first_row["my_r"]
    my_query_marked = mark_outer_operators(mark_query_terms(my_query, my_markings), ['OR'])
    my_query_text = f"*#algo-name-short\-F50\-cosine*\\ *Precision:* {my_p:.4f} *Recall:* {my_r:.4f}\\ {my_query_marked}"
    my_query_cell = f"[{my_query_text}]"

    # If multiple rows (multiple baselines), add rowspan to first two columns
    if num_rows > 1:
        title_cell = f"table.cell(colspan: 2, {title_cell})"
        my_query_cell = f"table.cell(rowspan: {num_rows}, {my_query_cell})"

    # Build paper_query markings for the first baseline
    paper_markings_0 = []
    if replacement_data:
        approach_0 = rows_data[0]["approach"]
        pairs_0 = replacement_data.get((review_id, approach_0), [])
        for pair in pairs_0:
            if pair["direction"] == "improve_mine":
                # paper_term is the good replacement -> highlight it
                paper_markings_0.append((pair["paper_term"], "highlight", pair["color"]))
            else:  # improve_paper
                # paper_term is replaceable -> underline it
                paper_markings_0.append((pair["paper_term"], "underline", pair["color"]))

    # Add the first baseline query row
    paper = f"*{rows_data[0]['approach']}* @{rows_data[0]['paper']}"
    paper_query = rows_data[0]["paper_query"]
    paper_p = rows_data[0]["paper_p"]
    paper_r = rows_data[0]["paper_r"]
    paper_query_marked = mark_outer_operators(mark_query_terms(paper_query, paper_markings_0), ['AND', 'NOT'])
    paper_query_text = f"{paper}\\ *Precision:* {paper_p:.4f} *Recall:* {paper_r:.4f}\\ {paper_query_marked}"
    paper_query_cell = f"[{paper_query_text}]"
    typst_lines.append(f"{title_cell}, {my_query_cell}, {paper_query_cell},")

    # Add remaining baseline query rows
    for row_data in rows_data[1:]:
        # Build paper_query markings for this baseline
        paper_markings = []
        if replacement_data:
            approach = row_data["approach"]
            pairs = replacement_data.get((review_id, approach), [])
            for pair in pairs:
                if pair["direction"] == "improve_mine":
                    paper_markings.append((pair["paper_term"], "highlight", pair["color"]))
                else:  # improve_paper
                    paper_markings.append((pair["paper_term"], "underline", pair["color"]))

        paper = f"*{row_data['approach']}* @{row_data['paper']}"
        paper_query = row_data["paper_query"]
        paper_p = row_data["paper_p"]
        paper_r = row_data["paper_r"]
        paper_query_marked = mark_outer_operators(mark_query_terms(paper_query, paper_markings), ['AND', 'NOT'])
        paper_query_text = f"{paper}\\ *Precision:* {paper_p:.4f} *Recall:* {paper_r:.4f}\\ {paper_query_marked}"
        paper_query_cell = f"[{paper_query_text}]"

        typst_lines.append(f"{paper_query_cell},")
    

if __name__ == "__main__":
    replacement_data = load_all_replacement_data(k=1)
    print(replacement_data)
    # exit(0)
    
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
            # positives = set(dataset_details[review_id]["positives"])
            # dataset, _, end_year = review_id_to_dataset(review_id)
            # precision, recall, retrieved_count, TP = evaluate_query(
            #     example["result"],
            #     positives,
            #     end_year=end_year,
            # )
            precision, recall, retrieved_count, TP = 1,1,1,1 #TODO remove
            

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

        
        dataframe_to_typst_query_table(table, output_path, replacement_data=replacement_data)
        print(f"\nTypst table written to: {output_path}")
