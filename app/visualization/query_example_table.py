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

if __name__ == "__main__":
    path = find_qg_results_file(
        CURRENT_BEST_RUN_FOLDER, top_k_type="cosine", betas_key="50"
    )
    dataframe = get_qg_results(path, min_positive_threshold=None)

    dataset_details = get_dataset_details()

    paper_query_examples = get_paper_query_examples()

    rows = []
    for paper, data in paper_query_examples.items():
        if paper == "Manual":
            continue
        print(f"Paper: {data['title']}")
        for example in data["examples"]:
            review_id = example.get("query_id")
            
            if not review_id or "usable_for_stats" in example and not example["usable_for_stats"]:
                print("Skipping example due to missing query_id or marked as not usable for stats.")
                continue
            positives = set(dataset_details[review_id]["positives"])
            dataset, _, end_year = review_id_to_dataset(review_id)
            precision, recall, retrieved_count, TP = evaluate_query(
                example["result"],
                positives,
                end_year=end_year,
            )
            
            # compare to manual
            manual_example = None
            for me in paper_query_examples["Manual"]["examples"]:
                if me["query_id"] == example["query_id"]:
                    manual_example = me
                    break
            manual_precision, manual_recall, manual_retrieved_count, TP = None, None, None, None
            if manual_example is not None:
                manual_precision, manual_recall, manual_retrieved_count, TP = evaluate_query(
                    manual_example["result"],
                    positives,
                    end_year=end_year,
                )
            
            # comapre to mine
            # get pubmed precisiona dn pubmed_recall from the row indataframe with the query that matches example["query"]
            row = dataframe[dataframe["query_id"] == example["query_id"]]
            pubmed_precision = row["pubmed_precision"].values[0]
            pubmed_recall = row["pubmed_recall"].values[0]
            pubmed_query = row["pubmed_query"].values[0]
            
            assert dataset_details[review_id]["title"] == example["query"]
            rows.append(
                {
                    "paper": paper,
                    "paper_p": precision,
                    "paper_r": recall,
                    "paper_stated_p": example.get("precision"),
                    "paper_stated_r": example.get("recall"),
                    "paper_stated_manual_p": example.get("manual_precision"),
                    "paper_stated_manual_r": example.get("manual_recall"),
                    "my_p": pubmed_precision,
                    "my_r": pubmed_recall,
                    "manual_p": manual_precision,
                    "manual_r": manual_recall,
                    "title": dataset_details[review_id]["title"],
                    "review_id": example["query_id"],
                    "paper_query": example["result"],
                    "my_query": pubmed_query,
                    "manual_query": manual_example["result"] if manual_example is not None else None,
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
            "manual_p",
            "manual_r",
            "paper_stated_manual_p",
            "paper_stated_manual_r",
        ]
        print("\nSummary table (queries omitted):")
        
        # print the table fully no matter if its too big
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(table[display_cols])