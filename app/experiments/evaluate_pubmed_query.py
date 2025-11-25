import json
import sys
import os
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.pubmed.retrieval import search_pubmed_year_month
from sklearn.metrics import recall_score, precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset

if __name__ == "__main__":
    dataset = load_dataset()
    eval_reviews = dataset["EVAL"]
    input_folder = Path("data/statistics/classifier_learning/csmed")
    output_folder = input_folder  # save in same folder

    # Term expansions dictionary (replace with your real expansions)
    term_expansions = {
        "run": ["run", "runs", "running"],
        "jump": ["jump", "jumps", "jumping"],
        # add all feature expansions here
    }

    # Loop through all JSONL files
    for jsonl_file in input_folder.glob("*.jsonl"):
        if jsonl_file.name.endswith("_PubMed.jsonl"):
            continue  # skip already processed files
        output_file = output_folder / f"{jsonl_file.name}".replace(".jsonl", "_PubMed.jsonl")

        with jsonl_file.open("r", encoding="utf-8") as f_in, \
            output_file.open("w", encoding="utf-8") as f_out:

            for line in f_in:
                data = json.loads(line)
                query_id = data["query_id"]
                tree_obj_json = data["obj"]

                # Load GreedyORDecisionTree
                tree = GreedyORDecisionTree.from_json(tree_obj_json)

                # Generate PubMed query
                pubmed_query_str = tree.pubmed_query(term_expansions=term_expansions)

                # search pubmed
                relevant_ids = search_pubmed_year_month(pubmed_query_str)

                # evaluate
                retrieved = set(str(x) for x in relevant_ids) # retrieved PMIDs
                positives = set([str(doc["pmid"]) for doc in eval_reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs

                # Print first 10 elements
                print("First 10 retrieved PMIDs:", list(retrieved)[:10])
                print("First 10 positive PMIDs:", list(positives)[:10])

                # Additional check: print types of elements
                print("Types of retrieved elements:", [type(x) for x in list(retrieved)[:10]])
                print("Types of positive elements:", [type(x) for x in list(positives)[:10]])

                exit(0)
                TP = len(retrieved & positives)
                precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
                recall = TP / len(positives) if len(positives) > 0 else 0.0

                print(f"Precision: {precision:.10f}")
                print(f"Recall: {recall:.10f}")

                # Write to new JSONL
                out_record = {
                    "query_id": query_id,
                    "precision": precision,
                    "recall": recall, 
                    "pubmed_query": pubmed_query_str,
                    "num_positives": len(relevant_ids),
                }
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        print(f"✅ Processed {jsonl_file.name} → {output_file.name}")