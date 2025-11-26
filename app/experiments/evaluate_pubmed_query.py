import json
import sys
import os
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.pubmed.retrieval import search_pubmed_year_month
from sklearn.metrics import recall_score, precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset
from app.dataset.utils import load_completed, load_qrels_from_rankings, load_synonym_map

if __name__ == "__main__":
    total_docs = 433660
    dataset = load_dataset()
    eval_reviews = dataset["EVAL"]
    input_folder = Path("data/statistics/classifier_learning/csmed")
    output_folder = input_folder  # save in same folder

    # Term expansions dictionary (replace with your real expansions)
    synonym_map = load_synonym_map(total_docs)

    # Loop through all JSONL files
    for jsonl_file in input_folder.glob("*.jsonl"):
        if f"docs={total_docs}" not in jsonl_file.name or jsonl_file.name.endswith("_PubMed.jsonl"):
            print("skipping file", jsonl_file.name)
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

                # vectorize_texts(set1, set2, min_f_occ=min_f_occ)

                # tree._find_optimal_threshold(
                #     X,
                #     y,
                #     metric="fbeta",
                #     constraint="recall",
                #     constraint_value=0.7,
                # )
                # Generate PubMed query
                pubmed_query_str = tree.pubmed_query(term_expansions=synonym_map)

                # search pubmed
                print(pubmed_query_str)
                relevant_ids = search_pubmed_year_month(pubmed_query_str)

                # evaluate
                retrieved = set(str(x) for x in relevant_ids) # retrieved PMIDs
                positives = set([str(doc["pmid"]) for doc in eval_reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
                print(positives)
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