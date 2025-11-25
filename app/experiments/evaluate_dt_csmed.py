
import sys
import os
import json
import numpy as np
import time
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.tree_learning.logical_query_generation import train_text_classifier

def load_completed(jsonl_path: Path):
    """Load already processed ids."""
    if not jsonl_path.exists():
        return set()
    completed = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed.add(record["id"])
            except json.JSONDecodeError:
                continue
    return completed

def load_qrels(ranking_files, num_pos, num_neutral):
    """
    Load relevant and non-relevant PMIDs from ranking .npz files.
    Top 100 PMIDs are relevant (1), PMIDs below rank 200 are non-relevant (0).
    Returns a dict: qrels[review_id] = {1: relevant_pmids, 0: non_relevant_pmids}
    """
    qrels_by_query_id = {}

    for rankings_file in ranking_files:
        arr = np.load(rankings_file)
        pmids = arr["ids"]  # numpy array

        # Extract review ID from filename (remove .npz)
        review_id = Path(rankings_file).stem

        qrels_by_query_id[review_id] = {}
        qrels_by_query_id[review_id]["pos"] = pmids[:num_pos].tolist()     # relevant
        qrels_by_query_id[review_id]["neg"] = pmids[num_pos+num_neutral:].tolist()     # non-relevant
        qrels_by_query_id[review_id]["neutral"] = pmids[num_pos:num_pos+num_neutral].tolist()     # non-relevant

    return qrels_by_query_id

def evaluate_dt_csmed(
    models,
    output_path: str = "data/statistics/classifier_learning/csmed",
    skip_existing=False,
    min_f_occ={0: 20, 1: 3},
    num_pos=500,
    num_neutral=500,
    total_docs=433660,
    ret_config={"model": "pubmedbert", "query_type": "title"},
    data_base_path="../systematic-review-datasets/data",
):
    """
    Train text classifiers on csmed.
    Skips terms already in the output file.
    """
    
    bag_of_words_path = f"{data_base_path}/bag_of_words/bag_of_words_docs={total_docs}.jsonl"
    unique_words_path = f"{data_base_path}/bag_of_words/unique_words_docs={total_docs}.jsonl"
    ranking_files = Path(f"{data_base_path}/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/").glob('*.npz')
    
    with open(unique_words_path, "r", encoding="utf-8") as f:
        unique_words = json.load(f)

    docs_by_pmid = {}
    with open(bag_of_words_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["id"]
            docs_by_pmid[pmid] = " ".join(entry["bow"])
    
    qrels_by_query_id = load_qrels(ranking_files, num_pos=num_pos, num_neutral=num_neutral)


    Path(output_path).mkdir(parents=True, exist_ok=True)
    # docs_by_pmid, pmids_by_mesh = classify_by_mesh(baseline_folder, n_docs)
    
    for model in models:
        file_path = Path(
            os.path.join(
                output_path,
                f"{str(model).replace(' ', '')},docs={total_docs}k,min_f_occ_={min_f_occ}.jsonl",
            )
        )
        ranking_file = f"/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/rankings/MedCPT/title/docs=26018/CD000009.npz"
        if skip_existing:
            completed = load_completed(file_path)
            print(
                f"Already computed {len(completed)} queries, skipping those..."
            )

        with file_path.open("a", encoding="utf-8") as out_f:

            for query_id, qrels in qrels_by_query_id.items():
                if skip_existing and query_id in completed:
                    continue

                relevant_records = [
                    docs_by_pmid[pmid]
                    for pmid in qrels["pos"]
                ]
                negative_ids = list(set(docs_by_pmid.keys()) - set(qrels["pos"]) - set(qrels["neutral"]))
                negative_records = [
                    docs_by_pmid[pmid]
                    for pmid in negative_ids
                ]

                start_time = time.time()
                result = train_text_classifier(
                    model,
                    relevant_records,
                    negative_records,
                    min_f_occ=min_f_occ,
                )
                duration = time.time() - start_time

                record = {
                    "query_id": query_id,
                    "num_positive": len(relevant_records),
                    "num_negative": len(negative_records),
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "threshold": model._optimal_threshold
                    if model._optimal_threshold
                    else "",
                    "time_seconds": duration,
                    "boolean_function_set1": result["boolean_function_set1"],
                    "boolean_function_set2": result["boolean_function_set2"],
                    "pretty_print": result["pretty_print"],
                    "obj": result["obj"],
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()  # ensures progress is safely written
        print(f"✅ Completed training for {output_path}")

if __name__ == "__main__":
    models = []
    for min_samples_split in [2, 5, 10]:
        for min_impurity_d_start in [0.2, 0.1, 0.01, 0.001]:
            for min_impurity_d_end in [0.2, 0.1, 0.01, 0.001]:
                for top_k_or_candidates in [100, 500, 1000]:
                    for class_weight in ["balanced", {1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 4, 0: 1}, {1: 5, 0: 1}, {1: 6, 0: 1}, {1: 3, 0: 0.5}, {1: 500, 0: 0.5}, {1: 500, 0: 1}]:
                        models.append(
                            GreedyORDecisionTree(
                                max_depth=4,
                                min_samples_split=min_samples_split,
                                min_impurity_decrease_range=[
                                    min_impurity_d_start,
                                    min_impurity_d_end,
                                ],
                                top_k_or_candidates=top_k_or_candidates,
                                class_weight=class_weight,  # "balanced",
                                verbose=True,
                            )
                        )

    args = {
        "models": models,
        "skip_existing": True,
        "min_f_occ": {0: 10, 1: 2},
        "total_docs": 433660,
    }
    evaluate_dt_csmed(**args)