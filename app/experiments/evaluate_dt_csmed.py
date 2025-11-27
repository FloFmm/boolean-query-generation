
import os
import json
import time
import numpy as np
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.tree_learning.logical_query_generation import train_text_classifier
from app.dataset.utils import load_completed, load_qrels_from_rankings, load_bow, statistics_sub_folder_path, load_vectors

def evaluate_dt_csmed(
    models,
    skip_existing=False,
    min_f_occ={0: 20, 1: 3},
    positive_selection_conf: dict = {
        "type": "abs", # "rel" or "rank"
        "num_pos": 100, # "f": lambda x: max(50,2x)
        "num_neutral": 400,
    },
    total_docs=433660,
    ret_config={"model": "pubmedbert", "query_type": "title"},
    data_base_path="../systematic-review-datasets/data",
    min_df=10,
    max_df=0.5,
    mesh=True
):
    """
    Train text classifiers on csmed.
    Skips terms already in the output file.
    """
    
    # synonym_map_path = f"{data_base_path}/bag_of_words/synonym_map_path_docs={total_docs}.jsonl"
    ranking_files = Path(f"{data_base_path}/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/").glob('*.npz')
    
    # with open(synonym_map_path, "r", encoding="utf-8") as f:
    #     synonym_map = json.load(f)
    #     unique_words = list(synonym_map.keys())

    X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
    print("Num Features:", len(feature_names), "examples:", feature_names[0], feature_names[-1], feature_names[-2])
    qrels_by_query_id = load_qrels_from_rankings(ranking_files, positive_selection_conf=positive_selection_conf)

    
    for model in models:
        folder_path = statistics_sub_folder_path(
            model=model, 
            total_docs=total_docs, 
            min_df=min_df, 
            max_df=max_df, 
            positive_selection_conf=positive_selection_conf,
            mesh=mesh
        )
        os.makedirs(folder_path, exist_ok=True)
        file_path = Path(os.path.join(folder_path, "results_dt.jsonl"))
        
        if skip_existing:
            completed = load_completed(file_path)
            print(
                f"Already computed {len(completed)} queries, skipping those..."
            )

        with file_path.open("a", encoding="utf-8") as out_f:

            for query_id, qrels in qrels_by_query_id.items():
                if skip_existing and query_id in completed:
                    continue
                keep_indices = []
                labels = []
                num_pos = 0
                for i, pmid in enumerate(ordered_pmids):
                    if pmid in qrels["neutral"]:
                        continue
                    if pmid in qrels["pos"]:
                        labels.append(1)
                        num_pos += 1
                    else:
                        labels.append(0)
                    keep_indices.append(i)
                    
                start_time = time.time()
                result = train_text_classifier(
                    clf=model,
                    X=X[keep_indices],
                    feature_names=feature_names,
                    labels=np.array(labels),
                )
                duration = time.time() - start_time

                record = {
                    "query_id": query_id,
                    "num_positive": num_pos,
                    "num_negative": len(keep_indices)-num_pos,
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
        print(f"✅ Completed training for {file_path}")

if __name__ == "__main__":
    models = []
    for min_samples_split in [2, 5, 10]:
        for min_impurity_d_start in [0.2, 0.1, 0.01, 0.001]:
            for min_impurity_d_end in [0.2, 0.1, 0.01, 0.001]:
                for top_k_or_candidates in [100, 500, 1000]:
                    for class_weight in ["balanced", {1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 4, 0: 1}, {1: 5, 0: 1}, {1: 6, 0: 1}, {1: 3, 0: 0.5}, {1: 500, 0: 0.5}, {1: 500, 0: 1}]:
                        models.append(
                            GreedyORDecisionTree(
                                max_depth=5,
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
        "positive_selection_conf":{
            "type": "abs", # "rel" or "rank"
            "num_pos": 100, # "f": lambda x: max(50,2x)
            "num_neutral": 400,
        },
    }
    evaluate_dt_csmed(**args)