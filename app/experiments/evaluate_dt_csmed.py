
import os
import sys
import json
import time
import numpy as np
from itertools import product
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.tree_learning.logical_query_generation import train_text_classifier
from app.dataset.utils import load_completed, load_qrels_from_rankings, generate_labels, load_bow, statistics_sub_folder_path, load_vectors, EVAL_QUERY_IDS

def evaluate_dt_csmed(
    model_args,
    skip_existing=True,
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
    conf = locals()
    
    # synonym_map_path = f"{data_base_path}/bag_of_words/synonym_map_path_docs={total_docs}.jsonl"
    ranking_files = Path(f"{data_base_path}/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/").glob('*.npz')
    
    # with open(synonym_map_path, "r", encoding="utf-8") as f:
    #     synonym_map = json.load(f)
    #     unique_words = list(synonym_map.keys())

    X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
    print("Num Features:", len(feature_names), "examples:", feature_names[0], feature_names[-1], feature_names[-2])
    qrels_by_query_id = load_qrels_from_rankings(ranking_files, positive_selection_conf=positive_selection_conf)

    # conf = {
    #     "model_args":model_args,
    #     "total_docs":total_docs,
    #     "positive_selection_conf":positive_selection_conf,
    #     "ret_config":ret_config,
    #     "min_df":min_df,
    #     "max_df":max_df,
    #     "mesh":mesh,
    #     "data_base_path":data_base_path,
    # }
    model = GreedyORDecisionTree(**model_args)
    folder_path = statistics_sub_folder_path(
        model=model, 
        total_docs=total_docs, 
        positive_selection_conf=positive_selection_conf,
        ret_config=ret_config,
        min_df=min_df, 
        max_df=max_df, 
        mesh=mesh
    )
    os.makedirs(folder_path, exist_ok=True)
    file_path = Path(os.path.join(folder_path, "results_dt.jsonl"))
    print("Saving to", file_path)
    conf_file_path = Path(os.path.join(folder_path, "config.json"))
    
    with conf_file_path.open("w", encoding="utf-8") as f:
        json.dump(conf, f, indent=4)
    
    if skip_existing:
        completed = load_completed(file_path)
        print(
            f"Already computed {len(completed)} queries, skipping those..."
        )

    with file_path.open("a" if skip_existing else "w", encoding="utf-8") as out_f:

        for query_id, qrels in qrels_by_query_id.items():
            if query_id not in EVAL_QUERY_IDS:
                print("skipping not included in EVAL_QUERY_IDS")
                continue
            if skip_existing and query_id in completed:
                continue
            
            keep_indices, labels = generate_labels(qrels, ordered_pmids)
            start_time = time.time()
            result = train_text_classifier(
                clf=model,
                X=X[keep_indices],
                feature_names=feature_names,
                labels=np.array(labels),
            )
            duration = time.time() - start_time
            num_pos = sum(labels)
            record = {
                "query_id": query_id,
                "num_positive": num_pos,
                "num_negative": len(keep_indices)-num_pos,
                "precision": result["precision"],
                "recall": result["recall"],
                "time_seconds": duration,
                "obj": result["obj"],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # ensures progress is safely written
    print(f"✅ Completed training for {file_path}")

if __name__ == "__main__":
    # parameters
    total_docs = [433660]
    max_depth = [5]
    num_pos = [50, 100]
    num_neutral = [500, 1000]
    min_samples_split = [2]
    min_impurity_d_start = [0.1, 0.01, 0.001, 0.2]
    min_impurity_d_end = [0.001, 0.01, 0.1, 0.2]
    top_k_or_candidates = [1000]
    class_weight = [
        {1: 3, 0: 1}, 1.0, {1: 1, 0: 1}, {1: 2, 0: 1},
        {1: 5, 0: 1}, {1: 6, 0: 1},
        {1: 3, 0: 0.5}, {1: 500, 0: 1}
    ]
    
    
    args = [
        {
        "model_args": {
            "max_depth": md,
            "min_samples_split": mss,
            "min_impurity_decrease_range": [mid_s, mid_e],
            "top_k_or_candidates": topk,
            "class_weight": cw,
            "verbose": True,
        },
        "skip_existing": True,
        "total_docs": td,
        "positive_selection_conf":{
            "type": "abs", # "rel" or "rank"
            "num_pos": nump, # "f": lambda x: max(50,2x)
            "num_neutral": numn,
        },
    }
    for topk, mss, td, md, cw, mid_s, mid_e, numn, nump in product(
            top_k_or_candidates,
            min_samples_split,
            total_docs,
            max_depth,
            class_weight,
            min_impurity_d_start,
            min_impurity_d_end,
            num_neutral,
            num_pos,
        )
    ]
    
    job_idx = int(sys.argv[1])
    if job_idx < len(args):
        evaluate_dt_csmed(**args[job_idx])