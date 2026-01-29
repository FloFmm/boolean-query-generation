import os
import copy
import sys
import json
import time
import pickle
import numpy as np
from filelock import FileLock
from app.config.config import DEBUG
from itertools import product
from pathlib import Path
from app.tree_learning.random_forest import RandomForest
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.dataset.utils import (
    generate_pseudo_labels_and_sample_weights,
    rf_statistics_path,
    qg_statistics_path,
    load_synonym_map,
    get_positives,
)
from app.pubmed.retrieval import search_pubmed_dynamic
from app.tree_learning.query_generation import (
    compute_rule_coverage,
    rules_to_pubmed_query,
)
from app.dataset.utils import review_id_to_dataset, load_vectors, run_path
from sklearn.metrics import recall_score, precision_score
from app.config.config import TRAIN_REVIEWS, BOW_PARAMS, RF_PARAMS, QG_PARAMS

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset


def evaluate_rf(
    run_name,
    query_id,
    X,
    positives,
    feature_names,
    sorted_ids,
    ordered_pmids,
    rf_params,
    qg_params,
    term_expansions=None,
    meta_data=None,
    max_retrieved=100_000,
):
    dataset, _, end_year = review_id_to_dataset(query_id)

    rf_base_path = rf_statistics_path(run_name=run_name, **rf_params)
    rf_model_path = Path(os.path.join(rf_base_path, f"{query_id}.pkl"))
    rf_results_path = Path(os.path.join(rf_base_path, f"rf_results.jsonl"))
    rf_config_path = Path(os.path.join(rf_base_path, f"rf_config.json"))
    qg_base_path = qg_statistics_path(
        run_name=run_name, rf_args=rf_params, qg_args=qg_params
    )
    qg_results_path = Path(os.path.join(qg_base_path, f"qg_results.jsonl"))
    qg_config_path = Path(os.path.join(qg_base_path, f"qg_config.json"))
    os.makedirs(qg_base_path, exist_ok=True)
    
    if meta_data is not None:
        qg_meta_path = Path(os.path.join(qg_base_path, f"qg_meta_data.json"))
        with open(qg_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)
            
    # check whether query already computed
    # with FileLock(qg_results_path.with_suffix(".privatelock")): # hold for the entire duration the lock for qg_results file
    #     if qg_results_path.exists():
    #         data = {}
    #         with open(qg_results_path, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 obj = json.loads(line)
    #                 if obj.get("query_id") == query_id:
    #                     data = obj
    #                     break

    #         if data:
    #             print("results already exists")
    #             return data
    #     else:
    with open(qg_config_path, "w", encoding="utf-8") as f:
        json.dump(qg_params, f, indent=4)

    pseudo_labels, sample_weight, top_k = generate_pseudo_labels_and_sample_weights(
        k=rf_params["top_k"],
        dont_cares=rf_params["dont_cares"],
        ordered_pmids=ordered_pmids,
        sorted_ids=sorted_ids,
        max_weight=rf_params["rank_weight"],
        num_positives=len(positives),
    )

    with FileLock(
        rf_model_path.with_suffix(".privatelock")
    ):  # hold for the time of the generation of the rf the lock to the model
        # check whether rf already exists
        if rf_model_path.exists():
            with open(rf_model_path, "rb") as f:
                rf = pickle.load(f)
                print("loaded existing rf model from disc", flush=True)
            # with open(rf_results_path, "r", encoding="utf-8") as f: # get rf stats for the qg result files
            #     for line in f:
            #         obj = json.loads(line)
            #         if obj.get("query_id") == query_id:
            #             rf_time_seconds = obj["time_seconds"]
            #             break
        else:
            with open(rf_config_path, "w", encoding="utf-8") as f:
                json.dump(rf_params, f, indent=4)
            rf = RandomForest(**rf_params)
            st = time.time()
            rf.fit(
                X,
                np.array(pseudo_labels),
                feature_names=feature_names,
                sample_weight=sample_weight,
            )
            rf_time_seconds = time.time() - st
            with open(rf_model_path, "wb") as f:
                pickle.dump(rf, f)

            rf_results = {
                "query_id": query_id,
                "num_positive": len(positives),
                "top_k": top_k,
                "time_seconds": rf_time_seconds,
            }
            with open(rf_results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rf_results) + "\n")

            print("finished fitting", flush=True)

    ### Generate Pubmed Query ###
    if qg_params["term_expansions"]:
        qg_params["term_expansions"] = term_expansions
    else:
        qg_params["term_expansions"] = None
    query_st = time.time()
    (pubmed_query_str, query_size), rules, optimization_score = rf.pubmed_query(
        X=X, labels=pseudo_labels, feature_names=feature_names, **qg_params
    )
    qg_time_seconds = time.time() - query_st
    # evaluate on local subset
    coverage = compute_rule_coverage(X=X, rules=rules)
    subset_preds = np.any(coverage, axis=0).astype(np.uint8)

    ground_truth_labels = [pmid in positives for pmid in ordered_pmids]
    subset_precision = precision_score(ground_truth_labels, subset_preds)
    subset_recall = recall_score(ground_truth_labels, subset_preds)
    pseudo_precision = precision_score(pseudo_labels, subset_preds)
    pseudo_recall = recall_score(pseudo_labels, subset_preds)

    retrieved = search_pubmed_dynamic(pubmed_query_str, end_year=end_year, max_retrieved=max_retrieved)
    retrieved = set(str(x) for x in retrieved)  # retrieved PMIDs
    true_positives = retrieved & positives
    TP = len(true_positives)
    pubmed_precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
    pubmed_recall = TP / len(positives) if len(positives) > 0 else 0.0

    qg_results = {
        "query_id": query_id,
        "num_positive": len(positives),
        "top_k": top_k,
        "pubmed_retrieved": len(retrieved),
        "pubmed_precision": pubmed_precision,
        "pubmed_recall": pubmed_recall,
        "subset_retrieved": int(subset_preds.sum()),
        "subset_precision": subset_precision,
        "subset_recall": subset_recall,
        "pseudo_precision": pseudo_precision,
        "pseudo_recall": pseudo_recall,
        "optimization_score": optimization_score,  # fromt he covering process
        "qg_time_seconds": qg_time_seconds,
        # "rf_time_seconds": rf_time_seconds,
        "query_size": query_size,
        "pubmed_query": pubmed_query_str,
    }
    with open(qg_results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(qg_results) + "\n")
    return qg_results


if __name__ == "__main__":
    query_ids = TRAIN_REVIEWS
    total_docs = BOW_PARAMS["total_docs"]
    ret_config = {"model": "pubmedbert", "query_type": "title"}
    dataset = load_dataset()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    sorted_ids = {}
    positives = {}
    for query_id in query_ids:
        rankings_file = Path(
            f"../systematic-review-datasets/data/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/{query_id}.npz"
        )
        if not rankings_file.exists():
            print(f"Skipping {rankings_file}, does not exist", flush=True)
            continue
        arr = np.load(rankings_file)
        sorted_ids[query_id] = arr["ids"]

        # ground truth
        positives[query_id] = get_positives(review_id=query_id, dataset=dataset)

    DEBUG = True
    buggy_config = {
        "max_depth": 14,
        "min_weight_fraction_leaf": 0.002,
        "top_k": 1.4,
        "rank_weight": 1.2,
        "max_features": 0.16,
        "class_weight": 0.1,
        "top_k_or_candidates": 500,
        "min_tree_occ": 0.07,
        "min_rule_occ": 0.08,
        "cover_beta": 4.7,
        "pruning_beta": 0.05,
    }
    rf_params = copy.deepcopy(RF_PARAMS)
    qg_params = copy.deepcopy(QG_PARAMS)
    for k, v in buggy_config.items():
        if k in rf_params:
            rf_params[k] = v
    for k, v in buggy_config.items():
        if k in qg_params:
            qg_params[k] = v

    results = []
    for query_id in (
        positives.keys()
    ):  # TODO change positives.keys( to all queries but some ranking are missing)
        r = evaluate_rf(
            query_id=query_id,
            X=X,
            positives=positives[query_id],
            feature_names=feature_names,
            sorted_ids=sorted_ids[query_id],
            ordered_pmids=ordered_pmids,
            rf_params=rf_params,
            qg_params=qg_params,
            term_expansions=term_expansions,
        )
        results.append(r)
