import os
import sys
import math
from app.config.config import BOW_PARAMS, TRAIN_REVIEWS
from app.experiments.evaluate_rf import evaluate_rf
from app.dataset.utils import (
    get_positives,
    get_sorted_ids,
    load_synonym_map,
    load_vectors,
)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, get_all_review_ids

if __name__ == "__main__":
    job_idx = int(sys.argv[1])
    
    
    best_params = [
        { # from the 12h f3 run (data/statistics/optuna/run_2_nodes_10tasks_1cpu_per_task)
            "rf_params": {
                "top_k": 0.7,
                "rank_weight": 5.0,
                "n_estimators": 50,
                "max_depth": 7,
                "min_weight_fraction_leaf": 0.0006000000000000001,
                "max_features": 0.16,
                "randomize_max_feature": 1.7999999999999998,
                "randomize_min_impurity_decrease_range": 2.1,
                "min_impurity_decrease_range_start": 0.022000000000000002,
                "min_impurity_decrease_range_end": 0.0395,
                "bootstrap": True,
                "n_jobs": None,
                "random_state": None,
                "verbose": False,
                "class_weight": 0.6000000000000001,
                "max_samples": None,
                "top_k_or_candidates": 1500,
                "prefer_pos_splits": 1.1,
                "max_or_features": 10,
            },
            "qg_params": {
                "min_tree_occ": 0.09,
                "min_rule_occ": 0.05,
                "cost_factor": 0.002,
                "min_rule_precision": 0.01,
                "cover_beta": 1.0,
                "pruning_beta": 0.1,
                "term_expansions": False,
                "mh_noexp": True,
                "tiab": True,
                "pruning_thresholds": {
                    "or": {
                        "false": {
                            "acceptance_metric": "tp_gain",
                            "acceptance_threshold": -0.1,
                            "removal_threshold": -0.01,
                        },
                        "true": {
                            "acceptance_metric": "precision_gain",
                            "acceptance_threshold": -0.1,
                            "removal_threshold": -0.01,
                        },
                    },
                    "and": {
                        "false": {
                            "acceptance_metric": "precision_gain",
                            "acceptance_threshold": -0.1,
                            "removal_threshold": -0.01,
                        },
                        "true": {
                            "acceptance_metric": "precision_gain",
                            "acceptance_threshold": -0.1,
                            "removal_threshold": -0.01,
                        },
                    },
                },
            },
        }
    ]
    run_name = "best"
    sorted_ids = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    dataset = load_dataset()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    all_query_ids = get_all_review_ids(dataset) - set(TRAIN_REVIEWS)
    bucket_size = math.ceil(len(all_query_ids)/10)
    all_query_ids = sorted(all_query_ids)[job_idx*bucket_size:(job_idx+1)*bucket_size]
    for query_id in all_query_ids:
        s_ids = get_sorted_ids(
            retriever_name=ret_config["model"],
            query_type=ret_config["query_type"],
            total_docs=BOW_PARAMS["total_docs"],
            query_id=query_id,
        )
        if s_ids is None:
            print(f"Skipping {query_id},ranking file does not exist", flush=True)
            continue
        sorted_ids[query_id] = s_ids

        # Ground truth
        positives[query_id] = get_positives(review_id=query_id, dataset=dataset)

    for best_p in best_params:
        for query_id in all_query_ids:
            qg_results = evaluate_rf(
                run_name=run_name,
                query_id=query_id,
                X=X,
                positives=positives[query_id],
                feature_names=feature_names,
                sorted_ids=sorted_ids[query_id],
                ordered_pmids=ordered_pmids,
                rf_params=best_p["rf_params"],
                qg_params=best_p["qg_params"],
                term_expansions=term_expansions,
            )
