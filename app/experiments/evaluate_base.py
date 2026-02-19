import os
import copy
from app.config.config import (
    BOW_PARAMS,
    CURRENT_BEST_RUN_FOLDER,
    TRAIN_REVIEWS,
    RF_PARAMS,
    QG_PARAMS,
)
from app.experiments.evaluate_rf import evaluate_rf
from app.parameter_tuning.optuna import load_initial_solutions, params_from_opt_params
from app.dataset.utils import (
    get_rf_and_qg_params,
    get_sorted_ids,
    load_synonym_map,
    load_vectors,
    get_dataset_details,
)

base_variations = {
    "no_Ors": {
        "randomize_min_impurity_decrease_range": 1.0,
        "min_impurity_decrease_range_start": 1.0, #no OR nodes
        "min_impurity_decrease_range_end": 1.0, #no OR nodes
    },
    "no_variation": {
        "acceptance_threshold": 1000_000.0, # only pruning, no variation
    },
    "random_forest": {
        # rf_params["max_features"] = "sqrt"
        # rf_params["randomize_max_feature"] = 1_000_000.0 # set amx features as RF would choose it
        "randomize_min_impurity_decrease_range": 1.0,
        "min_impurity_decrease_range_start": 1.0, #no OR nodes
        "min_impurity_decrease_range_end": 1.0, #no OR nodes
        "acceptance_threshold": 1000_000.0, # no variation
        "removal_threshold": 1000_000.0, # no greedy pruning
        "prefer_pos_splits": 1.0, # do not prefer positive splits
    }
}


if __name__ == "__main__":
    top_k_type = "cosine"
    rf_params, qg_params = get_rf_and_qg_params(
        CURRENT_BEST_RUN_FOLDER, top_k_type=top_k_type, betas_key="50"
    )
    
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    dataset_details = get_dataset_details()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    all_query_ids = sorted(dataset_details.keys() - set(TRAIN_REVIEWS))

    # worker distribution
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    # Split initial solutions evenly across tasks
    chunk_size = (len(all_query_ids) + n_tasks - 1) // n_tasks  # ceil division
    my_query_ids = all_query_ids[proc_id * chunk_size : (proc_id + 1) * chunk_size]

    for query_id in my_query_ids:
        s_ids, scores = get_sorted_ids(
            retriever_name=ret_config["model"],
            query_type=ret_config["query_type"],
            total_docs=BOW_PARAMS["total_docs"],
            query_id=query_id,
        )
        if s_ids is None:
            print(f"Skipping {query_id},ranking file does not exist", flush=True)
            continue
        sorted_ids[query_id] = s_ids
        sorted_scores[query_id] = scores

        # Ground truth
        positives[query_id] = set(dataset_details[query_id]["positives"])

    for base_name, param_changes in base_variations.items():
        run_name = f"evaluate_base_{base_name}_{CURRENT_BEST_RUN_FOLDER.split('/')[-1]}"
        rf_p = copy.deepcopy(rf_params)
        rf_p["top_k_type"] = top_k_type
        qg_p = copy.deepcopy(qg_params)
        
        for k, v in param_changes.items():
            if k == "acceptance_threshold" or k=="removal_threshold":
                for op in ["and", "or"]:
                    for case in [True, False]:
                        qg_p["pruning_thresholds"][op][case][k] = v
            else:
                rf_p[k] = v

        for query_id in my_query_ids:
            qg_results = evaluate_rf(
                run_name=run_name,
                query_id=query_id,
                X=X,
                positives=positives[query_id],
                feature_names=feature_names,
                sorted_ids=sorted_ids[query_id],
                sorted_scores=sorted_scores[query_id],
                ordered_pmids=ordered_pmids,
                rf_params=rf_p,
                qg_params=qg_p,
                term_expansions=term_expansions,
                meta_data=None,
                max_retrieved=1_000_000,
                always_retrieve=True,
                ignore_pubmed_errors=True,
            )
