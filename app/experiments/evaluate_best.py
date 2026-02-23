from itertools import product
import os
import copy
from app.config.config import BOW_PARAMS, TRAIN_REVIEWS, RF_PARAMS, QG_PARAMS
from app.experiments.evaluate_rf import evaluate_rf
from app.parameter_tuning.optuna import load_initial_solutions, params_from_opt_params
from app.dataset.utils import (
    get_sorted_ids,
    load_synonym_map,
    load_vectors,
    get_dataset_details,
)

if __name__ == "__main__":
    run_name = "best5"
    n_trials = 10
    betas = {3, 15, 30, 50}
    
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    initial_solutions = load_initial_solutions(betas)
    best_params = [s["params"] for s in initial_solutions]
    print(f"[LOADED] {len(best_params)} parameter configs")
    best_params = [
        {
            "rf_params": params_from_opt_params(opt_p, RF_PARAMS),
            "qg_params": params_from_opt_params(opt_p, QG_PARAMS),
        }
        for opt_p in best_params
    ]
    dataset_details = get_dataset_details()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    all_query_ids = sorted(dataset_details.keys() - set(TRAIN_REVIEWS))

    # worker distribution
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    # Create all combinations of query_ids and trials
    trial_ids = list(range(n_trials))
    all_combinations = list(product(all_query_ids, trial_ids))
    # Distribute combinations: each worker gets combinations where (combo_index % n_tasks) == proc_id
    my_combinations = [combo for i, combo in enumerate(all_combinations) if i % n_tasks == proc_id]
    print(f"Worker {proc_id}/{n_tasks} processing {len(my_combinations)} combinations", flush=True)
    
    for query_id in all_query_ids:
        s_ids, scores = get_sorted_ids(
            retriever_name=ret_config["model"],
            query_type=ret_config["query_type"],
            total_docs=BOW_PARAMS["total_docs"],
            query_id=query_id,
        )
        if s_ids is None:
            print(f"Skipping {query_id}, ranking file does not exist", flush=True)
            continue
        sorted_ids[query_id] = s_ids
        sorted_scores[query_id] = scores

        # Ground truth
        positives[query_id] = set(dataset_details[query_id]["positives"])

    for top_k_type in ["pos_count", "cosine", "fixed"]:
        for i, best_p in enumerate(best_params):
            for query_id, trial_n in my_combinations:
                rf_params = copy.deepcopy(best_p["rf_params"])
                qg_params = copy.deepcopy(best_p["qg_params"])
                rf_params["top_k_type"] = top_k_type
                qg_results = evaluate_rf(
                    run_name=run_name + f"/n{trial_n}",
                    query_id=query_id,
                    X=X,
                    positives=positives[query_id],
                    feature_names=feature_names,
                    sorted_ids=sorted_ids[query_id],
                    sorted_scores=sorted_scores[query_id],
                    ordered_pmids=ordered_pmids,
                    rf_params=rf_params,
                    qg_params=qg_params,
                    term_expansions=term_expansions,
                    meta_data=initial_solutions[i],
                    max_retrieved=1_000_000,
                    always_retrieve=True,
                    ignore_pubmed_errors=True,
                    skip_existing=True,
                )
