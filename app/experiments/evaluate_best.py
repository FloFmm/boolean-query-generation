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
    run_name = "best_new2"
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    dataset_details = get_dataset_details()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    initial_solutions = load_initial_solutions()
    best_params = [s["params"] for s in initial_solutions]
    print(f"[LOADED] {len(best_params)} parameter configs")
    best_params = [{
        "rf_params": params_from_opt_params(opt_p, RF_PARAMS), 
        "qg_params": params_from_opt_params(opt_p, QG_PARAMS)
        } for opt_p in best_params]
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
        
    for top_k_type in ["pos_count", "cosine", "fixed"]:
        for i, best_p in enumerate(best_params):
            for query_id in my_query_ids:
                rf_params=copy.deepcopy(best_p["rf_params"])
                qg_params=copy.deepcopy(best_p["qg_params"])
                rf_params["top_k_type"] = top_k_type
                qg_results = evaluate_rf(
                    run_name=run_name,
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
                )
