import os
import copy
import random
from itertools import product
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
    n_trials = 50
    n_query_ids = 5
    run_name = f"evaluate_best_{n_trials}_times"
    top_k_type = "cosine"
    os.makedirs(f"data/statistics/optuna/{run_name}", exist_ok=True)
    
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    dataset_details = get_dataset_details()
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    initial_solutions = load_initial_solutions(beta_min=50, beta_max=50)
    assert len(initial_solutions) == 1, "Expected exactly one best solution"
    best_params = initial_solutions[0]["params"]
    best_params = {
        "rf_params": params_from_opt_params(best_params, RF_PARAMS), 
        "qg_params": params_from_opt_params(best_params, QG_PARAMS)
        }
    
    all_query_ids = [qid for qid in dataset_details.keys() if len(dataset_details[qid]["positives"]) >= 50 and qid not in TRAIN_REVIEWS]
    all_query_ids = sorted(random.sample(all_query_ids, n_query_ids))
            
    # worker distribution: distribute 250 combinations (5 query_ids × 50 trials) across workers
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
        
    
    # Process assigned combinations
    for query_id, trial_n in my_combinations:
        rf_params = copy.deepcopy(best_params["rf_params"])
        qg_params = copy.deepcopy(best_params["qg_params"])
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
            meta_data=initial_solutions[0],
            max_retrieved=1_000_000,
            always_retrieve=True,
            ignore_pubmed_errors=True,
        )
