from itertools import product
import json
import os
import copy
from pathlib import Path
from app.config.config import BOW_PARAMS, TRAIN_REVIEWS 
from app.experiments.evaluate_rf import evaluate_rf
from app.parameter_tuning.get_best_params import get_best_params
from app.dataset.utils import (
    get_sorted_ids,
    load_synonym_map,
    load_vectors,
    get_dataset_details,
    qg_statistics_path,
)


def load_existing_pairs(run_name, best_params, top_k_type_list, n_trials):
    """Load all existing (query_id, best_params_index, trial, top_k_type) pairs from results."""
    existing_pairs = set()

    # Iterate through all possible combinations of parameters
    for best_params_index in range(len(best_params)):
        for trial in range(n_trials):  # assuming max n_trials trials
            for top_k_type in top_k_type_list:
                params = copy.deepcopy(best_params[best_params_index])
                rf_params = copy.deepcopy(params["rf_params"])
                qg_params = copy.deepcopy(params["qg_params"])
                rf_params["top_k_type"] = top_k_type

                qg_base_path = qg_statistics_path(
                    run_name=run_name + f"/n{trial}",
                    rf_args=rf_params,
                    qg_args=qg_params,
                )
                qg_results_path = Path(os.path.join(qg_base_path, "qg_results.jsonl"))

                if qg_results_path.exists():
                    with open(qg_results_path, "r", encoding="utf-8") as f:
                        for line in f:
                            obj = json.loads(line)
                            query_id = obj.get("query_id")
                            if query_id:
                                existing_pairs.add(
                                    (query_id, best_params_index, trial, top_k_type)
                                )

    return existing_pairs


if __name__ == "__main__":
    run_name = "best5"
    n_trials = 10
    betas = {3, 15, 30, 50}
    top_k_type_list = ["pos_count", "cosine", "fixed"]
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
    dataset_details = get_dataset_details()
    all_query_ids = sorted(dataset_details.keys() - set(TRAIN_REVIEWS))
    # get best params
    best_params, initial_solutions = get_best_params(betas=betas, term_expansion=False)

    # load existing pairs and subtract from all combinations
    all_combinations = set(
        product(
            all_query_ids,
            list(range(len(best_params))),
            list(range(n_trials)),
            top_k_type_list,
        )
    )
    print(f"Total combinations: {len(all_combinations)}")

    existing_pairs = load_existing_pairs(
        run_name=run_name,
        best_params=best_params,
        top_k_type_list=top_k_type_list,
        n_trials=n_trials,
    )
    all_combinations = list(all_combinations - existing_pairs)
    print(f"Remaining combinations: {len(all_combinations)}")
    print(f"Existing combinations: {len(existing_pairs)}")

    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)

    # worker distribution
    proc_id = int(os.environ.get("SLURM_PROCID", 0))
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))

    # Distribute combinations: each worker gets combinations where (combo_index % n_tasks) == proc_id
    my_combinations = [
        combo for i, combo in enumerate(all_combinations) if i % n_tasks == proc_id
    ]
    print(
        f"Worker {proc_id}/{n_tasks} processing {len(my_combinations)} combinations",
        flush=True,
    )

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

    for query_id, best_params_index, trial, top_k_type in my_combinations:
        last_run_failed = False
        for i in range(10):
            best_p = best_params[best_params_index]
            rf_params = copy.deepcopy(best_p["rf_params"])
            qg_params = copy.deepcopy(best_p["qg_params"])
            rf_params["top_k_type"] = top_k_type
            qg_results = evaluate_rf(
                run_name=run_name + f"/n{trial}",
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
                meta_data=initial_solutions[best_params_index],
                max_retrieved=200_000,
                min_retrieved=1,
                always_retrieve=True,
                ignore_pubmed_errors=True,
                skip_existing=True,
                last_run_failed=last_run_failed,
            )
            if qg_results is not None:
                break
            else:
                last_run_failed = True
