import optuna
import copy
import numpy as np
from pathlib import Path
import optuna
import numpy as np
from pathlib import Path
from functools import partial
from app.config.config import BOW_PARAMS
from app.experiments.evaluate_rf import evalaute_rf

def optimize_with_optuna_parallel(
    query_ids, 
    dataset, 
    ret_config,
    study_name="rf_optimization",
    db_path="sqlite:///optuna_results.db",
    n_trials=1000,
    n_jobs=4  # Number of parallel workers
):
    """
    Run Optuna hyperparameter optimization in parallel for RF model.

    Parameters
    ----------
    n_jobs : int
        Number of parallel trials to run.
    """

    # --- Load data once ---
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)

    # --- Prepare positives and ranking IDs ---
    sorted_ids = {}
    positives = {}
    total_docs = BOW_PARAMS["total_docs"]

    for query_id in query_ids:
        rankings_file = Path(
            f"../systematic-review-datasets/data/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/{query_id}.npz"
        )
        if not rankings_file.exists():
            print(f"Skipping {rankings_file}, does not exist")
            continue
        arr = np.load(rankings_file)
        sorted_ids[query_id] = arr["ids"]

        # Ground truth
        if query_id in dataset["EVAL"]:
            reviews = dataset["EVAL"]
        else:
            reviews = dataset["TRAIN"]
        positives[query_id] = set(
            [str(doc["pmid"]) for doc in reviews[query_id]["data"]["train"] if int(doc["label"]) == 1]
        )

    # --- Define Optuna objective ---
    def objective(trial):
        rf_params = copy.deepcopy(RF_PARAMS)
        rf_opt_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.00001, 0.01),
            "top_k": trial.suggest_float("top_k", 0,5, 2.0),
            "rank_weight": trial.suggest_float("rank_weight", 1.0, 5.0), # how much more weighted shall rank 1 be than rank k
            "max_features": trial.suggest_float("max_features", 0.01, 1.0),
            "min_impurity_decrease_range_start": trial.suggest_float("min_impurity_decrease_range_start", 0.001, 1.0),
            "min_impurity_decrease_range_end": trial.suggest_float("min_impurity_decrease_range_end", 0.001, 1.0), # TODO split start and end here
            "class_weight": trial.suggest_float("class_weight", 0.0, 1.0),
            "top_k_or_candidates": trial.suggest_categorical("top_k_or_candidates", [500, 1000, 1500]),
            #TODO maybe add randomizing variables
            # TODO make optuna parallel not the tree computation
        }
        for k, v in rf_opt_params.items():
            rf_params[k] = v
        qg_params = copy.deepcopy(QG_PARAMS)
        qg_opt_params = {
            "min_tree_occ": trial.suggest_float("min_tree_occ", 0.0, 0.1),
            "min_rule_occ": trial.suggest_float("min_rule_occ", 0.0, 0.1),
            "cover_beta": trial.suggest_float("cover_beta", 1.0, 5.0), # high to prefer covering training data fully
            "pruning_beta": trial.suggest_float("pruning_beta", 0.05, 1.0), # low to prefer precise rules
        }
        for k, v in qg_opt_params.items():
            qg_params[k] = v

        # Evaluate all queries
        scores = []
        for query_id in positives.keys():
            score = evalaute_rf(
                query_id=query_id,
                X=X,
                positives=positives[query_id],
                feature_names=feature_names,
                sorted_ids=sorted_ids[query_id],
                ordered_pmids=ordered_pmids,
                rf_params=rf_params,
                qg_params=qg_params,
                term_expansions=term_expansions
            )
            scores.append(score)
        # TODO use pruning to not always have to evalaute all 25
        return np.mean(scores) # TODO take max here (sadly our 8 combiantiosna re not aprt of the search space)
    

    # --- Create or load study ---
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=db_path,
        load_if_exists=True
    )

    # --- Run optimization in parallel ---
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("Best trial:")
    print(study.best_trial.params)
    print(f"Best value: {study.best_value}")

    return study



if __name__ == "__main__":
    study = optimize_with_optuna_parallel(
        query_ids=TRAIN_REVIEWS,
        BOW_PARAMS=BOW_PARAMS,
        dataset=dataset,
        load_vectors=load_vectors,
        load_synonym_map=load_synonym_map,
        evalaute_rf=evalaute_rf,
        ret_config={"model": "pubmedbert", "query_type": "title"},
        db_path="sqlite:///optuna_rf_parallel.db",
        n_trials=100,
        n_jobs=8  # Run 8 trials in parallel
    )