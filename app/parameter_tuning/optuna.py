import optuna
import traceback
import copy
import numpy as np
from pathlib import Path
import optuna
import sys
import os
import math
from datetime import datetime
import numpy as np
from pathlib import Path
from app.config.config import BOW_PARAMS, QG_PARAMS, RF_PARAMS, TRAIN_REVIEWS
from app.experiments.evaluate_rf import evalaute_rf
from app.dataset.utils import load_vectors, load_synonym_map

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset

def check_repeated_trial(trial):
  optuna_study = trial.study
  

  for past_trial in optuna_study.get_trials():
    if past_trial.number == trial.number:
      continue

    past_params = past_trial.params
    repeated_trial = True
    if set(trial.params.keys()) != set(past_params.keys()):
        repeated_trial = False
    for key in trial.params:
        a = trial.params[key]
        b = past_params[key]
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-9):
                repeated_trial = False
                break
        else:
            if a != b:
                repeated_trial = False
                break
    
    if repeated_trial is True:
      print(f"past Trial: {past_trial.number}, current Trial: {trial.number}")
      print(f"Params of past Trial: {past_params}")
      print(f"Params of current Trial: {trial.params}")
      return past_trial.value

  return None

def optimize_with_optuna_parallel(
    run_name,
    query_ids, 
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
    dataset = load_dataset()
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
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.002, step=0.0002), # 0.002 = 1000 docs for {1:1, 0:1} (1000/500k)
            "top_k": trial.suggest_float("top_k", 0.5, 2.0, step=0.1),
            "rank_weight": trial.suggest_float("rank_weight", 1.0, 5.0, step=0.1), # how much more weighted shall rank 1 be than rank k
            "max_features": trial.suggest_float("max_features", 0.01, 1.00, step=0.01),
            # "min_impurity_decrease_range_start": trial.suggest_float("min_impurity_decrease_range_start", 0.001, 1.0, step=0.001),
            # "min_impurity_decrease_range_end": trial.suggest_float("min_impurity_decrease_range_end", 0.001, 1.0, step=0.001), # TODO split start and end here
            "class_weight": trial.suggest_float("class_weight", 0.0, 1.0, step=0.1),
            "top_k_or_candidates": trial.suggest_categorical("top_k_or_candidates", [500, 1000, 1500]),
            "randomize_max_feature": trial.suggest_float("randomize_max_feature", 0.0, 3.0, step=0.3),
            "randomize_min_impurity_decrease_range": trial.suggest_float("randomize_min_impurity_decrease_range", 0.0, 3.0, step=0.3),
            "n_jobs": None,
            "verbose": False,
            "n_estimators": 1, # TODO set back to 50
        }
        for k, v in rf_opt_params.items():
            rf_params[k] = v
        qg_params = copy.deepcopy(QG_PARAMS)
        
        qg_opt_params = {
            "min_tree_occ": trial.suggest_float("min_tree_occ", 0.0, 0.1, step=0.01),
            "min_rule_occ": trial.suggest_float("min_rule_occ", 0.0, 0.1, step=0.01),
            "cover_beta": trial.suggest_float("cover_beta", 1.0, 5.0, step=0.1), # high to prefer covering training data fully
            "pruning_beta": trial.suggest_float("pruning_beta", 0.05, 1.0, step=0.05), # low to prefer precise rules
            "term_expansions": trial.suggest_categorical("term_expansions", [True, False]),
            "mh_noexp": trial.suggest_categorical("mh_noexp", [True, False]),
            "tiab": trial.suggest_categorical("tiab", [True, False]),
        }
        
        # check for repeated parameter combintation
        # repeated_value = check_repeated_trial(trial)
        # if repeated_value is not None:
        #     return repeated_value
        
        for k, v in qg_opt_params.items():
            qg_params[k] = v

        # Evaluate all queries
        results_list = []
        for query_id in positives.keys():
            try:
                qg_results = evalaute_rf(
                    run_name=run_name,
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
            except Exception as e:
                # Prune the Optuna trial
                traceback.print_exc()
                # trial.report(float("nan"), step=0)
                raise optuna.exceptions.TrialPruned()
                
            results_list.append(qg_results)
        # TODO use pruning to not always have to evalaute all 25
        
        trial.set_user_attr("results_list", results_list)
        trial.set_user_attr("qg_params", qg_params)
        trial.set_user_attr("rf_params", rf_params)
        
        f_scores = []
        beta = 3
        beta2 = beta ** 2
        for d in results_list:
            p = d.get("pubmed_precision", 0.0)
            r = d.get("pubmed_recall", 0.0)

            denom = (1 + beta2) * p * r + (r * beta2 + p - beta2 * p * r)  # simplified below
            # standard F_beta formula: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
            if p == 0 and r == 0:
                f3 = 0.0
            else:
                f3 = (1 + beta2) * p * r / (beta2 * p + r)
            f_scores.append(f3)
        
        if f_scores:
            return sum(f_scores) / len(f_scores)
        else:
            return 0.0
    

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
    time_out = 300
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_path = f"/data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/statistics/optuna/{run_name}"
    os.makedirs(run_path, exist_ok=True)
    
    study = optimize_with_optuna_parallel(
        run_name=run_name,
        query_ids=TRAIN_REVIEWS,
        ret_config={"model": "pubmedbert", "query_type": "title"},
        study_name="rf_optimization",
        db_path=f"sqlite:///{run_path}/optuna_rf_parallel.db?timeout={time_out}",
        n_trials=10,
        n_jobs=2,
    )