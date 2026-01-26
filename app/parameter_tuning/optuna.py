import traceback
import copy
import numpy as np
from pathlib import Path
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import sys
import os
from filelock import FileLock
from datetime import datetime
from pathlib import Path
from app.config.config import BOW_PARAMS, QG_PARAMS, RF_PARAMS, TRAIN_REVIEWS
from app.experiments.evaluate_rf import evaluate_rf
from app.dataset.utils import (
    load_vectors,
    load_synonym_map,
    qg_statistics_path,
    get_sorted_ids,
    get_positives,
)
from app.helper.helper import f_beta

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset

LAST_RF_P = None


def optimize_with_optuna_parallel(
    run_name,
    query_ids,
    ret_config,
    study_name="rf_optimization",
    run_path="sqlite:///optuna_results.db",
    n_trials=1000,
    n_jobs=4,  # Number of parallel workers
    opt_beta=3,
):
    """
    Run Optuna hyperparameter optimization in parallel for RF model.

    Parameters
    ----------
    n_jobs : int
        Number of parallel trials to run.
    """
    print("started loading dataset", flush=True)
    # --- Load data once ---
    dataset = load_dataset()
    print("finished loading dataset", flush=True)
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)

    # --- Prepare positives and ranking IDs ---
    sorted_ids = {}
    positives = {}
    total_docs = BOW_PARAMS["total_docs"]

    for query_id in query_ids:
        s_ids = get_sorted_ids(
            retriever_name=ret_config["model"],
            query_type=ret_config["query_type"],
            total_docs=total_docs,
            query_id=query_id,
        )
        if s_ids is None:
            print(f"Skipping {query_id},ranking rile does not exist", flush=True)
            continue
        sorted_ids[query_id] = s_ids

        # Ground truth
        positives[query_id] = get_positives(review_id=query_id, dataset=dataset)

    # --- Define Optuna objective ---
    def objective(trial):
        global LAST_RF_P
        rf_params = copy.deepcopy(RF_PARAMS)
        rf_change_prob = 0.1
        if np.random.rand() < rf_change_prob or LAST_RF_P is None:
            rf_opt_params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_weight_fraction_leaf": trial.suggest_float(
                    "min_weight_fraction_leaf", 0.0, 0.002, step=0.0002
                ),  # 0.002 = 1000 docs for {1:1, 0:1} (1000/500k)
                "top_k": trial.suggest_float("top_k", 0.5, 2.0, step=0.1),
                "dont_cares": trial.suggest_float("dont_cares", 0.0, 5.0, step=0.5),
                "rank_weight": trial.suggest_float(
                    "rank_weight", 1.0, 5.0, step=0.4
                ),  # how much more weighted shall rank 1 be than rank k
                "max_features": trial.suggest_float(
                    "max_features", 0.01, 1.00, step=0.01
                ),
                "min_impurity_decrease_range_start": trial.suggest_float(
                    "min_impurity_decrease_range_start", 0.001, 0.05, step=0.0035
                ),  # gini can at msot reduce by 0.5
                "min_impurity_decrease_range_end": trial.suggest_float(
                    "min_impurity_decrease_range_end", 0.001, 0.05, step=0.0035
                ),
                "class_weight": trial.suggest_float("class_weight", 0.0, 1.0, step=0.1),
                "top_k_or_candidates": trial.suggest_int(
                    "top_k_or_candidates", 500, 1500, step=500
                ),
                "randomize_max_feature": trial.suggest_float(
                    "randomize_max_feature", 0.0, 3.0, step=0.3
                ),
                "randomize_min_impurity_decrease_range": trial.suggest_float(
                    "randomize_min_impurity_decrease_range", 0.0, 3.0, step=0.3
                ),
                "n_jobs": None,
                "verbose": False,
                "n_estimators": 50,
            }
            LAST_RF_P = copy.deepcopy(rf_opt_params)
        else:
            rf_opt_params = {
                "max_depth": trial.suggest_int(
                    "max_depth", LAST_RF_P["max_depth"], LAST_RF_P["max_depth"]
                ),
                "min_weight_fraction_leaf": trial.suggest_float(
                    "min_weight_fraction_leaf",
                    LAST_RF_P["min_weight_fraction_leaf"],
                    LAST_RF_P["min_weight_fraction_leaf"],
                ),
                "top_k": trial.suggest_float(
                    "top_k", LAST_RF_P["top_k"], LAST_RF_P["top_k"]
                ),
                "dont_cares": trial.suggest_float(
                    "dont_cares", LAST_RF_P["dont_cares"], LAST_RF_P["dont_cares"]
                ),
                "rank_weight": trial.suggest_float(
                    "rank_weight", LAST_RF_P["rank_weight"], LAST_RF_P["rank_weight"]
                ),
                "max_features": trial.suggest_float(
                    "max_features", LAST_RF_P["max_features"], LAST_RF_P["max_features"]
                ),
                "min_impurity_decrease_range_start": trial.suggest_float(
                    "min_impurity_decrease_range_start",
                    LAST_RF_P["min_impurity_decrease_range_start"],
                    LAST_RF_P["min_impurity_decrease_range_start"],
                ),
                "min_impurity_decrease_range_end": trial.suggest_float(
                    "min_impurity_decrease_range_end",
                    LAST_RF_P["min_impurity_decrease_range_end"],
                    LAST_RF_P["min_impurity_decrease_range_end"],
                ),
                "class_weight": trial.suggest_float(
                    "class_weight", LAST_RF_P["class_weight"], LAST_RF_P["class_weight"]
                ),
                "top_k_or_candidates": trial.suggest_int(
                    "top_k_or_candidates",
                    LAST_RF_P["top_k_or_candidates"],
                    LAST_RF_P["top_k_or_candidates"],
                ),
                "randomize_max_feature": trial.suggest_float(
                    "randomize_max_feature",
                    LAST_RF_P["randomize_max_feature"],
                    LAST_RF_P["randomize_max_feature"],
                ),
                "randomize_min_impurity_decrease_range": trial.suggest_float(
                    "randomize_min_impurity_decrease_range",
                    LAST_RF_P["randomize_min_impurity_decrease_range"],
                    LAST_RF_P["randomize_min_impurity_decrease_range"],
                ),
            }

        for k, v in rf_opt_params.items():
            rf_params[k] = v
        qg_params = copy.deepcopy(QG_PARAMS)

        qg_opt_params = {
            "min_tree_occ": trial.suggest_float("min_tree_occ", 0.0, 0.1, step=0.01),
            "min_rule_occ": trial.suggest_float("min_rule_occ", 0.0, 0.1, step=0.01),
            "cover_beta": trial.suggest_float(
                "cover_beta", 1.0, 5.0, step=0.1
            ),  # high to prefer covering training data fully
            "pruning_beta": trial.suggest_float(
                "pruning_beta", 0.05, 1.0, step=0.05
            ),  # low to prefer precise rules
            "term_expansions": trial.suggest_categorical(
                "term_expansions", [True, False]
            ),
            "mh_noexp": trial.suggest_categorical("mh_noexp", [True, False]),
            "tiab": trial.suggest_categorical("tiab", [True, False]),
        }

        for k, v in qg_opt_params.items():
            qg_params[k] = v

        trial.set_user_attr("qg_params", copy.deepcopy(qg_params))
        trial.set_user_attr("rf_params", copy.deepcopy(rf_params))

        qg_base_path = qg_statistics_path(
            run_name=run_name, rf_args=rf_params, qg_args=qg_params
        )
        qg_results_path = Path(qg_base_path) / "qg_results.jsonl"
        with FileLock(qg_results_path.with_suffix(".privatelock")):
            if qg_results_path.exists():
                # Someone already computed this configuration or is currently at computing it
                # raise optuna.exceptions.TrialPruned()
                # do not use TrialPruned here as this can lead to still running trials being pruned
                # only prune inside the currently running trial
                print("[Failed because duplicate]")
                return float("nan")  # This will mark the trial as failed

            # Mark "in progress"
            qg_results_path.touch()

        # Evaluate all queries
        results_list = []
        opt_scores = []
        for i, query_id in enumerate(positives.keys()):
            try:
                qg_results = evaluate_rf(
                    run_name=run_name,
                    query_id=query_id,
                    X=X,
                    positives=positives[query_id],
                    feature_names=feature_names,
                    sorted_ids=sorted_ids[query_id],
                    ordered_pmids=ordered_pmids,
                    rf_params=copy.deepcopy(rf_params),
                    qg_params=copy.deepcopy(qg_params),
                    term_expansions=term_expansions,
                )
            except Exception as e:
                # Prune the Optuna trial
                traceback.print_exc()
                # trial.report(float("nan"), step=0)
                LAST_RF_P = None  # dont repeat bad parameters actively
                raise optuna.exceptions.TrialPruned()

            results_list.append(qg_results)

            f_score = f_beta(
                precision=qg_results.get("pubmed_precision", 0.0),
                recall=qg_results.get("pubmed_recall", 0.0),
                beta=opt_beta,
            )
            opt_scores.append(f_score)

            # report intermediate average to Optuna
            running_avg = sum(opt_scores) / len(opt_scores)
            trial.report(running_avg, step=i)

            # prune early if necessary
            if trial.should_prune():
                print(
                    f"[PRUNED because bad] Trial {trial.number} at step={i}, query_id={query_id}, current_value={sum(opt_scores) / len(opt_scores)}"
                )
                raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("results_list", results_list)

        if opt_scores:
            return sum(opt_scores) / len(opt_scores)
        else:
            return 0.0

    lock_path = Path(run_path) / "optuna.privatelock"
    # Create a lock (only the first node to reach this shall create the db)
    with FileLock(lock_path):
        # storage = JournalStorage(JournalFileBackend(str(Path(run_path) / "optuna_journal.log")))
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{run_path}/optuna.db",
            engine_kwargs={
                "connect_args": {"timeout": 300},
                "pool_pre_ping": True,
            },
        )
        # --- Create or load study ---
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            # storage=db_path,
            storage=storage,
            load_if_exists=True,
        )

    initial_good_params = copy.deepcopy(QG_PARAMS) | copy.deepcopy(RF_PARAMS)
    study.enqueue_trial(initial_good_params)

    # --- Run optimization in parallel ---
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("Best trial:")
    print(study.best_trial.params)
    print(f"Best value: {study.best_value}", flush=True)

    return study


if __name__ == "__main__":
    opt_beta = 6.0
    print("finished imports", flush=True)
    time_out = 300
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_10_nodes_10tasks_1cpu_per_task_opt_beta={opt_beta}"
    run_path = f"/data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/statistics/optuna/{run_name}"
    os.makedirs(run_path, exist_ok=True)

    # db_path = Path(f"sqlite:///{run_path}/optuna_rf_parallel.db?timeout={time_out}")
    # db_path = Path(run_path) / "optuna_journal.log"

    study = optimize_with_optuna_parallel(
        run_name=run_name,
        query_ids=TRAIN_REVIEWS,
        ret_config={"model": "pubmedbert", "query_type": "title_abstract"},
        study_name="rf_optimization",
        run_path=run_path,
        n_trials=64,
        n_jobs=1,  # this is threads (not using cpus-per-task)
        opt_beta=opt_beta,
    )
