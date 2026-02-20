import matplotlib.pyplot as plt
import optuna
import os
from collections import Counter
import glob

def print_stats(study):
    best = study.best_trial
    print("=== BEST TRIAL SUMMARY ===")
    print(f"Trial number: {best.number}")
    print(f"Objective value: {best.value}")
    print(f"State: {best.state}")
    # ✅ Hyperparameters (clean)
    print("\n=== PARAMETERS ===")
    for k, v in best.params.items():
        print(f"{k}: {v}")
    # ✅ Parameter distributions (important for debugging search space)
    print("\n=== PARAMETER DISTRIBUTIONS ===")
    for k, dist in best.distributions.items():
        print(f"{k}: {dist}")
    # ✅ User attributes (your custom stuff: rf_params, qg_params, results_list)
    print("\n=== USER ATTRIBUTES ===")
    for k, v in best.user_attrs.items():
        print(f"{k}: {v}")
    import numpy as np

    p_scores = []
    r_scores = []

    for d in best.user_attrs["results_list"]:
        p_scores.append(d.get("pubmed_precision", 0.0))
        r_scores.append(d.get("pubmed_recall", 0.0))

    avg_precision = np.mean(p_scores) if p_scores else 0.0
    avg_recall = np.mean(r_scores) if r_scores else 0.0

    print("=== BEST TRIAL AVERAGES ===")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Number of Queries:    {len(p_scores)}")

    # with constraint
    def print_stats(study, min_recall=0.7):
        best_trial_recall_constraint = None
        best_value = None

        for t in study.trials:
            # Only completed trials
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue

            results = t.user_attrs.get("results_list")
            if not results:
                continue

            recalls = [d.get("pubmed_recall", 0.0) for d in results]
            precisions = [d.get("pubmed_precision", 0.0) for d in results]

            avg_recall = np.mean(recalls) if recalls else 0.0
            avg_precision = np.mean(precisions) if precisions else 0.0

            if avg_recall >= min_recall:
                if best_value is None or t.value > best_value:
                    best_value = t.value
                    best_trial_recall_constraint = (t, avg_recall, avg_precision)

        # Print result
        if best_trial_recall_constraint is None:
            print(f"\n❌ No trial found with average recall ≥ {int(min_recall * 100)}%")
        else:
            t, avg_recall, avg_precision = best_trial_recall_constraint

            print(f"\n=== BEST TRIAL WITH AVG RECALL ≥ {int(min_recall * 100)}% ===")
            print(f"Trial number:      {t.number}")
            print(f"Objective value:  {t.value}")
            print(f"Average Recall:   {avg_recall:.4f}")
            print(f"Average Precision:{avg_precision:.4f}")

            print("\n=== PARAMETERS ===")
            for k, v in t.params.items():
                print(f"{k}: {v}")
            print("\n=== USER ATTRIBUTES ===")
            for k, v in t.user_attrs.items():
                print(f"{k}: {v}")

    print()
    print_stats(study, min_recall=0.7)
    print()
    print_stats(study, min_recall=0.75)
    print()
    print_stats(study, min_recall=0.8)
    print()
    print_stats(study, min_recall=0.9)
    print()
    print_stats(study, min_recall=0.95)
    print()

def print_state_counts(study):
    states = [t.state for t in study.trials]
    state_counts = Counter(states)

    print("=== STUDY TRIAL STATE SUMMARY ===")

    for state, count in sorted(state_counts.items(), key=lambda x: x[0].name):
        print(f"{state.name:<10}: {count}")
    print(f"Total trials: {len(study.trials)}")
    
    return state_counts

if __name__ == "__main__":
    path = "data/statistics/optuna"
    
    # Optional: specify beta values to filter (e.g., [0.5, 0.7, 0.9] or None for all)
    opt_betas = None  # Set to list of values like [0.5, 0.7] to filter, or None for all files
    
    # go thourhg all optuna.db files in the apth and its subdirectories. each time print all stats and the directory it came from (use glob)
    db_files = glob.glob(f"{path}/**/optuna.db", recursive=True)
    betas = [3.0, 15.0, 30.0, 50.0]
    sum_state_counts = Counter()
    for db_file in sorted(db_files):
        directory = os.path.dirname(db_file)
        if not any(directory.endswith(f"opt_beta={beta}") for beta in betas):
            continue
        print(f"\n=== STUDY FROM: {directory} ===")
        study = optuna.load_study(
            study_name="rf_optimization", storage=f"sqlite:///{db_file}"
        )
        # print_stats(study)
        state_counts = print_state_counts(study)
        sum_state_counts.update(state_counts)
    print("\n=== AGGREGATED TRIAL STATE COUNTS ACROSS ALL STUDIES ===")
    for state, count in sorted(sum_state_counts.items(), key=lambda x: x[0].name):
        print(f"{state.name:<10}: {count}")
    print("COMPLETE + PRUNED:", sum_state_counts[optuna.trial.TrialState.COMPLETE] + sum_state_counts[optuna.trial.TrialState.PRUNED])
        
        