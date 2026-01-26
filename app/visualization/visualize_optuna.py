import optuna

study_name = "rf_optimization"
db_path = "sqlite:///data/statistics/optuna/run_2_nodes_10tasks_1cpu_per_task/optuna.db"  # make sure this matches your DB
study = optuna.load_study(study_name=study_name, storage=db_path)

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


# 1️⃣ plot_param_importances
# This is what you want for “impact of each variable on performance”.
from optuna.visualization import plot_param_importances

fig = plot_param_importances(study)
fig.show()

# study → your Optuna study object
# Returns a bar chart showing which parameters most influenced the objective
# Works best with TPE sampler, because it models parameter importance


# 2️⃣ plot_parallel_coordinate
# Shows interactions between hyperparameters and the objective:
from optuna.visualization import plot_parallel_coordinate

fig = plot_parallel_coordinate(study)
fig.show()
# Each axis → hyperparameter
# Lines → trials
# Objective value is usually color-coded
# Great for spotting correlations between parameters and performance.


# 3️⃣ plot_slice
# Gives a scatter plot per hyperparameter:

from optuna.visualization import plot_slice

fig = plot_slice(study)
fig.show()
# X-axis → parameter value
# Y-axis → objective value
# Lets you see how objective changes with each parameter


# 4️⃣ plot_contour
# Shows 2D interactions between two parameters:
from optuna.visualization import plot_contour

fig = plot_contour(study, params=["max_depth", "max_features"])
fig.show()
# Contour heatmap of objective over two parameters
# Very useful to see synergy / trade-offs


# 5
df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
# for t in study.trials[:50]:
#     print(f"\n=== Trial {t.number} ===")
#     print(t.user_attrs.get("results_list"))

# print(df.columns)
df.head()
param = "max_depth"
avg_df = df.groupby(f"params_{param}")["value"].mean().reset_index()
avg_df = avg_df.sort_values(f"params_{param}")
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(avg_df[f"params_{param}"], avg_df["value"], marker="o")
plt.xlabel(param)
plt.ylabel("Average objective")
plt.title(f"Average objective vs {param}")
plt.grid(True)
# plt.show()
plt.savefig(
    f"data/statistics/optuna/images/average_objective_vs_{param}.png",
    dpi=300,
    bbox_inches="tight",
)


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


from collections import Counter

print()
print()
states = [t.state for t in study.trials]
state_counts = Counter(states)

print("\n=== STUDY TRIAL STATE SUMMARY ===")

for state, count in state_counts.items():
    print(f"{state.name:<10}: {count}")

print("\n=== TOTAL ===")
print(f"Total trials: {len(study.trials)}")
