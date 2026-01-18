import optuna
study_name = "rf_optimization"
db_path = "sqlite:////data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/statistics/optuna/run2/optuna_rf_parallel.db"  # make sure this matches your DB
study = optuna.load_study(
    study_name=study_name,
    storage=db_path
)

# # 1️⃣ plot_param_importances
# # This is what you want for “impact of each variable on performance”.
# from optuna.visualization import plot_param_importances

# fig = plot_param_importances(study)
# fig.show()

# # study → your Optuna study object
# # Returns a bar chart showing which parameters most influenced the objective
# # Works best with TPE sampler, because it models parameter importance


# # 2️⃣ plot_parallel_coordinate
# # Shows interactions between hyperparameters and the objective:
# from optuna.visualization import plot_parallel_coordinate
# fig = plot_parallel_coordinate(study)
# fig.show()
# # Each axis → hyperparameter
# # Lines → trials
# # Objective value is usually color-coded
# # Great for spotting correlations between parameters and performance.


# # 3️⃣ plot_slice
# # Gives a scatter plot per hyperparameter:

# from optuna.visualization import plot_slice

# fig = plot_slice(study)
# fig.show()
# # X-axis → parameter value
# # Y-axis → objective value
# # Lets you see how objective changes with each parameter


# # 4️⃣ plot_contour
# # Shows 2D interactions between two parameters:
# from optuna.visualization import plot_contour
# fig = plot_contour(study, params=["max_depth", "max_features"])
# fig.show()
# # Contour heatmap of objective over two parameters
# # Very useful to see synergy / trade-offs


# 5
df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
for t in study.trials[:50]:
    print(f"\n=== Trial {t.number} ===")
    print(t.user_attrs.get("results_list"))
exit(0)

print(df.columns)
df.head()
param = "max_depth"
avg_df = df.groupby(f"params_{param}")["value"].mean().reset_index()
avg_df = avg_df.sort_values(f"params_{param}")
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(avg_df[f"params_{param}"], avg_df["value"], marker="o")
plt.xlabel(param)
plt.ylabel("Average objective")
plt.title(f"Average objective vs {param}")
plt.grid(True)
# plt.show()
plt.savefig(f"/data/horse/ws/flml293c-master-thesis/boolean-query-generation/data/statistics/optuna/images/average_objective_vs_{param}.png", dpi=300, bbox_inches="tight")