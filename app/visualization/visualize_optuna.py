from app.config.config import apply_matplotlib_style, COLORS, COLORMAPS
from app.visualization.helper import prettify_axes
import optuna
import matplotlib.pyplot as plt
import os
from optuna.visualization.matplotlib import (
    plot_param_importances,
    plot_slice,
    plot_contour,
)  # , plot_parallel_coordinate
from collections import Counter

apply_matplotlib_style()


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

    print()
    print()
    states = [t.state for t in study.trials]
    state_counts = Counter(states)

    print("\n=== STUDY TRIAL STATE SUMMARY ===")

    for state, count in state_counts.items():
        print(f"{state.name:<10}: {count}")

    print("\n=== TOTAL ===")
    print(f"Total trials: {len(study.trials)}")


def plot_param_importances_custom(study, out_path: str):
    """
    Plot parameter importances with pretty-printed labels.

    Parameters:
    - study: Optuna study object
    - out_path: Path to save the figure
    """
    ax = plot_param_importances(study)
    # Set bar colors to primary color
    for patch in ax.patches:
        patch.set_color(COLORS["primary"])
    # Remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    prettify_axes(ax)
    plt.savefig(out_path)
    plt.close()


def plot_contour_custom(study, params: list[str], out_path: str, title: str):
    """
    Plot contour plot with pretty-printed labels.

    Parameters:
    - study: Optuna study object
    - params: List of two parameter names for contour plot
    - out_path: Path to save the figure
    - title: Optional custom title for the plot
    """
    ax = plot_contour(study, params=params)
    # Remove scatter points (black dots), keep only contours
    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PathCollection):
            collection.remove()
    # Update colormap from config for contourf collections
    cmap = plt.get_cmap(COLORMAPS["optuna"])
    for collection in ax.collections:
        if hasattr(collection, "set_cmap"):
            collection.set_cmap(cmap)
    ax.set_title(title)
    prettify_axes(ax)
    plt.savefig(out_path)
    plt.close()


def plot_slice_custom(study, params: list[str], out_path: str, title: str):
    """
    Plot slice plots for specific hyperparameters only.

    Parameters:
    - study: Optuna study object
    - params: List of parameter names to include in the slice plot
    - out_path: Path to save the figure
    - title: Optional custom title for the plot (applied as suptitle)
    """
    # Use official plot_slice with params filter
    ax = plot_slice(study, params=params)

    # Handle both single axis and array of axes
    if hasattr(ax, "__iter__"):
        axes_list = list(ax.flat)
    else:
        axes_list = [ax]

    # Update colormap and prettify labels
    cmap = plt.get_cmap(COLORMAPS["optuna"])
    for a in axes_list:
        prettify_axes(a)
        # Update colormap for scatter collections
        for collection in a.collections:
            if hasattr(collection, "set_cmap"):
                collection.set_cmap(cmap)

    plt.suptitle(title)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # Output directory for thesis images
    SAVE_DIR = "../master-thesis-writing/writing/thesis/images/graphs/optuna"
    os.makedirs(SAVE_DIR, exist_ok=True)

    study_name = "rf_optimization"
    # db_path = "sqlite:///data/statistics/optuna/run_2_nodes_10tasks_1cpu_per_task/optuna.db"
    # db_path = "sqlite:///data/statistics/optuna/run_10_nodes_10tasks_1cpu_per_task_opt_beta=6.0/optuna.db"
    # db_path = "sqlite:///data/statistics/optuna/run_10_nodes_10tasks_1cpu_per_task_opt_beta=10.0/optuna.db"
    db_path = "sqlite:///data/statistics/optuna/run_10_nodes_10tasks_1cpu_per_task_opt_beta=50.0/optuna.db"
    study = optuna.load_study(study_name=study_name, storage=db_path)

    # 1 plot_param_importances
    # title cannot be overwritten or deleted
    plot_param_importances_custom(
        study, out_path=os.path.join(SAVE_DIR, "param_importances.png")
    )

    # 2 plot_parallel_coordinate
    # plot_parallel_coordinate(study)
    # plt.savefig(os.path.join(SAVE_DIR, "parallel_coordinate.png"))
    # plt.close()

    # 3 plot_slice (for specific parameters only)
    slice_params = {
        "slice_1": [
            "mh_noexp",
            "tiab",
            "term_expansions",
        ],
        "slice2": [
            "cover_beta",
            "top_k",
            "class_weight",
        ],
        "slice_appendix1": [
            "dont_cares",
            "max_depth",
            "min_rule_occ",
            "min_tree_occ",
            "min_weight_fraction_leaf",
            "rank_weight",
            "pruning_beta",
        ],
        "slice_appendix2": [
            "max_features",
            "randomize_max_feature",
            "top_k_or_candidates",
            "min_impurity_decrease_range_start",
            "min_impurity_decrease_range_end",
            "randomize_min_impurity_decrease_range",
        ],
    }
    for name, params in slice_params.items():
        plot_slice_custom(
            study,
            params=params,
            out_path=os.path.join(SAVE_DIR, f"{name}.png"),
            title=None,
        )

    # 4 plot_contours
    combinations = [
        ("cover_beta", "pruning_beta"),
        ("term_expansions", "tiab"),
        ("top_k", "dont_cares"),
    ]
    for param1, param2 in combinations:
        plot_contour_custom(
            study,
            params=[param1, param2],
            out_path=os.path.join(SAVE_DIR, f"contour_{param1}_{param2}.png"),
            title=None,
        )

    # 5
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    df.head()
    param = "max_depth"
    avg_df = df.groupby(f"params_{param}")["value"].mean().reset_index()
    avg_df = avg_df.sort_values(f"params_{param}")

    plt.figure()
    plt.plot(avg_df[f"params_{param}"], avg_df["value"], marker="o")
    plt.xlabel(param)
    plt.ylabel("Average objective")
    plt.title(f"Average objective vs {param}")
    plt.grid(True)
    plt.savefig(
        os.path.join(SAVE_DIR, f"average_objective_vs_{param}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    exit(0)
    # 6 statistics
    print_stats(study=study)
