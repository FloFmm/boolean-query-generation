from app.config.config import apply_matplotlib_style, COLORS, COLORMAPS
from app.visualization.helper import REPLACEMENTS, prettify_axes
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
    # from app.visualization.helper import REPLACEMENTS
    # print(REPLACEMENTS)
    # exit(0)
    
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
        "slice1": [
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
            "top_k_or_candidates",
            "pruning_beta",
            "max_features",
        ],
        "slice_appendix2": [
            "max_depth",
            "dont_cares",
            "rank_weight",
        ],
        "slice_appendix3": [
            "min_rule_occ",
            "min_tree_occ",
            "min_weight_fraction_leaf",
        ],
        "slice_appendix4": [
            "min_impurity_decrease_range_start",
            "min_impurity_decrease_range_end",
            "randomize_min_impurity_decrease_range",
            # "randomize_max_feature",
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
