import json
from app.config.config import (
    BASE_VARIATIONS,
    BASE_VARIATIONS_NAMES,
    CURRENT_BEST_RUN_FOLDER,
    RESULT_TABLE_OPERATOR_METRICS_ORDERED,
    RESULT_TABLE_PERFORMANCE_METRICS_ORDERED,
    RESULT_TABLE_PERFORMANCE_METRICS,
    RESULT_TABLE_OPERATOR_METRICS,
    TRAIN_REVIEWS,
)
from app.dataset.utils import calc_missing_columns_in_result_df, get_dataset_details, get_qg_results
from app.statistics.query_size import calc_manual_query_size


if __name__ == "__main__":
    baseline_dict = {
        "tar2018": [
            {
                "name": "Manual",
                "Precision": (0.0217, None),
                "F1": (0.0407, None),
                "F3": (0.1439, None),
                "Recall": (0.9338, None),
                # 77.6,
                "\\#Rules": (1.0, 0),
                # 3.3,
                # 45.9,
                # 0.5,
            },  # original, conceptional and obj all from https://bevankoopman.github.io/papers/irj2020-comparison.pdf (same as in other verions of that paper, was also chosen as source from ChatGPT paper)
            (
                "Conceptual",
                0.0021,
                0.0037,
                0.0114,
                0.6286,
                None,
            ),  # highest recall, highest f3
            (
                "Objective",
                0.0002,
                0.0005,
                0.0022,
                0.8780,
                None,
            ),  # highest recall (since highest f3 has very low recall)
            (
                "ChatGPT",
                0.0752,
                0.0642,
                0.0847,
                0.5035,
                None,
            ),  # https://arxiv.org/pdf/2302.03495, highest recall, highest F3, with example q4
            # ("FI-BE-CONTXT", 0.0003, 0.0005, 0.0029, 0.9676 , None), # https://www.sciencedirect.com/science/article/pii/S1386505622002428 2 of the 3 above 80% recall frameworks (last of the 3 is simply bad (almost same recall as this and much lower precision)) -> simplys ay we only cosnidered above 80% recall in selection of those two values and then the best 2 of those 3
            {
                "name": "Fine-Tuned LLM",
                "Precision": (0.0499, None),
                "F1": None,
                "F3": (0.2401, None),
                "Recall": (0.8387, None),
            },  # highest recall (for clef TAR 2017+2018) from https://arxiv.org/pdf/2602.00005 (Autobool)
            (
                "Semantic",
                0.0236,
                0.0458,
                0.1872,
                0.8159,
                None,
            ),  # https://www.sciencedirect.com/science/article/pii/S1386505622002428 Agglo FI-BioBE-CONTXT -> most competitive in preicsion and recall from the 3 configs that have above 80% recall (only considering above 80% recall)
        ],
        "tar2019": [  # no value found
            # ("Original", "≤0.012\*", "", "", ""),
        ],
        "sr_updates": [
            # ("Original", "≤0.004\*", "", "", ""),
        ],
        "sigir2017": [
            # ("Original", "≤0.089\*", "", "", ""),
        ],
    }

    # set manual counts
    manual_counts = calc_manual_query_size()["tar2018"]
    for data in baseline_dict["tar2018"]:
        if isinstance(data, dict) and data["name"] == "Manual":
            for k, v in manual_counts.items():
                data[k] = (v["avg"], v["std"])

    # autobool
    dataset_details = get_dataset_details()
    autobool_df = get_qg_results(
        "data/examples/autobool_results.jsonl",
        min_positive_threshold=50,
        recompute_query_Size=True,
        include_top_k_type=False,
        datasets=["tar2018"],
        query_ids=dataset_details.keys()-TRAIN_REVIEWS,
    )
    name = "Fine-Tuned LLM (self-evaluation)"
    values = {"name": name}
    for metric in RESULT_TABLE_PERFORMANCE_METRICS_ORDERED:
        values[metric] = (
            autobool_df[RESULT_TABLE_PERFORMANCE_METRICS[metric]["key"]].mean(),
            autobool_df[RESULT_TABLE_PERFORMANCE_METRICS[metric]["key"]].std(),
        )
    for metric in RESULT_TABLE_OPERATOR_METRICS_ORDERED:
        values[metric] = (
            autobool_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]].mean(),
            autobool_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]].std(),
        )
    baseline_dict["tar2018"].append(values)
    print(
        name, "samples", len(autobool_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]])
    )

    # for name, path in other_baseline_paths.items():
    for name, _ in BASE_VARIATIONS.items():
        path = f"data/statistics/optuna/evaluate_base_{name}_{CURRENT_BEST_RUN_FOLDER.split('/')[-1]}"
        base_df = get_qg_results(
            path, min_positive_threshold=50, datasets=["tar2017", "tar2018"]
        )

        base_df = calc_missing_columns_in_result_df(base_df)

        name = BASE_VARIATIONS_NAMES[name.lower()]
        values = {"name": name}
        for metric in RESULT_TABLE_PERFORMANCE_METRICS_ORDERED:
            values[metric] = (
                base_df[RESULT_TABLE_PERFORMANCE_METRICS[metric]["key"]].mean(),
                base_df[RESULT_TABLE_PERFORMANCE_METRICS[metric]["key"]].std(),
            )
        for metric in RESULT_TABLE_OPERATOR_METRICS_ORDERED:
            values[metric] = (
                base_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]].mean(),
                base_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]].std(),
            )
        baseline_dict["tar2018"].append(values)
        print(
            name, "samples", len(base_df[RESULT_TABLE_OPERATOR_METRICS[metric]["key"]])
        )

    # # store the baseline dict as json to the path data/examples/baseline_values.json
    with open("data/examples/baseline_values.json", "w") as f:
        json.dump(baseline_dict, f, indent=4)
