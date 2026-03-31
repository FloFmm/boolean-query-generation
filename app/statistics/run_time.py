import json
from pathlib import Path
from statistics import median
from app.config.config import CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import get_qg_results


if __name__ == "__main__":
    no_ors_config = {
        "top_k": 1.5,
        "top_k_type": "cosine",
        "dont_cares": 2.0,
        "rank_weight": 1.5,
        "n_estimators": 50,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_weight_fraction_leaf": 0.0002,
        "max_features": 0.5,
        "randomize_max_feature": 0.9,
        # "randomize_min_impurity_decrease_range": 1.0,
        # "min_impurity_decrease_range_start": 1.0,
        # "min_impurity_decrease_range_end": 1.0,
        # "bootstrap": True,
        # "n_jobs": None,
        # "random_state": None,
        # "verbose": False,
        "class_weight": 0.2,
        # "max_samples": None,
        "top_k_or_candidates": 500,
        "prefer_pos_splits": 1.1,
        "max_or_features": 10
    }
    
    
    
    for path in [CURRENT_BEST_RUN_FOLDER, "data/statistics/optuna/evaluate_base_no_Ors_best5"]:
        print(path)
        qg_files = list(Path(path).glob("**/rf_results.jsonl"))
        time_seconds = []
        for jsonl_path in qg_files:
            # laod the config fromt he same path as rf_results.jsonl and comapre with no_ors_config. if not the same then continue
            config_path = jsonl_path.parent / "rf_config.json"
            config_data = json.load(open(config_path, "r"))
            skip = False
            for k,v in no_ors_config.items():
                if k not in config_data or config_data[k] != v:
                    # print(k)
                    # print(config_data)
                    # exit(0)
                    # print(f"config in {config_path} does not match no_ors_config, skipping")
                    skip = True
            if skip:
                continue
            # if config_data != no_ors_config:
            #     print(f"config in {config_path} does not match no_ors_config, skipping")
            #     continue
            
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().rstrip('\x00\n')
                    if not line:
                        continue
                    data = json.loads(line)  # noqa: F821
                    time_seconds.append(data["time_seconds"])
        
        print("Median run time:", median(time_seconds), len(time_seconds))
        
        qg_data = get_qg_results(path)
        print(f"Median run time: {qg_data['qg_time_seconds'].median():.5f} seconds ({len(qg_data['qg_time_seconds'])} samples)")
        print()