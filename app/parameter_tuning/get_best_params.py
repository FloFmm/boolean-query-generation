import json
import os

from app.config.config import QG_PARAMS, RF_PARAMS
from app.parameter_tuning.optuna import load_initial_solutions, params_from_opt_params

def get_best_params(betas, term_expansion=False):
    with open(f"data/final/best_params_te={term_expansion}.json", "r") as f:
        initial_solutions = json.load(f)
    initial_solutions = [s for s in initial_solutions if s["beta"] in betas]
    best_params = [s["params"] for s in initial_solutions]
    print(f"[LOADED] {len(best_params)} parameter configs")
    best_params = [
        {
            "rf_params": params_from_opt_params(opt_p, RF_PARAMS),
            "qg_params": params_from_opt_params(opt_p, QG_PARAMS),
        }
        for opt_p in best_params
    ]
    return best_params, initial_solutions

if __name__ == "__main__":
    betas = {3, 15, 30, 50}
    
    sorted_ids = {}
    sorted_scores = {}
    positives = {}
    for te in [False, True]:
        ret_config = {"model": "pubmedbert", "query_type": "title_abstract"}
        initial_solutions = load_initial_solutions(betas, term_expansion=te)
        # safe best paramters to dis
        params_path = f"data/final/best_params_te={te}.json"
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(initial_solutions, f, indent=4)