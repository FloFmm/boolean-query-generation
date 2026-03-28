import json
from pathlib import Path
from statistics import median
from app.config.config import CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import get_qg_results


if __name__ == "__main__":
    path = CURRENT_BEST_RUN_FOLDER
    qg_files = list(Path(path).glob("**/rf_results.jsonl"))
    time_seconds = []
    for jsonl_path in qg_files:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip('\x00\n')
                if not line:
                    continue
                data = json.loads(line)  # noqa: F821
                time_seconds.append(data["time_seconds"])
    
    print("Median run time:", median(time_seconds))
    
    qg_data = get_qg_results(CURRENT_BEST_RUN_FOLDER)
    print(f"Median run time: {qg_data['qg_time_seconds'].median():.2f} seconds")