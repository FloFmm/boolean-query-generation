import sys
import os
import json
from app.dataset.utils import dataset_details_path, get_positives

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import (
    load_dataset,
)
print("finished imports")
dataset = load_dataset()
# dataset already loaded earlier
reviews = dataset["EVAL"] | dataset["TRAIN"]

queryid_to_details = {
    query_id: review_data["dataset_details"]
    for query_id, review_data in reviews.items()
}
for query_id, details in queryid_to_details.items():
    details["real_num_positives"] = len(get_positives(query_id=query_id, dataset=dataset))

out_path = dataset_details_path()
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queryid_to_details, f, indent=2)