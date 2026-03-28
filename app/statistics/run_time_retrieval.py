import time
import os
import sys
import pickle
import json
import statistics

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here

from app.dataset.utils import get_dataset_details

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import (
    create_retriever,
    build_global_corpus,
    load_dataset,
)

print("finished imports")
# Give either a custom query or a query_id
ret_name = "pubmedbert"
ret_conf = {
    "type": "dense",
    "model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "max_length": 512,
}
total_docs = 503679

### Corpus ###
os.makedirs("data/tmp", exist_ok=True)
DATASET_PATH = f"data/tmp/dataset_{total_docs}.pkl"
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = load_dataset()
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)

GLOBAL_CORPUS_PATH = f"data/tmp/global_corpus_{total_docs}.pkl"
if os.path.exists(GLOBAL_CORPUS_PATH):
    with open(GLOBAL_CORPUS_PATH, "rb") as f:
        global_corpus = pickle.load(f)
else:
    global_corpus = build_global_corpus(dataset)
    with open(GLOBAL_CORPUS_PATH, "wb") as f:
        pickle.dump(global_corpus, f)
print("Documents:", len(global_corpus))

retriever = create_retriever(ret_name, ret_conf, collection=global_corpus)
dataset_details = get_dataset_details()

runtimes = []
for review_id, details in dataset_details.items():
    query = details["title"] + " " + details["abstract"]
    st = time.time()
    ranking = retriever.search(query=query, cutoff=10_000, return_docs=False)
    et = time.time()
    runtime = et - st
    runtimes.append(runtime)
    print(f"Retrieval time: {runtime:.2f} seconds")



if runtimes:
    avg_runtime = sum(runtimes) / len(runtimes)
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)
    median_runtime = statistics.median(runtimes)
else:
    avg_runtime = min_runtime = max_runtime = median_runtime = None

results = {
    "average": avg_runtime,
    "min": min_runtime,
    "max": max_runtime,
    "median": median_runtime,
    "raw_runtimes": runtimes
}

output_path = "data/statistics/retrieval_runtimes.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved runtime statistics to {output_path}")
