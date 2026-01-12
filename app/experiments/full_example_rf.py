import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here

import sys
import pickle
import numpy as np
from app.dataset.utils import load_vectors
from app.tree_learning.random_forest import RandomForest
from app.dataset.utils import generate_labels, load_synonym_map
from app.pubmed.retrieval import search_pubmed_dynamic

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import create_retriever, build_global_corpus, load_dataset
print("finished imports")
# Give either a custom query or a query_id 
query_id = "CD009784"#"CD002115"#"CD008760"
query = "Management of faecal incontinence and constipation in adults with central neurological diseases"#cancer with legs heart attack"
ret_name = "pubmedbert"
ret_conf = {"type": "dense",
            "model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512
            }
total_docs = 433660
QG_PARAMS = {
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.05,
    "cost_factor": 0.002, # 50 ANDs are worth 0.1 F3 score
    "min_rule_precision": 0.01,
    "beta": 2,
    }
RF_PARAMS = {
    "n_estimators": 3,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.00005,
    "max_features": "sqrt",
    "randomize_max_feature": 1,
    "min_impurity_decrease_range": (0.01, 0.01),
    "randomize_min_impurity_decrease_range": 1,
    "bootstrap": True,
    "n_jobs": None,
    "random_state": None,
    "verbose": True,
    "class_weight": 1.0,
    "max_samples": None,
    "top_k_or_candidates": 500,
    "prefer_pos_splits": 1.1
}

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
print("Queries EVAL:", len(dataset["EVAL"]))
print("Queries TRAIN:", len(dataset["TRAIN"]))

### Dense Retrieval ###
if query_id is not None:
    if query_id in dataset["EVAL"]:
        reviews = dataset["EVAL"]
    else:
        reviews = dataset["TRAIN"]
    query = reviews[query_id]["dataset_details"]["title"]

REVIEWS_PATH = f"data/tmp/reviews_qid={query_id}_d={total_docs}.pkl"
with open(REVIEWS_PATH, "wb") as f:
    pickle.dump(reviews, f)
print("Query:", query)

retriever = create_retriever(ret_name, ret_conf, collection=global_corpus)

ranking = retriever.search(
    query=query, cutoff=10_000, return_docs=False
)
sorted_ids = sorted(ranking, key=ranking.get, reverse=True)
SORTED_IDS_PATH = f"data/tmp/sorted_ids_qid={query_id}_d={total_docs}.pkl"
with open(SORTED_IDS_PATH, "wb") as f:
    pickle.dump(sorted_ids, f)
        
### Train Decision Tree ###


X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=10, max_df=0.1, mesh=True)
print("Num Features:", len(feature_names))
print("Examples:")
for i in range(10):
    print('"' + feature_names[i] + '"')


qrels = {
    "pos": sorted_ids[:100],
    "neutral": sorted_ids[500:]
}
keep_indices, labels = generate_labels(qrels, ordered_pmids)
print("labels generated")

rf = RandomForest(**RF_PARAMS)
X = X[keep_indices]
rf.fit(
    X, np.array(labels), feature_names=feature_names
)
print("finished fitting")
### Generate Pubmed Query ###
# synonym_map = load_synonym_map(total_docs)
pubmed_query_str, query_size = rf.pubmed_query(
    X=X,
    labels=labels,
    feature_names=feature_names,
    min_tree_occ=QG_PARAMS["min_tree_occ"],
    min_rule_occ=QG_PARAMS["min_rule_occ"],
    cost_factor=QG_PARAMS["cost_factor"],
    min_rule_precision=QG_PARAMS["min_rule_precision"],
    beta=QG_PARAMS["beta"],
)
print("PubMed Query:", pubmed_query_str)

### Evaluate on PubMed ###
if query_id is not None:
    retrieved = search_pubmed_dynamic(pubmed_query_str)
    retrieved = set(str(x) for x in retrieved) # retrieved PMIDs
    positives = set([str(doc["pmid"]) for doc in reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
    print("Positives:", positives)
    TP = len(retrieved & positives)
    precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
    recall = TP / len(positives) if len(positives) > 0 else 0.0
    print(f"Pubmed Precision: {precision:.10f}")
    print(f"Pubmed Recall: {recall:.10f}")

    # evaluate on local subset
    # subset_preds = tree.predict(X)
    # label_lookup = {doc["pmid"]: int(doc["label"]) for doc in reviews[query_id]["data"]["train"]}
    # ground_truth = [label_lookup.get(pmid, 0) for pmid in ordered_pmids]
    # subset_precision = precision_score(ground_truth, subset_preds)
    # subset_recall = recall_score(ground_truth, subset_preds)
    # print(f"Subset Precision: {subset_precision:.10f}")
    # print(f"Subset Recall: {subset_recall:.10f}")