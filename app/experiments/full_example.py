import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here

import sys
import numpy as np
from app.dataset.utils import load_vectors
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.tree_learning.logical_query_generation import train_text_classifier
from app.dataset.utils import generate_labels, load_synonym_map
from app.pubmed.retrieval import search_pubmed_dynamic
from sklearn.metrics import recall_score, precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import create_retriever, build_global_corpus, load_dataset

# Give either a custom query or a query_id 
query_id = "CD009784"#"CD002115"#"CD008760"
query = "Management of faecal incontinence and constipation in adults with central neurological diseases"#cancer with legs heart attack"
ret_name = "pubmedbert"
ret_conf = {"type": "dense",
            "model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512
            }
total_docs = 433660

### Corpus ###
dataset = load_dataset()
global_corpus = build_global_corpus(dataset)
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
print("Query:", query)

retriever = create_retriever(ret_name, ret_conf, collection=global_corpus)

ranking = retriever.search(
    query=query, cutoff=10_000, return_docs=False
)
sorted_ids = sorted(ranking, key=ranking.get, reverse=True)

### Train Decision Tree ###
qrels = {
    "pos": sorted_ids[:100],
    "neutral": sorted_ids[500:]
}

X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=10, max_df=0.1, mesh=True)
print("Num Features:", len(feature_names))
print("Examples:")
for i in range(10):
    print('"' + feature_names[i] + '"')

model_args = {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_impurity_decrease_range": [0.01, 0.01],
            "top_k_or_candidates": 1000,
            "class_weight": {1:2, 0:1},
            "verbose": True,
        }

tree = GreedyORDecisionTree(**model_args)

keep_indices, labels = generate_labels(qrels, ordered_pmids)
result = train_text_classifier(
    clf=tree,
    X=X[keep_indices],
    feature_names=feature_names,
    labels=np.array(labels),
)

print()
print(tree.pretty_print(verbose=True))
print("Precision Tree", result["precision"])
print("Recall Tree", result["recall"])

### Generate Pubmed Query ###
synonym_map = load_synonym_map(total_docs)
best_threshold, best_score, final_constraint_score = tree._find_optimal_threshold(
    X[keep_indices],
    np.array(labels),
    metric="f3",#"pubmed_f2",
    constraint={"metric": "pubmed_count", "value": -1 * 50_000},#"pubmed_count",-1 * 50_000,#num_pos*50,
    term_expansions=synonym_map
)
pubmed_query_str, query_size = tree.pubmed_query(term_expansions=synonym_map)
pubmed_query_str_no_exp, query_size_no_exp = tree.pubmed_query()
print("PubMed Query:", pubmed_query_str)
print("PubMed Query (No Expansion):", pubmed_query_str_no_exp)
print("Optimal Threshol:", best_threshold)
print("Query Size:", query_size)
print("Query Size (No Expansion):", query_size_no_exp)

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
    subset_preds = tree.predict(X)
    label_lookup = {doc["pmid"]: int(doc["label"]) for doc in reviews[query_id]["data"]["train"]}
    ground_truth = [label_lookup.get(pmid, 0) for pmid in ordered_pmids]
    subset_precision = precision_score(ground_truth, subset_preds)
    subset_recall = recall_score(ground_truth, subset_preds)
    print(f"Subset Precision: {subset_precision:.10f}")
    print(f"Subset Recall: {subset_recall:.10f}")