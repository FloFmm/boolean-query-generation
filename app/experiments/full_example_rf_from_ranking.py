import pickle
import numpy as np
from app.dataset.utils import load_vectors
from app.tree_learning.random_forest import RandomForest
from app.dataset.utils import generate_labels
from app.pubmed.retrieval import search_pubmed_dynamic
print("finished imports")
# Give either a custom query or a query_id 
query_id = "CD009784"#"CD002115"#"CD008760"
total_docs = 433660
end_year = 2018
QG_PARAMS = {
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.05,
    "cost_factor": 0.002, # 50 ANDs are worth 0.1 F3 score
    "min_rule_precision": 0.01,
    "beta": 0.7,
    }
RF_PARAMS = {
    "n_estimators": 10,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.00005,
    "max_features": 0.5,#"sqrt",
    "randomize_max_feature": 1,
    "min_impurity_decrease_range": (0.01, 0.01),
    "randomize_min_impurity_decrease_range": 1,
    "bootstrap": True,
    "n_jobs": None,
    "random_state": None,
    "verbose": True,
    "class_weight": {1:1, 0:1},
    "max_samples": None,
    "top_k_or_candidates": 500,
    "prefer_pos_splits": 1.1
}

SORTED_IDS_PATH = f"data/tmp/sorted_ids_qid={query_id}_d={total_docs}.pkl"
with open(SORTED_IDS_PATH, "rb") as f:
    sorted_ids = pickle.load(f)

REVIEWS_PATH = f"data/tmp/reviews_qid={query_id}_d={total_docs}.pkl"
with open(REVIEWS_PATH, "rb") as f:
    reviews = pickle.load(f)
                
### Train Decision Tree ###
X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=10, max_df=0.2, mesh=True)
print("Num Features:", len(feature_names))
print("Examples:")
for i in range(10):
    print('"' + feature_names[i] + '"')


# start from here
qrels = {
    "pos": sorted_ids[:100],
    "neutral": sorted_ids[500:]
}
keep_indices, labels = generate_labels(qrels, ordered_pmids, sample_prob=1.0)
print("labels generated", len(keep_indices), len(labels))

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
pubmed_query_str = f'({pubmed_query_str}) AND ("1800"[DP] : "{end_year}"[DP])'
print("PubMed Query:", pubmed_query_str)

### Evaluate on PubMed ###
if query_id is not None:
    retrieved = search_pubmed_dynamic(pubmed_query_str, end_year=end_year)
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