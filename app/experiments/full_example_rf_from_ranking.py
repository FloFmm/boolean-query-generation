import pickle
import numpy as np
import time
import sys
import os
from pathlib import Path
from app.dataset.utils import load_vectors, load_synonym_map, get_sorted_ids
from app.tree_learning.random_forest import RandomForest
from app.dataset.utils import generate_labels_and_sample_weights, review_id_to_dataset#, generate_labels
from app.pubmed.retrieval import search_pubmed_dynamic
from app.tree_learning.query_generation import compute_rule_coverage, rules_to_pubmed_query
from sklearn.metrics import recall_score, precision_score
from app.config.config import QG_PARAMS, RF_PARAMS, BOW_PARAMS, DEBUG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset

# test changes
DEBUG = True
RF_PARAMS["verbose"] = True
RF_PARAMS["max_depth"] = 10
# RF_PARAMS["rank_weight"] = 1.0
# RF_PARAMS["min_weight_fraction_leaf"] = 0.0
# RF_PARAMS["class_weight"] = 0.0
# RF_PARAMS["rank_weight"] = 1.0
RF_PARAMS["n_estimators"] = 5
# QG_PARAMS["min_rule_occ"] = 0
# QG_PARAMS["min_rule_occ"] = 0
# QG_PARAMS["cover_beta"] = 3.0


print("finished imports")
# Give either a custom query or a query_id 
query_id = "CD009784"#"CD002115"#"CD008760"
dataset_name, _, end_year = review_id_to_dataset(query_id)
total_docs = 503679 #433660
BOW_PARAMS["total_docs"] = total_docs

term_expansions = load_synonym_map(**BOW_PARAMS)

# SORTED_IDS_PATH = f"data/tmp/sorted_ids_qid={query_id}_d={total_docs}.pkl"
# if SORTED_IDS_PATH.exits():
#     with open(SORTED_IDS_PATH, "rb") as f:
#         sorted_ids = pickle.load(f)
# else:
sorted_ids = get_sorted_ids(
    retriever_name="pubmedbert", 
    query_type="title_abstract", 
    total_docs=total_docs, 
    query_id=query_id
)
print("loaded rankings from file")

REVIEWS_PATH = Path(f"data/tmp/reviews_qid={query_id}_d={total_docs}.pkl")
if REVIEWS_PATH.exists():
    with open(REVIEWS_PATH, "rb") as f:
        reviews = pickle.load(f)
else:
    dataset = load_dataset()
    if query_id in dataset["EVAL"]:
        reviews = dataset["EVAL"]
    else:
        reviews = dataset["TRAIN"]
    with open(REVIEWS_PATH, "wb") as f:
        pickle.dump(reviews, f)
positives = set([str(doc["pmid"]) for doc in reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
                
                
### Train Decision Tree ###
X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
print("Num Features:", len(feature_names))
print("Examples:")
for i in range(10):
    print('"' + feature_names[i] + '"')

lg_time = time.time()
labels, sample_weight, top_k = generate_labels_and_sample_weights(k=RF_PARAMS["top_k"],
                                                           ordered_pmids=ordered_pmids, 
                                                           sorted_ids=sorted_ids, 
                                                           max_weight=RF_PARAMS["rank_weight"],
                                                           num_positives=len(positives))
lg_time = time.time() - lg_time
rf_time = time.time()
rf = RandomForest(**RF_PARAMS)
rf.fit(
    X, np.array(labels), feature_names=feature_names, sample_weight=sample_weight
)
rf_time = time.time() - rf_time
print("finished fitting")
### Generate Pubmed Query ###
# synonym_map = load_synonym_map(total_docs)

qg_time = time.time()
(pubmed_query_str, query_size), rules, opt_score = rf.pubmed_query(
    X=X,
    labels=labels,
    feature_names=feature_names,
    min_tree_occ=QG_PARAMS["min_tree_occ"],
    min_rule_occ=QG_PARAMS["min_rule_occ"],
    cost_factor=QG_PARAMS["cost_factor"],
    min_rule_precision=QG_PARAMS["min_rule_precision"],
    cover_beta=QG_PARAMS["cover_beta"],
    pruning_beta=QG_PARAMS["pruning_beta"],
    pruning_thresholds=QG_PARAMS["pruning_thresholds"],
    term_expansions=term_expansions if QG_PARAMS["term_expansions"] else None,
    mh_noexp=QG_PARAMS["mh_noexp"],
    tiab=QG_PARAMS["tiab"],
)
qg_time = time.time() - qg_time
# pubmed_query_str = f'({pubmed_query_str}) AND ("1800"[DP] : "{end_year}"[DP])'
print()
print("PubMed Query:", pubmed_query_str)
print()
print("SUBSET STATS")
# evaluate on local subset
coverage = compute_rule_coverage(X=X, rules=rules)
subset_preds = np.any(coverage, axis=0).astype(np.uint8)
label_lookup = {doc["pmid"]: int(doc["label"]) for doc in reviews[query_id]["data"]["train"]}
ground_truth = [label_lookup.get(pmid, 0) for pmid in ordered_pmids]
pseudo_relevant = set(sorted_ids[:top_k])
pseudo_ground_truth = [pmid in pseudo_relevant  for pmid in ordered_pmids]
subset_precision = precision_score(ground_truth, subset_preds)
subset_recall = recall_score(ground_truth, subset_preds)
pseudo_precision = precision_score(pseudo_ground_truth, subset_preds)
pseudo_recall = recall_score(pseudo_ground_truth, subset_preds)
print(f"Pseudolabel Precision: {pseudo_precision:.10f}")
print(f"Pseudolabel Recall: {pseudo_recall:.10f}")
print(f"Subset Precision: {subset_precision:.10f}")
print(f"Subset Recall: {subset_recall:.10f}")
print("rf_time", rf_time)
print("qg_time", qg_time)
print("lg_time", lg_time)

print()
base_str = pubmed_query_str
### Evaluate on PubMed ###

for tiab in [True, False]:
    for mh_noexp in [True, False]:
        for te in [term_expansions, None]:
            pubmed_query_str, query_size = rules_to_pubmed_query(
                rules=rules,
                feature_names=feature_names,
                term_expansions=te,
                tiab=tiab,
                mh_noexp=mh_noexp,
            )
            print()
            print("TIAB", tiab, "MH_NOEXP", mh_noexp, "term_expansions", te is not None)
            print()
            if query_id is not None:
                retrieved = search_pubmed_dynamic(pubmed_query_str, end_year=end_year)
                retrieved = set(str(x) for x in retrieved) # retrieved PMIDs
                print("Positives:", positives)
                true_positives = retrieved & positives
                TP = len(true_positives)
                precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
                recall = TP / len(positives) if len(positives) > 0 else 0.0
                print(f"Pubmed Precision: {precision:.10f}")
                print(f"Pubmed Recall: {recall:.10f}")

                # map PMID -> rank position (0-based)
                rank_lookup = {str(pmid): rank for rank, pmid in enumerate(sorted_ids)}
                # collect positions of positives that appear in the ranking
                positive_positions = [
                    rank_lookup[pmid]
                    for pmid in positives
                    if pmid in rank_lookup
                ]
                tp_positions = [
                    rank_lookup[pmid]
                    for pmid in true_positives
                    if pmid in rank_lookup
                ]
                # sort for readability
                positive_positions.sort()
                tp_positions.sort()
                print("Positions of positives (0-based):")
                print(positive_positions)
                print("Positions of TP (0-based):")
                print(tp_positions)

for t in rf.estimators_:
    print(t.pretty_print(verbose=True, X=X))
