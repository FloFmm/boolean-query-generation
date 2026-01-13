import pickle
import numpy as np
from app.dataset.utils import load_vectors, load_synonym_map
from app.tree_learning.random_forest import RandomForest
from app.dataset.utils import generate_labels_and_sample_weights#, generate_labels
from app.pubmed.retrieval import search_pubmed_dynamic
from app.tree_learning.query_generation import compute_rule_coverage, rules_to_pubmed_query
from sklearn.metrics import recall_score, precision_score
print("finished imports")
# Give either a custom query or a query_id 
query_id = "CD009784"#"CD002115"#"CD008760"
total_docs = 433660
end_year = 2018
QG_PARAMS = {
    "top_k": 200,
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.02,
    "cost_factor": 0.002, # 50 ANDs are worth 0.1 F3 score
    "min_rule_precision": 0.01,
    "cover_beta": 2.0, # high to prefer covering training data fully
    "pruning_beta": 0.1, # low to prefer precise rules
    "synonym_expansion": True,
    "mh_noexp": True,
    "tiab": True,
    "pruning_thresholds": {
            "or": {
                False: {  # removal in negated term false -> is positive term
                    "acceptance_metric": "tp_gain",
                    "acceptance_threshold": -0.1,
                    "removal_threshold": -0.01,
                },
                True: {
                    "acceptance_metric": "precision_gain",
                    "acceptance_threshold": -0.1,
                    "removal_threshold": -0.01,
                },
            },
            "and": {
                False: {
                    "acceptance_metric": "precision_gain",
                    "acceptance_threshold": -0.1,
                    "removal_threshold": -0.01,
                },
                True: {
                    "acceptance_metric": "precision_gain",
                    "acceptance_threshold": -0.1,
                    "removal_threshold": -0.01,
                },
            },
        }
    }
RF_PARAMS = {
    "n_estimators": 32,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.00005,
    "max_features": 0.1,#"sqrt",
    "randomize_max_feature": 1,
    "min_impurity_decrease_range": (0.01, 0.01),
    "randomize_min_impurity_decrease_range": 1,
    "bootstrap": True,
    "n_jobs": 32,
    "random_state": None,
    "verbose": True,
    "class_weight": 0.2,
    "max_samples": None,
    "top_k_or_candidates": 500,
    "prefer_pos_splits": 1.1,
    "max_or_features": 10,
}
RET_PARAMS = {
}
term_expansions = load_synonym_map(total_docs)

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

labels, sample_weight = generate_labels_and_sample_weights(k=QG_PARAMS["top_k"],
                                                           ordered_pmids=ordered_pmids, 
                                                           sorted_ids=sorted_ids, 
                                                           max_weight=100)
rf = RandomForest(**RF_PARAMS)
rf.fit(
    X, np.array(labels), feature_names=feature_names, sample_weight=sample_weight
)
print("finished fitting")
### Generate Pubmed Query ###
# synonym_map = load_synonym_map(total_docs)
(pubmed_query_str, query_size), rules = rf.pubmed_query(
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
    term_expansions=term_expansions if QG_PARAMS["synonym_expansion"] else None,
    mh_noexp=QG_PARAMS["mh_noexp"],
    tiab=QG_PARAMS["tiab"],
)
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
pseudo_relevant = set(sorted_ids[:QG_PARAMS["top_k"]])
pseudo_ground_truth = [pmid in pseudo_relevant  for pmid in ordered_pmids]
subset_precision = precision_score(ground_truth, subset_preds)
subset_recall = recall_score(ground_truth, subset_preds)
pseudo_precision = precision_score(pseudo_ground_truth, subset_preds)
pseudo_recall = recall_score(pseudo_ground_truth, subset_preds)
print(f"Pseudolabel Precision: {pseudo_precision:.10f}")
print(f"Pseudolabel Recall: {pseudo_recall:.10f}")
print(f"Subset Precision: {subset_precision:.10f}")
print(f"Subset Recall: {subset_recall:.10f}")

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
                positives = set([str(doc["pmid"]) for doc in reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
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
