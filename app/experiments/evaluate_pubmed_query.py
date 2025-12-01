import json
import sys
import os
import numpy as np
from pathlib import Path
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.pubmed.retrieval import search_pubmed_year_month
from sklearn.metrics import recall_score, precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset
from app.dataset.utils import abbreviate_params, load_completed, load_qrels_from_rankings, generate_labels, load_synonym_map, statistics_base_path, load_vectors


def evaluate_pubmed_query(
    optimization_metric,
    constraint_metric,
    constraint_value,
    skip_existing=False,
    data_base_path="../systematic-review-datasets/data",
):
    dataset = load_dataset()
    eval_reviews = dataset["EVAL"]
    input_folder = statistics_base_path()

    # Loop through all JSONL files
    last_conf = None
    for jsonl_file in input_folder.glob("*/results_dt.jsonl"):
        print("Processing", jsonl_file)
        conf = {
            "optimization_metric":optimization_metric,
            "constraint_metric":constraint_metric,
            "constraint_value":constraint_value,
        }
        
        folder_path = jsonl_file.parent / f"{abbreviate_params(**conf)}"
        output_file = Path(os.path.join(folder_path, "results_qg.json"))
        conf_file_path = Path(os.path.join(folder_path, "config.json"))
        
        with conf_file_path.open("w", encoding="utf-8") as f:
            json.dump(conf, f, indent=4)
            
        if skip_existing:
            completed = load_completed(output_file)
            
        config_file = jsonl_file.parent / "config.json"
        with config_file.open("r", encoding="utf-8") as f:
            conf = json.load(f)
        total_docs, min_df, max_df, mesh, ret_config, positive_selection_conf = conf["total_docs"], conf["min_df"], conf["max_df"], conf["mesh"], conf["ret_config"], conf["positive_selection_conf"]
        
        def conf_equal(conf, last_conf, keys):
            if last_conf is None or conf is None:
                return False
            
            for key in keys:
                if key not in conf or key not in last_conf:
                    return False
                if conf[key] != last_conf[key]:
                    return False
            return True
        
        if not conf_equal(conf, last_conf, ["total_docs"]):
            synonym_map = load_synonym_map(conf["total_docs"])
        
        if not conf_equal(conf, last_conf, ["total_docs", "min_df", "max_df", "mesh"]):
            X, ordered_pmids, feature_names = load_vectors(total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
            
        if not conf_equal(conf, last_conf, ["total_docs", "ret_config", "positive_selection_conf"]):
            ranking_files = Path(f"{data_base_path}/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/").glob('*.npz')
            qrels_by_query_id = load_qrels_from_rankings(ranking_files, positive_selection_conf=positive_selection_conf)
        

        with jsonl_file.open("r", encoding="utf-8") as f_in, \
            output_file.open("a", encoding="utf-8") as f_out:

            for line in f_in:
                data = json.loads(line)
                query_id = data["query_id"]
                if query_id in completed:
                    print(f"skipping {query_id} since already present")
                tree_obj_json = data["obj"]

                # Load GreedyORDecisionTree
                tree = GreedyORDecisionTree.from_json(tree_obj_json)
                
                keep_indices, labels = generate_labels(qrels_by_query_id[query_id], ordered_pmids)
                num_pos = sum(labels)
                tree._find_optimal_threshold(
                    X[keep_indices],
                    np.array(labels),
                    metric=optimization_metric,#"pubmed_f2",
                    constraint=constraint_metric,#"pubmed_count",
                    constraint_value=constraint_value,#-1 * 50_000,#num_pos*50,
                    term_expansions=synonym_map
                )
                # Generate PubMed query
                pubmed_query_str = tree.pubmed_query(term_expansions=synonym_map)

                # evaluate on pubmed
                print(pubmed_query_str)
                retrieved = search_pubmed_year_month(pubmed_query_str)
                retrieved = set(str(x) for x in retrieved) # retrieved PMIDs
                positives = set([str(doc["pmid"]) for doc in eval_reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
                TP = len(retrieved & positives)
                precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
                recall = TP / len(positives) if len(positives) > 0 else 0.0

                print(f"Pubmed precision: {precision:.10f}")
                print(f"Pubmed Recall: {recall:.10f}")

                # evaluate on local subset
                subset_preds = tree.predict(X)
                ground_truth = []
                for pmid in ordered_pmids:
                    for doc in eval_reviews[query_id]["data"]["train"]:
                        if pmid == doc["id"]:
                            ground_truth.append(int(doc["label"]))
                subset_precision = precision_score(ground_truth, subset_preds)
                subset_recall = recall_score(ground_truth, subset_preds)
                print(f"Subset precision: {subset_precision:.10f}")
                print(f"Subset Recall: {subset_recall:.10f}")

                # Write to new JSONL
                out_record = {
                    "query_id": query_id,
                    "tp": len(positives),
                    "pubmed_retrieved": len(retrieved),
                    "pubmed_precision": precision,
                    "pubmed_recall": recall, 
                    "subset_precision": subset_precision,
                    "subset_recall": subset_recall,
                    "pubmed_query": pubmed_query_str,
                    
                }
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        print(f"✅ Processed {jsonl_file.name} → {output_file.name}")


if __name__ == "__main__":
    args = {
        "skip_existing": True,
        "optimization_metric": "pubmed_f3",
        "constraint_metric": "pubmed_count",
        "constraint_value": -1 * 50_000,
    }
    evaluate_pubmed_query(**args)