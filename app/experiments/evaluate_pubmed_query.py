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
        output_file = jsonl_file.parent / f"pubmed_query,{abbreviate_params(optimization_metric=optimization_metric, constraint_metric=constraint_metric, constraint_value=constraint_value)}.jsonl"
        if skip_existing and output_file.exists():
            print(f"Output file{output_file} already exists. Skipping...")
            continue
            
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
            output_file.open("w", encoding="utf-8") as f_out:

            for line in f_in:
                data = json.loads(line)
                query_id = data["query_id"]
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

                # search pubmed
                print(pubmed_query_str)
                relevant_ids = search_pubmed_year_month(pubmed_query_str)

                # evaluate
                retrieved = set(str(x) for x in relevant_ids) # retrieved PMIDs
                positives = set([str(doc["pmid"]) for doc in eval_reviews[query_id]["data"]["train"] if int(doc["label"])==1])         # relevant PMIDs
                TP = len(retrieved & positives)
                precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
                recall = TP / len(positives) if len(positives) > 0 else 0.0

                print(f"Precision: {precision:.10f}")
                print(f"Recall: {recall:.10f}")

                # Write to new JSONL
                out_record = {
                    "query_id": query_id,
                    "precision": precision,
                    "recall": recall, 
                    "pubmed_query": pubmed_query_str,
                    "num_positives": len(relevant_ids),
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