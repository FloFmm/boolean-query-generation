import json
import sys
import os
import re
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.metrics import recall_score, precision_score
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree
from app.pubmed.retrieval import search_pubmed_dynamic

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset
from app.dataset.utils import (
    abbreviate_params,
    load_completed,
    load_qrels_from_rankings,
    generate_labels,
    load_synonym_map,
    statistics_base_path,
    load_vectors,
    EVAL_QUERY_IDS,
    get_positives,
)


def evaluate_pubmed_query(
    optimization_metric,
    constraint={"metric": "pubmed_count", "value": -1 * 50_000},
    skip_existing=True,
    data_base_path="../systematic-review-datasets/data",
    dt_data_paths=None,
):
    dataset = load_dataset()
    eval_reviews = dataset["EVAL"]
    if not dt_data_paths:
        input_folder = statistics_base_path()
        dt_data_paths = [
            jsonl_file.parent for jsonl_file in input_folder.glob("*/results_dt.jsonl")
        ]

    # Loop through all JSONL files
    last_conf = None
    for folder_path_dt in dt_data_paths:
        folder_path_dt = Path(folder_path_dt)
        print("Processing", folder_path_dt)
        conf_qg = {
            "optimization_metric": optimization_metric,
            "constraint": constraint,
        }

        folder_path_qg = folder_path_dt / f"{abbreviate_params(**conf_qg)}"
        os.makedirs(folder_path_qg, exist_ok=True)
        output_file = Path(os.path.join(folder_path_qg, "results_qg.jsonl"))
        conf_file_qg = Path(os.path.join(folder_path_qg, "config.json"))

        with conf_file_qg.open("w", encoding="utf-8") as f:
            json.dump(conf_qg, f, indent=4)

        if skip_existing:
            completed = load_completed(output_file)

        results_file_dt = folder_path_dt / "results_dt.jsonl"
        config_file_dt = folder_path_dt / "config.json"
        with config_file_dt.open("r", encoding="utf-8") as f:
            conf = json.load(f)
        total_docs, min_df, max_df, mesh, ret_config, positive_selection_conf = (
            conf["total_docs"],
            conf["min_df"],
            conf["max_df"],
            conf["mesh"],
            conf["ret_config"],
            conf["positive_selection_conf"],
        )

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
            X, ordered_pmids, feature_names = load_vectors(
                total_docs, min_df=min_df, max_df=max_df, mesh=mesh
            )

        if not conf_equal(
            conf, last_conf, ["total_docs", "ret_config", "positive_selection_conf"]
        ):
            ranking_files = Path(
                f"{data_base_path}/rankings/{ret_config['model']}/{ret_config['query_type']}/docs={total_docs}/"
            ).glob("*.npz")
            qrels_by_query_id = load_qrels_from_rankings(
                ranking_files, positive_selection_conf=positive_selection_conf
            )

        with (
            results_file_dt.open("r", encoding="utf-8") as f_in,
            output_file.open("a" if skip_existing else "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                data = json.loads(line)
                query_id = data["query_id"]
                if query_id not in EVAL_QUERY_IDS:
                    print("skipping not included id")
                    continue

                if skip_existing and query_id in completed:
                    print(f"skipping {query_id} since already present")
                    continue
                print("Processing query id", query_id)
                tree_obj_json = data["obj"]

                # Load GreedyORDecisionTree
                tree = GreedyORDecisionTree.from_json(tree_obj_json)

                # regex that matches ANY non-alphanumeric character (anything not a-z, A-Z, 0-9, or underscore)
                features_with_punct = set()
                punctuation_pattern = re.compile(r"[^A-Za-z0-9_]")
                for name in tree.get_feature_names():
                    if name.endswith("[mh]"):
                        continue
                    if name[0] == '"' and name[-1] == "]":
                        parts = name.split("[")
                        if len(parts) != 2 or len(parts[0]) <= 2:
                            features_with_punct.add(name)
                        elif punctuation_pattern.search(parts[0][1:-1]):
                            features_with_punct.add(name)
                    elif punctuation_pattern.search(name):
                        features_with_punct.add(name)
                if features_with_punct:
                    print(
                        "Skipping: Some feature names contain punctuation or special characters."
                    )
                    print(features_with_punct)
                    continue

                keep_indices, labels = generate_labels(
                    qrels_by_query_id[query_id], ordered_pmids
                )
                num_pos = sum(labels)
                best_threshold, best_score, final_constraint_score = (
                    tree._find_optimal_threshold(
                        X[keep_indices],
                        np.array(labels),
                        metric=optimization_metric,  # "pubmed_f2",
                        constraint=constraint,  # "pubmed_count",-1 * 50_000,#num_pos*50,
                        term_expansions=synonym_map,
                    )
                )
                # Generate PubMed query
                pubmed_query_str, query_size = tree.pubmed_query(
                    term_expansions=synonym_map
                )

                # evaluate on pubmed
                print(pubmed_query_str)
                retrieved = search_pubmed_dynamic(pubmed_query_str)
                retrieved = set(str(x) for x in retrieved)  # retrieved PMIDs
                positives = get_positives(
                    review_id=query_id, dataset=dataset
                )  # relevant PMIDs
                TP = len(retrieved & positives)
                precision = TP / len(retrieved) if len(retrieved) > 0 else 0.0
                recall = TP / len(positives) if len(positives) > 0 else 0.0

                print(f"Pubmed precision: {precision:.10f}")
                print(f"Pubmed Recall: {recall:.10f}")

                # evaluate on local subset
                subset_preds = tree.predict(X)

                # print("tp subest:", [pmid for pmid, pred in zip(ordered_pmids, subset_preds) if pred == 1 and pmid in positives])
                # print("tp_pubmed:", retrieved & positives)
                # print(subset_preds)
                # print("tree._optimal_threshold", tree._optimal_threshold)
                ground_truth = [int(str(pmid) in positives) for pmid in ordered_pmids]
                # print("pubmed_retrieved", len(retrieved))
                # print("pubmed_groundtruth", len(positives), positives)
                # print("subset_retrieved", subset_preds.sum())
                # p2 = [x for x,l in label_lookup.items() if l]
                # print("subset_groundtruth", len(p2), p2)
                # print(len(ground_truth), len(subset_preds))

                subset_precision = precision_score(ground_truth, subset_preds)
                subset_recall = recall_score(ground_truth, subset_preds)
                print(f"Subset precision: {subset_precision:.10f}")
                print(f"Subset Recall: {subset_recall:.10f}")
                # print(tree.pretty_print(verbose=True))
                # Write to new JSONL
                out_record = {
                    "query_id": query_id,
                    "tp": len(positives),
                    "pubmed_retrieved": len(retrieved),
                    "pubmed_precision": precision,
                    "pubmed_recall": recall,
                    "subset_retrieved": int(subset_preds.sum()),
                    "subset_precision": subset_precision,
                    "subset_recall": subset_recall,
                    "threshold": best_threshold,
                    "optimization_score": best_score,
                    "constraint_score": final_constraint_score,
                    "query_size": query_size,
                    "pubmed_query": pubmed_query_str,
                }
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        print(f"✅ Processed {results_file_dt.name} → {output_file.name}")


def main():
    optimization_metric = ["pubmed_f3", "f3"]
    constraint = [None, {"metric": "pubmed_count", "value": -1 * 50_000}]

    input_folder = statistics_base_path()

    all_dt_files = [
        jsonl_file.parent for jsonl_file in input_folder.glob("*/results_dt.jsonl")
    ]
    batch_size = 8
    dt_data_paths_batches = [
        all_dt_files[i : i + batch_size]
        for i in range(0, len(all_dt_files), batch_size)
    ]

    args = [
        {
            "skip_existing": True,
            "optimization_metric": om,
            "constraint": c,
            "dt_data_paths": dt_path,
        }
        for c, om, dt_path in product(
            constraint, optimization_metric, dt_data_paths_batches
        )
    ]

    job_idx = int(sys.argv[1])
    if job_idx < len(args):
        evaluate_pubmed_query(**args[job_idx])


def test():
    optimization_metric = ["pubmed_f3"]
    constraint = [None]

    # input_folder = statistics_base_path()
    # dt_data_paths = [[jsonl_file.parent for jsonl_file in input_folder.glob("*/results_dt.jsonl")]]
    dt_data_paths = [
        [
            "data/statistics/csmed/GreedyORDecisionTree(md=5,mss=2,midr=[0.1,0.1],tkoc=1000,cw={1:500,0:0.5}),d=433660,psc={'type':'abs','num_pos':100,'num_neutral':1000},rc={'model':'pubmedbert','query_type':'title'},mindf=10,maxdf=0.5,mesh=True"
        ]
    ]

    args = [
        {
            "skip_existing": False,
            "optimization_metric": om,
            "constraint": c,
            "dt_data_paths": dt_path,
        }
        for c, om, dt_path in product(constraint, optimization_metric, dt_data_paths)
    ]

    for i in range(len(args)):
        evaluate_pubmed_query(**args[i])


if __name__ == "__main__":
    main()
    # test()
