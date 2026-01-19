import json
import os
import random
import numpy as np
import math
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from app.parameter_tuning.compute_top_k import approximate_y
from app.config.config import TAR2017_TRAIN, TAR2017_TEST, TAR2018_TEST, CSMED_COCHRANE_REVIEWS, TOP_K, BOW_PARAMS

ABBREVIATIONS = {
    "total_docs": "d",
    "min_df": "mindf",
    "max_df": "maxdf",
    "positive_selection_conf": "psc",
    "mesh": "mesh",
    "optimization": "opt", 
    "constraint": "c", 
    "ret_config": "rc",
    "lower_case": "lc",
    "mesh_ancestors": "ma",
    "rm_numbers": "rmn",
    "rm_punct": "rmp",
    "related_words": "rw",
    "bootstrap":"boot,",
    "class_weight":"cw",
    "max_depth": "maxd",
    "max_features": "maxf",
    "max_or_features": "mof",
    "max_samples": "maxs",
    "min_impurity_decrease_range_start": "midrs",
    "min_impurity_decrease_range_end": "midre",
    "min_samples_split":"mins",
    "min_weight_fraction_leaf": "mwfl",
    "n_estimators": "ne",
    # "n_jobs": "nj",
    "prefer_pos_splits": "pfs",
    # "random_state":"rs",
    "randomize_max_feature": "rmf",
    "randomize_min_impurity_decrease_range" : "rmidr",
    "rank_weight": "rweight",
    "top_k": "k",
    "top_k_or_candidates": "tkoc",
    "term_expansions": "te",
    # "verbose": "v",
}

ABBREVIATIONS_REV = {v: k for k, v in ABBREVIATIONS.items()}

EVAL_QUERY_IDS_OLD = ["CD011602", "CD011926", "CD010225", "CD003137", "CD002069", "CD011724", "CD010633", "CD007497", "CD011549", "CD007103", "CD010411", "CD011447", "CD009925", "CD000384", "CD009669", "CD009780", "CD010387", "CD010653", "CD004288", "CD011732", "CD007379", "CD010139", "CD011472", "CD012009", "CD012216", "CD008366", "CD003344", "CD006342", "CD010685", "CD005055", "CD010226", "CD008760", "CD008170", "CD002898", "CD006995", "CD011515", "CD009782", "CD006839", "CD002115", "CD009784"]

def abbreviate_value(value) -> str:
    if isinstance(value, float):
        return format(value, ".6g")

    if isinstance(value, dict):
        items = []
        for k in sorted(value):
            abbr_k = ABBREVIATIONS.get(k, k)
            items.append(f"{abbr_k}:{abbreviate_value(value[k])}")
        return "{" + ",".join(items) + "}"

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(abbreviate_value(v) for v in value) + "]"

    return str(value)

def abbreviate_params(**kwargs) -> str:
    """
    Converts keyword parameters into the "key=value" format
    using the ABBREVIATIONS dict.
    """
    ignore = {}
    parts = []
    for full in sorted(kwargs):
        if full in ignore:
            continue
        # value = kwargs[full]
        # if isinstance(value, float):
        #     value_str = format(value, ".6g")
        # else:
        #     value_str = str(value)
        # abbr = ABBREVIATIONS.get(full, full) # default take full
        abbr = ABBREVIATIONS.get(full, full)
        value_str = abbreviate_value(kwargs[full])
        parts.append(f"{abbr}={value_str}")
    return ",".join(parts)

def data_base_path():
    return "../systematic-review-datasets/data"

def statistics_base_path():
    return Path("../boolean-query-generation/data/statistics")

def bag_of_words_path(**args):
    """params: total_docs, lower_case, mesh_ancestors, rm_numbers, rm_punct"""
    args = {k: v for k, v in args.items() if k not in ("min_df", "max_df", "mesh")}
    return Path(f"{data_base_path()}/bag_of_words/bag_of_words,{abbreviate_params(**args)}.jsonl")

def synonym_map_path(**args):
    """params: total_docs, lower_case, mesh_ancestors, rm_numbers, rm_punct"""
    args = {k: v for k, v in args.items() if k not in ("min_df", "max_df", "mesh")}
    return Path(f"../systematic-review-datasets/data/bag_of_words/synonym_map,{abbreviate_params(**args)}.json")

def run_path(run_name, **bow_args):
    return Path(os.path.join(statistics_base_path(), "optuna", run_name, f"{abbreviate_params(**bow_args)}"))

def rf_statistics_path(run_name, **args):
    args = {k: v for k, v in args.items() if k not in ("verbose", "random_state", "n_jobs")}
    return Path(os.path.join(run_path(run_name=run_name, **BOW_PARAMS), abbreviate_params(**args)))

def qg_statistics_path(run_name, rf_args, qg_args):
    qg_args = {k: v for k, v in qg_args.items() if k not in ("pruning_thresholds")}
    return Path(f"{rf_statistics_path(run_name=run_name, **rf_args)}/{abbreviate_params(**qg_args)}")

def statistics_sub_folder_path(model, **args):
    """params: model, total_docs, min_df, max_df, positive_selection_conf, mesh"""
    params = abbreviate_params(**args)
    path = Path(
        os.path.join(
                statistics_base_path(),
                f"{str(model).replace(' ', '')},{params}".replace(' ', ''),
            )
        )
    
    return path

def faeature_names_path(**args):
    """params: total_docs, min_df, max_df, mesh"""
    return Path(f"{data_base_path()}/bag_of_words/feature_names,{abbreviate_params(**args)}.pkl")

def vectors_path(**args):
    """params: total_docs, min_df, max_df, mesh"""
    return Path(f"{data_base_path()}/bag_of_words/vectors,{abbreviate_params(**args)}.pkl")

def load_synonym_map(**args):
    with open(synonym_map_path(**args), "r", encoding="utf-8") as f:
        synonym_map = json.load(f)
    return synonym_map

def load_completed(jsonl_path: Path):
    """Load already processed ids."""
    if not jsonl_path.exists():
        return set()
    completed = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed.add(record["query_id"])
            except json.JSONDecodeError:
                continue
    return completed

def load_vectors(**bow_args):
    X_path = vectors_path(**bow_args)
    features_path = faeature_names_path(**bow_args)
    
    bow = load_bow(**bow_args)
    ordered_pmids = list(bow.keys())

    # If cached vectors exist → load and return
    if os.path.exists(X_path) and os.path.exists(features_path):
        print("Loading vectors from disc")
        with open(X_path, "rb") as f:
            X = pickle.load(f)
        with open(features_path, "rb") as f:
            feature_names = pickle.load(f)
        return X, ordered_pmids, feature_names

    print("Compute vectors, since none where found on disc")
    # Otherwise compute them
    # docs_by_pmid = {
    #     pmid: " ".join([w for w in bow if "[mh]" not in w])
    #     for pmid, bow in load_bow(total_docs).items()
    # }

    # texts = list(docs_by_pmid.values())   # CountVectorizer expects a list of strings

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x, 
        preprocessor=lambda x: x,
        token_pattern=None,
        binary=True,
        # stop_words="english",
        min_df=min_df,
        max_df=max_df
    )
    
    X = vectorizer.fit_transform(list(bow.values()))
    feature_names = vectorizer.get_feature_names_out()

    # Save results
    with open(X_path, "wb") as f:
        pickle.dump(X, f)
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)

    print("Done laoding vectors")
    return X, ordered_pmids, feature_names

def load_qrels_from_rankings(ranking_files, positive_selection_conf):
    """
    Load relevant and non-relevant PMIDs from ranking .npz files.
    Top 100 PMIDs are relevant (1), PMIDs below rank 200 are non-relevant (0).
    Returns a dict: qrels[review_id] = {1: relevant_pmids, 0: non_relevant_pmids}
    """
    qrels_by_query_id = {}

    for rankings_file in ranking_files:
        arr = np.load(rankings_file)
        pmids = arr["ids"]  # numpy array

        # Extract review ID from filename (remove .npz)
        review_id = Path(rankings_file).stem

        qrels_by_query_id[review_id] = {}
        if positive_selection_conf["type"] == "abs":
            num_pos, num_neutral = positive_selection_conf["num_pos"], positive_selection_conf["num_neutral"]
            qrels_by_query_id[review_id]["pos"] = pmids[:num_pos].tolist()     # relevant
            # qrels_by_query_id[review_id]["neg"] = pmids[num_pos+num_neutral:].tolist()     # non-relevant
            qrels_by_query_id[review_id]["neutral"] = pmids[num_pos:num_pos+num_neutral].tolist()     # non-relevant
        else:
            raise NotImplementedError("Not implemented yet. positive_selection_conf['type']=", positive_selection_conf["type"])
    return qrels_by_query_id

def generate_labels_old(qrels, ordered_pmids, sample_prob=1.0):
    keep_indices = []
    labels = []
    neutral_pmids = set(qrels["neutral"])
    pos_pmids = set(qrels["pos"])
    rand = random.random
    
    for i, pmid in enumerate(ordered_pmids):
        if pmid in neutral_pmids:
            continue
        
        if pmid in pos_pmids:
            labels.append(1)
        else:
            if sample_prob < 1.0 and rand() > sample_prob:
                continue
            labels.append(0)
        keep_indices.append(i)
    return keep_indices, labels

def generate_labels_and_sample_weights(
    ordered_pmids,
    sorted_ids,
    k,
    max_weight:float=1.5,
    num_positives=None,
):
    if isinstance(k, float):
        top_k = math.ceil(approximate_y(TOP_K[0.7][0], TOP_K[0.7][1], num_positives) * k)
    N = len(ordered_pmids)

    # map pmid -> index in X
    pmid_to_index = {
        str(pmid): i
        for i, pmid in enumerate(ordered_pmids)
    }

    # defaults: very irrelevant (set to max weight)
    y = np.zeros(N, dtype=np.int8)
    
    max_weight = int(max_weight*100)
    min_weight = 100
    avg_weight = int((max_weight + min_weight) / 2)
    sample_weight = np.full(N, avg_weight, dtype=np.int32) # negatives get average weight

    # iterate only over ranked PMIDs
    for r, pmid in enumerate(sorted_ids):
        if r >= 3 * top_k:
            break
        
        pmid = str(pmid)
        idx = pmid_to_index[pmid]

        # top ramp (100 -> 0)
        if r < top_k:
            y[idx] = 1
            sample_weight[idx] = max(1, round(
                (max_weight-min_weight) * (1 - r / top_k) + min_weight
            ))
        # flat zero region
        elif r < 3 * top_k:
            y[idx] = 0
            sample_weight[idx] = 0
        # bottom ramp (0 -> 100)
        else: # 3k <= r < 4k
            y[idx] = 0
            sample_weight[idx] = max(1, round(
                avg_weight * ((r - 2 * top_k) / top_k)
            ))

    return y, sample_weight, top_k

def load_bow(**bow_args):
    bow_by_pmid = {}
    mesh = bow_args["mesh"]
    with open(bag_of_words_path(**bow_args), "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["id"]
            if mesh:
                bow_by_pmid[pmid] = entry["bow"]
            else:
                bow_by_pmid[pmid] = [w for w in entry["bow"] if not "[mh]" in w]
    return bow_by_pmid

def load_statistics_data_rf(filter_vars=None):
    """
    Load and aggregate JSONL experiment results by file.

    Args:
        folder (str): Path containing JSONL result files.
        model (str): Model name to filter filenames, e.g., "GreedyORDecisionTree".
        filter_vars (dict, optional): Example: {'n_docs': '50k'} to filter filenames.

    Returns:
        pd.DataFrame: Averaged metrics per file and associated hyperparameters.
    """
    input_folder = statistics_base_path()
    records = []
    for jsonl_file in input_folder.glob("*/results_rf.jsonl"):
        config_file = jsonl_file.parent / "config.json"
        with config_file.open("r", encoding="utf-8") as f:
            conf = json.load(f)


        params = {
            "file": str(jsonl_file.parent.name),
            "max_depth": int(conf["model_args"]["max_depth"]),
            "min_samples_split": int(conf["model_args"]["min_samples_split"]),
            "min_impurity_decrease_start": int(conf["model_args"]["min_impurity_decrease_range_start"]),
            "min_impurity_decrease_end": int(conf["model_args"]["min_impurity_decrease_range_end"]),
            "top_k_or_candidates": int(conf["model_args"]["top_k_or_candidates"]),
            "class_weight": str(conf["model_args"]["class_weight"]),
            "total_docs": int(conf["total_docs"]),
            "min_df": int(conf["min_df"]),
            "max_df": float(conf["max_df"]),
            "mesh": bool(conf["mesh"]),
        }

        # Optional filter for other parameters in filename
        if filter_vars and not all(conf[k] == v for k, v in filter_vars.items()):
            continue

        file_records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                file_records.append(data)
                data["f1"] = (
                    2
                    * data["precision"]
                    * data["recall"]
                    / (data["precision"] + data["recall"])
                )
                pretty = data.get("pretty_print", "")
                data["leafs"] = pretty.count("class")
                data["ORs"] = pretty.count("OR")
                data["IFs"] = pretty.count("if")

        if not file_records:
            continue

        df_file = pd.DataFrame(file_records)
        mean_metrics = df_file.mean(numeric_only=True).to_dict()
        records.append({**params, **mean_metrics})

    if not records:
        print("No matching files or records found.")
        return pd.DataFrame()

    return pd.DataFrame(records)

def load_statistics_data(input_folder, filter_vars=None, qg=True, metrics=None):
    """
    Load and aggregate JSONL experiment results by file.

    Args:
        folder (str): Path containing JSONL result files.
        model (str): Model name to filter filenames, e.g., "GreedyORDecisionTree".
        filter_vars (dict, optional): Example: {'n_docs': '50k'} to filter filenames.

    Returns:
        pd.DataFrame: Averaged metrics per file and associated hyperparameters.
    """
    # i = 0
    # for results_file_rf in input_folder.glob("*/results_rf.jsonl"):
    #     for results_file_qg in results_file_rf.parent.glob("*/results_qg.jsonl"):
    #         i+=1
    # print(i) 
    # exit(0)
    records = []
    for results_file_rf in input_folder.glob("*/rf_results.jsonl"):
        
        config_file_rf = results_file_rf.parent / "rf_config.json"
        with config_file_rf.open("r", encoding="utf-8") as f_rf:
            conf_rf = json.load(f_rf)
            
        # Optional filter for other parameters in filename
        if filter_vars and not all(conf_rf.get(k, v) == v for k, v in filter_vars.items()):
            continue
            
        params_rf = {
            "file": str(results_file_rf.parent.name),
            "max_depth": int(conf_rf["model_args"]["max_depth"]),
            "min_samples_split": int(conf_rf["model_args"]["min_samples_split"]),
            "min_impurity_decrease_start": float(conf_rf["model_args"]["min_impurity_decrease_range_start"]),
            "min_impurity_decrease_end": float(conf_rf["model_args"]["min_impurity_decrease_range_end"]),
            "top_k_or_candidates": int(conf_rf["model_args"]["top_k_or_candidates"]),
            "class_weight": str(conf_rf["model_args"]["class_weight"]),
            "min_df": int(conf_rf["min_df"]),
            "max_df": float(conf_rf["max_df"]),
            "mesh": bool(conf_rf["mesh"]),
        }
        
        file_records_rf = []
        with results_file_rf.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["query_id"] not in EVAL_QUERY_IDS:
                    continue
                data = {f"{k}_rf" if not k.endswith("_rf") else k: v for k, v in data.items()}
                file_records_rf.append(data)
                
        samples_rf = len(file_records_rf) 

        if not file_records_rf:
            continue
        df_file_rf = pd.DataFrame(file_records_rf)
        df_file_rf["f1_rf"] = 2 * df_file_rf["precision_rf"] * df_file_rf["recall_rf"] / (df_file_rf["precision_rf"] + df_file_rf["recall_rf"])
        df_file_rf["f3_rf"] = (1 + 3**2) * (
            df_file_rf["precision_rf"] * df_file_rf["recall_rf"]
        ) / (3**2 * df_file_rf["precision_rf"] + df_file_rf["recall_rf"])
        mean_metrics_rf = df_file_rf.mean(numeric_only=True).to_dict()
        
        if qg: 
            for results_file_qg in results_file_rf.parent.glob("*/qg_results.jsonl"):
                config_file_qg = results_file_qg.parent / "qg_config.json"
                with config_file_qg.open("r", encoding="utf-8") as f_qg:
                    conf_qg = json.load(f_qg)
                    
                if filter_vars and not all(conf_qg.get(k, v) == v for k, v in filter_vars.items()):
                    continue
                params_qg = {
                    "optimization_metric": conf_qg["optimization_metric"],
                    "constraint": conf_qg["constraint"]["metric"] + "=" + str(conf_qg["constraint"]["value"]) if conf_qg["constraint"] else None#["metric"],
                    # "constraint_value": conf_qg["constraint"]["value"],
                }
                
                file_records_qg = []
                with results_file_qg.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: # skip empty or whitespace-only lines
                            continue
                        data = json.loads(line)
                        if data["query_id"] not in EVAL_QUERY_IDS:
                            continue
                        
                        data = {f"{k}_qg" if not k.endswith("_qg") else k: v for k, v in data.items()}
                        if not "query_size_qg" in data:
                            continue
                        for k, v in data["query_size_qg"].items():
                            data[f"query_size_{k}_qg"] = v
                        file_records_qg.append(data)
                samples_qg = len(file_records_qg)
                
                if not file_records_qg:
                    continue
                df_file_qg = pd.DataFrame(file_records_qg)
                df_file_qg["pubmed_f1_qg"] = 2 * df_file_qg["pubmed_precision_qg"] * df_file_qg["pubmed_recall_qg"] / (df_file_qg["pubmed_precision_qg"] + df_file_qg["pubmed_recall_qg"])
                df_file_qg["pubmed_f3_qg"] = (1 + 3**2) * (
                    df_file_qg["pubmed_precision_qg"] * df_file_qg["pubmed_recall_qg"]
                ) / (3**2 * df_file_qg["pubmed_precision_qg"] + df_file_qg["pubmed_recall_qg"])

                mean_metrics_qg = df_file_qg.mean(numeric_only=True).to_dict()
                
                data_dict = {**params_rf, 
                             **params_qg, 
                             **mean_metrics_qg, 
                             **mean_metrics_rf,
                             "samples_qg": samples_qg,
                             "samples_rf": samples_rf,
                             }
                
                if all([m[0] in data_dict.keys() for m in metrics]):
                    records.append(data_dict)
                else:
                    print("skipping file", results_file_rf)
        else:
            data_dict = {**params_rf, 
                         **mean_metrics_rf,
                         "samples_rf": samples_rf,
                         }
            if all([m[0] in data_dict.keys() for m in metrics]):
                records.append(data_dict)
            else:
                print("skipping file", results_file_rf)

    if not records:
        print("No matching files or records found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    

    params = list(params_rf.keys())
    if qg:
        params = list(set(params) | set(params_qg.keys()))
    return df, params

def review_id_to_dataset(review_id):
    # there is some overlap between my custom created TAR2017/TAR2018 and sr_updates 
    # -> we have to check first whether review_id is in review_id in CSMED_COCHRANE_REVIEWS["tar2019"]
    review_id = str(review_id)
    if review_id in CSMED_COCHRANE_REVIEWS["tar2019"]:
        if review_id in review_id in TAR2017_TRAIN:
            return "tar2017", "TRAIN", 2017
        if review_id in review_id in TAR2017_TEST:
            return "tar2017", "TEST", 2017
        if review_id in review_id in TAR2018_TEST:
            return "tar2018", "TEST", 2018
        if review_id in review_id in TAR2018_TEST:
            return "tar2018", "TEST", 2018
        return "tar2019", None, 2019
    if review_id in CSMED_COCHRANE_REVIEWS["sigir2017"]:
        return "sigir2017", None, 2017
    if review_id in CSMED_COCHRANE_REVIEWS["sr_updates"]:
        return "sr_updates", None, 2019
    
    return "unknown", None, -1

