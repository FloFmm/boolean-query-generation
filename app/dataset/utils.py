import json
import os
import numpy as np
import re
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

ABBREVIATIONS = {
    "total_docs": "d",
    "min_df": "mindf",
    "max_df": "maxdf",
    "positive_selection_conf": "psc",
    "mesh": "mesh",
    "optimization": "om", 
    "constraint": "c", 
    "ret_config": "rc",
}
ABBREVIATIONS_REV = {v: k for k, v in ABBREVIATIONS.items()}

EVAL_QUERY_IDS = ["CD011602", "CD011926", "CD010225", "CD003137", "CD002069", "CD011724", "CD010633", "CD007497", "CD011549", "CD007103", "CD010411", "CD011447", "CD009925", "CD000384", "CD009669", "CD009780", "CD010387", "CD010653", "CD004288", "CD011732", "CD007379", "CD010139", "CD011472", "CD012009", "CD012216", "CD008366", "CD003344", "CD006342", "CD010685", "CD005055", "CD010226", "CD008760", "CD008170", "CD002898", "CD006995", "CD011515", "CD009782", "CD006839", "CD002115", "CD009784"]

def abbreviate_params(**kwargs) -> str:
    """
    Converts keyword parameters into the "key=value" format
    using the ABBREVIATIONS dict.
    """
    parts = []
    for full, value in kwargs.items():
        abbr = ABBREVIATIONS.get(full, full)  # fallback: keep same
        parts.append(f"{abbr}={value}")
    return ",".join(parts)

def data_base_path():
    return "../systematic-review-datasets/data"

def statistics_base_path():
    return Path("../boolean-query-generation/data/statistics/csmed")

def bag_of_words_path(total_docs):
    return Path(f"{data_base_path()}/bag_of_words/bag_of_words,{abbreviate_params(total_docs=total_docs)}.jsonl")

def synonym_map_path(total_docs):
    return Path(f"../systematic-review-datasets/data/bag_of_words/synonym_map,{abbreviate_params(total_docs=total_docs)}.jsonl")

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

def load_synonym_map(total_docs):
    with open(synonym_map_path(total_docs), "r", encoding="utf-8") as f:
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

def load_vectors(total_docs: int, min_df: int, max_df: int, mesh: bool):
    X_path = vectors_path(total_docs=total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
    features_path = faeature_names_path(total_docs=total_docs, min_df=min_df, max_df=max_df, mesh=mesh)
    
    bow = load_bow(total_docs=total_docs, mesh=mesh)
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

def generate_labels(qrels, ordered_pmids):
    keep_indices = []
    labels = []
    for i, pmid in enumerate(ordered_pmids):
        if pmid in qrels["neutral"]:
            continue
        if pmid in qrels["pos"]:
            labels.append(1)
        else:
            labels.append(0)
        keep_indices.append(i)
    return keep_indices, labels

def load_bow(total_docs: int, mesh: bool = True):
    bow_by_pmid = {}
    with open(bag_of_words_path(total_docs), "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["id"]
            if mesh:
                bow_by_pmid[pmid] = entry["bow"]
            else:
                bow_by_pmid[pmid] = [w for w in entry["bow"] if not "[mh]" in w]
    return bow_by_pmid

def load_statistics_data_dt(filter_vars=None):
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
    for jsonl_file in input_folder.glob("*/results_dt.jsonl"):
        config_file = jsonl_file.parent / "config.json"
        with config_file.open("r", encoding="utf-8") as f:
            conf = json.load(f)


        params = {
            "file": str(jsonl_file.parent.name),
            "max_depth": int(conf["model_args"]["max_depth"]),
            "min_samples_split": int(conf["model_args"]["min_samples_split"]),
            "min_impurity_decrease_start": int(conf["model_args"]["min_impurity_decrease_range"][0]),
            "min_impurity_decrease_end": int(conf["model_args"]["min_impurity_decrease_range"][1]),
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

def load_statistics_data(filter_vars=None, qg=True, metrics=None):
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
    
    # i = 0
    # for results_file_dt in input_folder.glob("*/results_dt.jsonl"):
    #     for results_file_qg in results_file_dt.parent.glob("*/results_qg.jsonl"):
    #         i+=1
    # print(i) 
    # exit(0)
    records = []
    for results_file_dt in input_folder.glob("*/results_dt.jsonl"):
        
        config_file_dt = results_file_dt.parent / "config.json"
        with config_file_dt.open("r", encoding="utf-8") as f_dt:
            conf_dt = json.load(f_dt)
            
        # Optional filter for other parameters in filename
        if filter_vars and not all(conf_dt.get(k, v) == v for k, v in filter_vars.items()):
            continue
            
        params_dt = {
            "file": str(results_file_dt.parent.name),
            "max_depth": int(conf_dt["model_args"]["max_depth"]),
            "min_samples_split": int(conf_dt["model_args"]["min_samples_split"]),
            "min_impurity_decrease_start": float(conf_dt["model_args"]["min_impurity_decrease_range"][0]),
            "min_impurity_decrease_end": float(conf_dt["model_args"]["min_impurity_decrease_range"][1]),
            "top_k_or_candidates": int(conf_dt["model_args"]["top_k_or_candidates"]),
            "class_weight": str(conf_dt["model_args"]["class_weight"]),
            "total_docs": int(conf_dt["total_docs"]),
            "min_df": int(conf_dt["min_df"]),
            "max_df": float(conf_dt["max_df"]),
            "mesh": bool(conf_dt["mesh"]),
        }
        
        file_records_dt = []
        with results_file_dt.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data["query_id"] not in EVAL_QUERY_IDS:
                    continue
                data = {f"{k}_dt" if not k.endswith("_dt") else k: v for k, v in data.items()}
                file_records_dt.append(data)
                
        samples_dt = len(file_records_dt) 

        if not file_records_dt:
            continue
        df_file_dt = pd.DataFrame(file_records_dt)
        df_file_dt["f1_dt"] = 2 * df_file_dt["precision_dt"] * df_file_dt["recall_dt"] / (df_file_dt["precision_dt"] + df_file_dt["recall_dt"])
        df_file_dt["f3_dt"] = (1 + 3**2) * (
            df_file_dt["precision_dt"] * df_file_dt["recall_dt"]
        ) / (3**2 * df_file_dt["precision_dt"] + df_file_dt["recall_dt"])
        mean_metrics_dt = df_file_dt.mean(numeric_only=True).to_dict()
        
        if qg: 
            for results_file_qg in results_file_dt.parent.glob("*/results_qg.jsonl"):
                config_file_qg = results_file_qg.parent / "config.json"
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
                
                data_dict = {**params_dt, 
                             **params_qg, 
                             **mean_metrics_qg, 
                             **mean_metrics_dt,
                             "samples_qg": samples_qg,
                             "samples_dt": samples_dt,
                             }
                
                if all([m[0] in data_dict.keys() for m in metrics]):
                    records.append(data_dict)
                else:
                    print("skipping file", results_file_dt)
        else:
            data_dict = {**params_dt, 
                         **mean_metrics_dt,
                         "samples_dt": samples_dt,
                         }
            if all([m[0] in data_dict.keys() for m in metrics]):
                records.append(data_dict)
            else:
                print("skipping file", results_file_dt)

    if not records:
        print("No matching files or records found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    

    params = list(params_dt.keys())
    if qg:
        params = list(set(params) | set(params_qg.keys()))
    return df, params
