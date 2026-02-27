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
from app.config.config import (
    TAR2017_TRAIN,
    TAR2017_TEST,
    TAR2018_TEST,
    CSMED_COCHRANE_REVIEWS,
    TOP_K,
    BOW_PARAMS,
    FIXED_TOP_K,
    COSINE_PCT_THRESHOLD,
)
from app.helper.helper import f_beta

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
    "bootstrap": "boot,",
    "class_weight": "cw",
    "max_depth": "maxd",
    "max_features": "maxf",
    "max_or_features": "mof",
    "max_samples": "maxs",
    "min_impurity_decrease_range_start": "midrs",
    "min_impurity_decrease_range_end": "midre",
    "min_samples_split": "mins",
    "min_weight_fraction_leaf": "mwfl",
    "n_estimators": "ne",
    # "n_jobs": "nj",
    "prefer_pos_splits": "pfs",
    # "random_state":"rs",
    "randomize_max_feature": "rmf",
    "randomize_min_impurity_decrease_range": "rmidr",
    "rank_weight": "rweight",
    "top_k": "k",
    "top_k_type": "ktype",
    "top_k_or_candidates": "tkoc",
    "term_expansions": "te",
    "dont_cares": "dc",
    "cost_factor": "cf",
    "cover_beta": "cb",
    "min_rule_occ": "mro",
    "min_rule_precision": "mrp",
    "min_tree_occ": "mto",
    "pruning_beta": "pb",
    # "verbose": "v",
}

ABBREVIATIONS_REV = {v: k for k, v in ABBREVIATIONS.items()}

EVAL_QUERY_IDS_OLD = [
    "CD011602",
    "CD011926",
    "CD010225",
    "CD003137",
    "CD002069",
    "CD011724",
    "CD010633",
    "CD007497",
    "CD011549",
    "CD007103",
    "CD010411",
    "CD011447",
    "CD009925",
    "CD000384",
    "CD009669",
    "CD009780",
    "CD010387",
    "CD010653",
    "CD004288",
    "CD011732",
    "CD007379",
    "CD010139",
    "CD011472",
    "CD012009",
    "CD012216",
    "CD008366",
    "CD003344",
    "CD006342",
    "CD010685",
    "CD005055",
    "CD010226",
    "CD008760",
    "CD008170",
    "CD002898",
    "CD006995",
    "CD011515",
    "CD009782",
    "CD006839",
    "CD002115",
    "CD009784",
]


def abbreviate_value(value) -> str:
    if isinstance(value, float):
        return format(value, ".6g")

    if isinstance(value, dict):
        items = []
        for k in sorted(value):
            abbr_k = ABBREVIATIONS.get(k, k)
            items.append(f"{abbr_k}={abbreviate_value(value[k])}")
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
    return Path(
        f"{data_base_path()}/bag_of_words/bag_of_words,{abbreviate_params(**args)}.jsonl"
    )


def synonym_map_path(**args):
    """params: total_docs, lower_case, mesh_ancestors, rm_numbers, rm_punct"""
    args = {k: v for k, v in args.items() if k not in ("min_df", "max_df", "mesh")}
    return Path(
        f"../systematic-review-datasets/data/bag_of_words/synonym_map,{abbreviate_params(**args)}.json"
    )


def run_path(run_name, **bow_args):
    return Path(
        os.path.join(
            statistics_base_path(),
            "optuna",
            run_name,
            f"{abbreviate_params(**bow_args)}",
        )
    )


def rf_statistics_path(run_name, **args):
    args = {
        k: v for k, v in args.items() if k not in ("verbose", "random_state", "n_jobs")
    }
    return Path(
        os.path.join(
            run_path(run_name=run_name, **BOW_PARAMS), abbreviate_params(**args)
        )
    )


def qg_statistics_path(run_name, rf_args, qg_args):
    qg_args = {k: v for k, v in qg_args.items() if k not in ("pruning_thresholds")}
    return Path(
        f"{rf_statistics_path(run_name=run_name, **rf_args)}/{abbreviate_params(**qg_args)}"
    )


def statistics_sub_folder_path(model, **args):
    """params: model, total_docs, min_df, max_df, positive_selection_conf, mesh"""
    params = abbreviate_params(**args)
    path = Path(
        os.path.join(
            statistics_base_path(),
            f"{str(model).replace(' ', '')},{params}".replace(" ", ""),
        )
    )

    return path


def faeature_names_path(**args):
    """params: total_docs, min_df, max_df, mesh"""
    return Path(
        f"{data_base_path()}/bag_of_words/feature_names,{abbreviate_params(**args)}.pkl"
    )


def vectors_path(**args):
    """params: total_docs, min_df, max_df, mesh"""
    return Path(
        f"{data_base_path()}/bag_of_words/vectors,{abbreviate_params(**args)}.pkl"
    )


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
        min_df=bow_args["min_df"],
        max_df=bow_args["max_df"],
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


def document_count(word, X, feature_names):
    """
    Returns in how many documents `word` appears.

    Parameters
    ----------
    word : str
        The word to look up.
    X : scipy.sparse matrix
        Binary document-term matrix from CountVectorizer.
    feature_names : array-like
        Feature names from vectorizer.get_feature_names_out().

    Returns
    -------
    int
        Number of documents containing the word.
    """
    try:
        idx = feature_names.tolist().index(word)
    except ValueError:
        return 0  # word not in vocabulary

    # Sum over documents for this word
    return int(X[:, idx].sum())


def ranking_file_path(retriever_name, query_type, total_docs, query_id=None):
    base_dir = Path(
        f"../systematic-review-datasets/data/rankings/"
        f"{retriever_name}/{query_type}/docs={total_docs}"
    )

    if query_id is not None:
        return base_dir / f"{query_id}.npz"
    else:
        return list(base_dir.glob("*.npz"))


def get_sorted_ids(retriever_name, query_type, total_docs, query_id):
    rankings_file = ranking_file_path(retriever_name, query_type, total_docs, query_id)
    if not rankings_file.exists():
        return None
    arr = np.load(rankings_file)
    sorted_ids = arr["ids"]
    scores = arr["scores"]
    return sorted_ids, scores

def select_k_positive_dependent(num_positives: int) -> int:
    """
    Uses your provided approximation function.
    """
    return approximate_y(
        TOP_K[0.7][0],
        TOP_K[0.7][1],
        num_positives,
    )

def select_k_cosine_threshold(scores: np.ndarray, cosine_percentage_threshold) -> int:
    """
    Cosine-threshold-based top-k:
    threshold = 5% lower than the average of top-5 scores.
    """
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return 0

    top5 = valid_scores[:5]
    threshold = (1.0 - cosine_percentage_threshold) * np.mean(top5)

    return min(max(int(np.sum(valid_scores >= threshold)), 50), 3000)


def generate_pseudo_labels_and_sample_weights(
    ordered_pmids,
    sorted_ids,
    k,
    dont_cares,
    max_weight: float = 1.5,
    top_k_type="pos_count",
    num_positives=None,
    sorted_scores=None,
):
    if top_k_type == "pos_count":
        top_k = select_k_positive_dependent(num_positives=num_positives) 
    elif top_k_type == "fixed":
        top_k = FIXED_TOP_K
    elif top_k_type == "cosine":
        top_k = select_k_cosine_threshold(sorted_scores, COSINE_PCT_THRESHOLD)
    else:
        raise Exception
    top_k = math.ceil(top_k* k)

    N = len(ordered_pmids)

    # map pmid -> index in X
    pmid_to_index = {str(pmid): i for i, pmid in enumerate(ordered_pmids)}

    # defaults: very irrelevant (set to max weight)
    y = np.zeros(N, dtype=np.int8)

    max_weight = int(max_weight * 100)
    min_weight = 100
    # avg_weight = int((max_weight + min_weight) / 2) # negatives get average weight (not any more)
    sample_weight = np.full(N, max_weight, dtype=np.int32)

    len_ramp1 = top_k
    len_ramp2 = top_k
    len_dont_cares = int(dont_cares * top_k)
    end_ramp1 = len_ramp1
    start_ramp2 = top_k + len_dont_cares
    end_ramp2 = start_ramp2 + len_ramp2
    # iterate only over ranked PMIDs
    for r, pmid in enumerate(sorted_ids):
        if r >= end_ramp2:
            break

        pmid = str(pmid)
        idx = pmid_to_index[pmid]

        # top ramp 1-k
        if r <= end_ramp1:
            y[idx] = 1
            sample_weight[idx] = max(
                1, round((max_weight - min_weight) * (1 - r / len_ramp1) + min_weight)
            )
        # flat zero region k <= r < k+n*k
        elif r < start_ramp2:
            y[idx] = 0
            sample_weight[idx] = 0
        # bottom ramp (0 -> 100)
        else:  # k + n*k <= r < k + n*k + k
            y[idx] = 0
            sample_weight[idx] = max(
                1,
                round(
                    (max_weight - min_weight) * ((r - start_ramp2) / len_ramp2)
                    + min_weight
                ),
            )

    return y, sample_weight, top_k


def get_positives(review_id, dataset):
    positives = set()
    if review_id in dataset["EVAL"]:
        reviews = dataset["EVAL"]
    else:
        reviews = dataset["TRAIN"]
    for split_name, data in reviews[review_id][
        "data"
    ].items():  # 'train', 'val', 'test' etc.
        for doc in data:
            if int(doc["label"]) == 1:
                positives.add(str(doc["pmid"]))
    return positives


def get_all_review_ids(dataset):
    return set(dataset["EVAL"].keys()) | set(dataset["TRAIN"].keys())


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
                bow_by_pmid[pmid] = [w for w in entry["bow"] if "[mh]" not in w]
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
            "min_impurity_decrease_start": int(
                conf["model_args"]["min_impurity_decrease_range_start"]
            ),
            "min_impurity_decrease_end": int(
                conf["model_args"]["min_impurity_decrease_range_end"]
            ),
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
        if filter_vars and not all(
            conf_rf.get(k, v) == v for k, v in filter_vars.items()
        ):
            continue

        params_rf = {
            "file": str(results_file_rf.parent.name),
            "max_depth": int(conf_rf["model_args"]["max_depth"]),
            "min_samples_split": int(conf_rf["model_args"]["min_samples_split"]),
            "min_impurity_decrease_start": float(
                conf_rf["model_args"]["min_impurity_decrease_range_start"]
            ),
            "min_impurity_decrease_end": float(
                conf_rf["model_args"]["min_impurity_decrease_range_end"]
            ),
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
                # if data["query_id"] not in EVAL_QUERY_IDS:
                #     continue
                data = {
                    f"{k}_rf" if not k.endswith("_rf") else k: v
                    for k, v in data.items()
                }
                file_records_rf.append(data)

        samples_rf = len(file_records_rf)

        if not file_records_rf:
            continue
        df_file_rf = pd.DataFrame(file_records_rf)
        df_file_rf["f1_rf"] = (
            2
            * df_file_rf["precision_rf"]
            * df_file_rf["recall_rf"]
            / (df_file_rf["precision_rf"] + df_file_rf["recall_rf"])
        )
        df_file_rf["f3_rf"] = (
            (1 + 3**2)
            * (df_file_rf["precision_rf"] * df_file_rf["recall_rf"])
            / (3**2 * df_file_rf["precision_rf"] + df_file_rf["recall_rf"])
        )
        mean_metrics_rf = df_file_rf.mean(numeric_only=True).to_dict()

        if qg:
            for results_file_qg in results_file_rf.parent.glob("*/qg_results.jsonl"):
                config_file_qg = results_file_qg.parent / "qg_config.json"
                with config_file_qg.open("r", encoding="utf-8") as f_qg:
                    conf_qg = json.load(f_qg)

                if filter_vars and not all(
                    conf_qg.get(k, v) == v for k, v in filter_vars.items()
                ):
                    continue
                params_qg = {
                    "optimization_metric": conf_qg["optimization_metric"],
                    "constraint": conf_qg["constraint"]["metric"]
                    + "="
                    + str(conf_qg["constraint"]["value"])
                    if conf_qg["constraint"]
                    else None,  # ["metric"],
                    # "constraint_value": conf_qg["constraint"]["value"],
                }

                file_records_qg = []
                with results_file_qg.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:  # skip empty or whitespace-only lines
                            continue
                        data = json.loads(line)
                        # if data["query_id"] not in EVAL_QUERY_IDS:
                        #     continue

                        data = {
                            f"{k}_qg" if not k.endswith("_qg") else k: v
                            for k, v in data.items()
                        }
                        if not "query_size_qg" in data:
                            continue
                        for k, v in data["query_size_qg"].items():
                            data[f"query_size_{k}_qg"] = v
                        file_records_qg.append(data)
                samples_qg = len(file_records_qg)

                if not file_records_qg:
                    continue
                df_file_qg = pd.DataFrame(file_records_qg)
                df_file_qg["pubmed_f1_qg"] = (
                    2
                    * df_file_qg["pubmed_precision_qg"]
                    * df_file_qg["pubmed_recall_qg"]
                    / (
                        df_file_qg["pubmed_precision_qg"]
                        + df_file_qg["pubmed_recall_qg"]
                    )
                )
                df_file_qg["pubmed_f3_qg"] = (
                    (1 + 3**2)
                    * (
                        df_file_qg["pubmed_precision_qg"]
                        * df_file_qg["pubmed_recall_qg"]
                    )
                    / (
                        3**2 * df_file_qg["pubmed_precision_qg"]
                        + df_file_qg["pubmed_recall_qg"]
                    )
                )

                mean_metrics_qg = df_file_qg.mean(numeric_only=True).to_dict()

                data_dict = {
                    **params_rf,
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
            data_dict = {
                **params_rf,
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


def dataset_names(short_name):
    """
    Maps dataset short names to human-readable names.

    Args:
        short_name (str): Short identifier of the dataset.

    Returns:
        str: Human-readable dataset name.
    """
    mapping = {
        "tar2017": "CLEF TAR 2017",
        "tar2018": "CLEF TAR 2018",
        "tar2019": "CLEF TAR 2019",
        "sigir2017": "SIGIR 2017",
        "sr_updates": "SR Updates",
    }
    return mapping.get(short_name, short_name)


def dataset_details_path():
    return Path(
        "../systematic-review-datasets/data/dataset_details/dataset_details.json"
    )


def get_dataset_details() -> dict:
    """
    Load the dataset details dictionary from disk.

    Returns:
        dict: Mapping from query_id -> dataset_details
    """
    path = dataset_details_path()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def get_qg_results(path, min_positive_threshold=None, query_ids=None):
    records = []
    # if path is already a file then only take the data from that file
    files = []
    if os.path.isfile(path) and path.endswith("qg_results.jsonl"):
        files = [path]
    else:        
        files = list(Path(path).glob("**/qg_results.jsonl"))
    for jsonl_path in files:
        # Read meta data for betas
        meta_path = os.path.join(os.path.dirname(jsonl_path), "qg_meta_data.json")
        betas_str = ""
        if os.path.exists(meta_path):
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
                betas = sorted(meta.get("betas", {}).keys(), key=int)
                betas_str = ",".join(map(str, betas))

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["file_path"] = str(jsonl_path)
                data["selection_betas"] = betas_str
                records.append(data)
    df = pd.DataFrame(records)
    # Print sample count
    print(f"{len(df)} samples")
    
    if min_positive_threshold is not None:
        df = df[df["num_positive"] >= min_positive_threshold].copy()
    if query_ids is not None:
        df = df[df["query_id"].isin(query_ids)].copy()
    return df


def find_qg_results_file(base_folder, top_k_type="cosine", betas_key="50"):
    for root, dirs, files in os.walk(base_folder):
        if "qg_results.jsonl" in files and "qg_meta_data.json" in files and "qg_config.json" in files:
            meta_path = os.path.join(root, "qg_meta_data.json")
            config_path = os.path.join(os.path.dirname(root), "rf_config.json")
            with open(meta_path, "r") as f:
                meta_data = json.load(f)
            with open(config_path, "r") as f:
                config_data = json.load(f)
            if (
                "betas" in meta_data and 
                betas_key in meta_data["betas"] and 
                config_data["top_k_type"] == top_k_type
            ):
                return os.path.join(root, "qg_results.jsonl")
    return None 

def get_rf_and_qg_params(base_folder, top_k_type="cosine", betas_key="50"):
    path = Path(find_qg_results_file(base_folder, top_k_type=top_k_type, betas_key=betas_key))

    def _normalize_json_bool_strings(obj):
        """
        Recursively walk `obj` (dict/list/primitive) and convert string
        values that look like boolean/null literals into their Python
        equivalents. This only converts exact string values (case-insensitive)
        'true' -> True, 'false' -> False, 'null'/'none' -> None.
        """
        if isinstance(obj, dict):
            return {_normalize_json_bool_strings(k): _normalize_json_bool_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize_json_bool_strings(v) for v in obj]
        if isinstance(obj, str):
            low = obj.strip().lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in ("null", "none"):
                return None
            return obj
        return obj

    with open(path.parent / "qg_config.json", "r") as f:
        qg_params = json.load(f)
    with open(path.parent.parent / "rf_config.json", "r") as f:
        rf_params = json.load(f)

    qg_params = _normalize_json_bool_strings(qg_params)
    rf_params = _normalize_json_bool_strings(rf_params)

    return rf_params, qg_params

def get_paper_query_examples(paper=None, query_id=None):
    path = Path("data/examples/baseline.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if paper is None:
        return data 
    if paper is not None:
        if query_id is None:
            return data[paper]
        else:
            for example in data[paper]["examples"]:
                if example["query_id"] == query_id:
                    return example

def calc_missing_columns_in_result_df(df):
    df = df.copy()

    # Add dataset column
    df["dataset"] = df["query_id"].apply(lambda qid: review_id_to_dataset(qid)[0])
    # Map tar2017 to tar2018 (tar2017 is part of 2018)
    df.loc[df["dataset"] == "tar2017", "dataset"] = "tar2018"

    # Add num_positive_bucket
    df["num_positive_bucket"] = df["num_positive"].apply(
        lambda x: "\<50" if x < 50 else "\>\=50"
    )

    # Add source_file (last two directory names of file_path)
    df["source_file"] = df["file_path"].apply(
        lambda fp: str(Path(*Path(fp).parts[-3:-1]))
    )

    df["pubmed_f1"] = df.apply(
        lambda row: f_beta(
            precision=row["pubmed_precision"], recall=row["pubmed_recall"], beta=1
        ),
        axis=1,
    )
    df["pubmed_f3"] = df.apply(
        lambda row: f_beta(
            precision=row["pubmed_precision"], recall=row["pubmed_recall"], beta=3
        ),
        axis=1,
    )
    for k in ["paths", "ANDs", "NOTs", "added_ORs", "synonym_ORs"]:
        df[f"query_size_{k}"] = df["query_size"].apply(lambda x: x[k])
    df["all_ORs"] = df["pubmed_query"].apply(lambda x: x.count("OR"))
    df["logical_operators"] = df["pubmed_query"].apply(
        lambda x: x.count("OR") + x.count("AND") + x.count("NOT")
    )
    assert all(df["logical_operators"] == (
        df["query_size_ANDs"]
        + df["query_size_NOTs"]
        + df["query_size_added_ORs"]
        + df["query_size_synonym_ORs"]
        + df["query_size_paths"]
        - 1
    ))
    assert all(df["all_ORs"] == (
        + df["query_size_added_ORs"]
        + df["query_size_synonym_ORs"]
        + df["query_size_paths"]
        - 1
    ))
    return df
