import json
import os
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

ABBREVIATIONS = {
    "total_docs": "d",
    "min_df": "mindf",
    "max_df": "maxdf",
    "positive_selection_conf": "psc",
    "mesh": "mesh",
    "optimization_metric": "om", 
    "constraint_metric": "cm", 
    "constraint_value": "cv"
}
ABBREVIATIONS_REV = {v: k for k, v in ABBREVIATIONS.items()}

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
                completed.add(record["id"])
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