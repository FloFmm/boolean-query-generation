import json
import os
import numpy as np
from pathlib import Path

def data_base_path():
    return "../systematic-review-datasets/data"

def bag_of_words_path(total_docs):
    return f"{data_base_path()}/bag_of_words/bag_of_words_docs={total_docs}.jsonl"

def synonym_map_path(total_docs):
    return f"../systematic-review-datasets/data/bag_of_words/synonym_map_docs={total_docs}.jsonl"

def statistics_file_path(output_path, model, total_docs, min_f_occ, positive_selection_conf):
    file_path = Path(
        os.path.join(
                output_path,
                f"{str(model).replace(' ', '')},d={total_docs},mfo={min_f_occ},psc={positive_selection_conf}.jsonl".replace(' ', ''),
            )
        )
    return file_path

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
            qrels_by_query_id[review_id]["neg"] = pmids[num_pos+num_neutral:].tolist()     # non-relevant
            qrels_by_query_id[review_id]["neutral"] = pmids[num_pos:num_pos+num_neutral].tolist()     # non-relevant
        else:
            raise NotImplementedError("Not implemented yet. positive_selection_conf['type']=", positive_selection_conf["type"])
    return qrels_by_query_id

def load_bow(total_docs: int):
    bow_by_pmid = {}
    with open(bag_of_words_path(total_docs), "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pmid = entry["id"]
            bow_by_pmid[pmid] = entry["bow"]
    return bow_by_pmid