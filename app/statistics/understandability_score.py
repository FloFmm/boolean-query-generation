from app.config.config import BOW_PARAMS
from app.dataset.utils import load_synonym_map, load_vectors
import os
import json
import re
from statistics import mean
import numpy as np


def extract_terms(pubmed_query: str):
    """
    Split on AND, OR, NOT, (, ) and normalize PubMed field tags.
    """
    tokens = re.split(r"\b(?:AND|OR|NOT)\b|\(|\)", pubmed_query)
    terms = set()
    for t in tokens:
        t = t.replace("[mh:noexp]", "[mh]").replace("[tiab]", "")
        if "[mh]" not in t:
            t = t.replace('"', '')
        t = t.strip()
        if t:
            terms.add(t)
    return terms


def document_counts(words, X, feature_index):
    """
    Vectorized count of how many documents each word appears in.
    Returns a dict: word -> document count
    """
    idxs = [feature_index[w] for w in words if w in feature_index]
    if not idxs:
        return {w: 0 for w in words}
    counts = X[:, idxs].sum(axis=0).A1  # convert sparse matrix to 1D array
    return dict(zip([w for w in words if w in feature_index], counts))


def analyze_qg_results(root_folder, X, feature_names):
    results = {}
    feature_index = {f: i for i, f in enumerate(feature_names)}  # O(1) lookup

    for root, _, files in os.walk(root_folder):
        if "qg_results.jsonl" not in files or "qg_config.json" not in files:
            continue

        config_path = os.path.join(root, "qg_config.json")
        results_path = os.path.join(root, "qg_results.jsonl")

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Only consider term_expansions == False
        if config.get("term_expansions", True):
            continue

        # Collect all terms from this file
        all_terms = set()
        entries = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                query = entry.get("pubmed_query", "")
                terms = extract_terms(query)
                all_terms.update(terms)
                entries.append((query, terms))

        # Get document counts for all terms at once
        term_doc_counts = document_counts(all_terms, X, feature_index)

        doc_counts = []
        for query, terms in entries:
            for term in terms:
                dc = term_doc_counts.get(term, 0)
                print(term)
                if dc <= 0:
                    print(query)
                    print(f"|{term}|")
                    assert False
                doc_counts.append(dc)

        if doc_counts:
            results[root] = {
                "num_terms": len(doc_counts),
                "avg_document_count": mean(doc_counts),
            }

    return results


if __name__ == "__main__":
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    qg_results_path = "data/statistics/optuna/best0"

    stats = analyze_qg_results(qg_results_path, X, feature_names)

    for folder, info in stats.items():
        print(folder)
        print(f"  terms: {info['num_terms']}")
        print(f"  avg document count: {info['avg_document_count']:.2f}")
