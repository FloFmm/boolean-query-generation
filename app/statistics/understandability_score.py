import os

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here

from app.config.config import BOW_PARAMS
from app.dataset.utils import load_synonym_map, load_vectors
import json
import re
from statistics import mean, median
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re

import random
from itertools import combinations, product

import random
from itertools import combinations
from statistics import mean, median
from sentence_transformers import SentenceTransformer, util
from gensim.models import KeyedVectors

def remove_tags(w):
    w = (
        w.replace("[mh]", "")
        .replace("[mh:noexp]", "")
        .replace("[tiab]", "")
        .replace('"', "")
        .replace("(", "")
        .replace(")", "")
    )
    return w

def compute_similarity_stats(
    entries,
    feature_names,
    max_queries: int = 10,
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    random_pairs: int = 10000,
    use_word2vec: bool = False,
    wv_model=None,  # gensim KeyedVectors model
):
    """
    Calculate average similarity for three cases:
      1) inner_or: two words from the same inner-OR list
      2) and_list: two words from different inner-OR lists but within the same AND-list (same outer-OR element)
      3) cross_query: two words from different queries (entries)
    """
    rng = random.Random(42)
    # --- 1) collect candidate pairs (but don't explode memory) ---
    inner_candidates = set()  # (w1, w2)
    and_candidates = set()  # (w1, w2)

    for i, (_, query_list, _) in enumerate(entries):
        if i >= max_queries:
            break
        # collect unique words per query (flat)
        for and_list in query_list:  # outer_or = AND-list
            for inner_or in and_list:  # inner_or = list of words (the inner OR)
                assert inner_or
                # inner_or candidate pairs (within same inner OR)
                if len(inner_or) >= 2:
                    inner_candidates.update(combinations(inner_or, 2))

        # and_list candidate pairs: within same outer_or (AND-list) but different inner_or lists
        for and_list in query_list:
            # take all pairs of distinct inner_or lists
            if len(and_list) < 2:
                continue
            # produce cross-list pairs (one from list i, one from list j)
            for i in range(len(and_list)):
                for j in range(i + 1, len(and_list)):
                    for a in and_list[i]:
                        for b in and_list[j]:
                            and_candidates.add((a, b))

    inner_candidates = {(remove_tags(a), remove_tags(b)) for (a, b) in inner_candidates}
    and_candidates = {(remove_tags(a), remove_tags(b)) for (a, b) in and_candidates}
    random_candidates = set()
    for _ in range(random_pairs):
        a, b = rng.sample(sorted(feature_names), 2)
        random_candidates.add((remove_tags(a), remove_tags(b)))
    # --- 3) prepare embedding lookup (encode all unique words once) ---
    needed_words = set()
    for a, b in inner_candidates | and_candidates | random_candidates:
        needed_words.add(a)
        needed_words.add(b)

    if use_word2vec:
        if wv_model is None:
            raise ValueError("Must provide `wv_model` when use_word2vec=True")
        # Word2Vec embeddings are directly accessible
        
        idx = {w: i for i, w in enumerate(needed_words)}
        embeddings = {
            w: wv_model[w] if w in wv_model else np.zeros(wv_model.vector_size)
            for w in needed_words
        }
        
        # def phrase_mean_vector(w, wv_model):
        #     w = w.split("/")[0]
        #     w = w.replace(",", "")
        #     words = w.lower().split()
        #     if any([word not in wv_model for word in words]):
        #         return None   # true OOV
        #     vecs = [wv_model[word] for word in words if word in wv_model]
        #     return np.mean(vecs, axis=0)

        # embeddings = {}
        # for w in needed_words:
        #     vec = phrase_mean_vector(w, wv_model)
        #     if vec is not None:
        #         embeddings[w] = vec
        #     else:
        #         embeddings[w] = np.zeros(wv_model.vector_size)
    else:
        model = SentenceTransformer(model_name)
        unique_list = list(needed_words)
        embeddings_tensor = model.encode(
            unique_list, convert_to_tensor=True, show_progress_bar=False
        )
        idx = {w: i for i, w in enumerate(unique_list)}

    # --- 4) compute similarities for each sampled pair ---
    def compute_similarities(pairs):
        sims = []
        for a, b in pairs:
            if use_word2vec:
                vec_a = embeddings[a]
                vec_b = embeddings[b]
                # cosine similarity
                if np.linalg.norm(vec_a) == 0:
                    # print("oov", a)
                    continue
                if np.linalg.norm(vec_b) == 0:
                    # print("oov", b)
                    continue
                # print("contained", a)
                # print("contained", b)
                score = float(
                    np.dot(vec_a, vec_b)
                    / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                )
            else:
                ia = idx.get(a)
                ib = idx.get(b)
                if ia is None or ib is None:
                    continue
                score = util.cos_sim(
                    embeddings_tensor[ia], embeddings_tensor[ib]
                ).item()
            sims.append(score)
        return sims

    inner_sims = compute_similarities(inner_candidates)
    and_sims = compute_similarities(and_candidates)
    random_sims = compute_similarities(random_candidates)

    # --- 5) summarize ---
    def summarize(sim_list):
        if not sim_list:
            return {
                "count": 0,
                "avg": None,
                "median": None,
                "min": None,
                "max": None,
                "raw": [],
            }
        return {
            "count": len(sim_list),
            "avg": mean(sim_list),
            "median": median(sim_list),
            "min": min(sim_list),
            "max": max(sim_list),
            # 'raw': sim_list,   # remove this if you don't want the raw scores included
        }

    return {
        "inner_or": summarize(inner_sims),
        "and_list": summarize(and_sims),
        "random_pairs": summarize(random_sims),
    }


def query_str_to_list(formula: str):
    """
    Transform a boolean formula string into a list of CNFs (list of list of words).
    Each CNF corresponds to one OR-clause.
    """

    # Split top-level ORs
    formula = formula.replace("[tiab]", "")
    formula = formula.replace("[mh:noexp]", "[mh]")
    or_clauses = re.split(r"\)\s+OR\s+\(", formula, flags=re.IGNORECASE)
    for r_str in [") OR ", " OR ("]:
        new_or_clauses = []
        for oc in or_clauses:
            new_or_clauses += oc.split(r_str)
        or_clauses = new_or_clauses

    result = []

    for clause in or_clauses:
        # Remove surrounding parentheses
        clause = clause.strip()
        if clause.startswith("(") and clause.endswith(")"):
            clause = clause[1:-1]

        # Split by AND
        and_parts = clause.split(" AND ")
        all_end_parts = []
        for part in and_parts:
            all_end_parts += part.split(" NOT ")
        and_parts = all_end_parts

        cnf_clause = []

        for part in and_parts:
            part = part.strip(" ()")
            or_parts = part.split(" OR ")
            assert (
                not any(["OR" in op for op in or_parts])
                and not any(["AND" in op for op in or_parts])
                and not any(["NOT" in op for op in or_parts])
            )
            new_or_parts = []
            for op in or_parts:
                if "[mh]" not in op:
                    op = op.replace('"', "")
                new_or_parts.append(op)
            cnf_clause.append(new_or_parts)

        result.append(cnf_clause)

    return result


# def extract_terms(pubmed_query: str):
#     """
#     Split on AND, OR, NOT, (, ) and normalize PubMed field tags.
#     """
#     tokens = re.split(r"\b(?:AND|OR|NOT)\b|\(|\)", pubmed_query)
#     terms = set()
#     for t in tokens:
#         t = t.replace("[mh:noexp]", "[mh]").replace("[tiab]", "")
#         if "[mh]" not in t:
#             t = t.replace('"', '')
#         t = t.strip()
#         if t:
#             terms.add(t)
#     return terms


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


def analyze_qg_results(root_folder):
    all_terms = set()
    entries = []
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

        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                query = entry.get("pubmed_query", "")
                query_list = query_str_to_list(query)
                terms = [x for z in query_list for y in z for x in y]  # flatten
                all_terms.update(terms)
                entries.append((query, query_list, terms))
    return entries, all_terms


def document_count_stats(entries, all_terms, X, feature_names):
    results = {}
    feature_index = {f: i for i, f in enumerate(feature_names)}  # O(1) lookup

    # Get document counts for all terms at once
    term_doc_counts = document_counts(all_terms, X, feature_index)

    doc_counts = []
    for query, query_list, terms in entries:
        for term in terms:
            dc = term_doc_counts.get(term, 0)
            if dc <= 0:
                print(query)
                print(f"|{term}|")
                assert False
            doc_counts.append(dc)

    if doc_counts:
        results = {
            "num_terms": len(doc_counts),
            "avg_document_count": mean(doc_counts),
            "median_document_count": median(doc_counts),
            "min_document_count": min(doc_counts),
            "max_document_count": max(doc_counts),
        }

    return results


def get_word_similarity(word1, word2, model_name="pritamdeka/S-PubMedBert-MS-MARCO"):
    """
    Calculates semantic similarity between two biomedical words using a PubMedBERT-based transformer.
    """
    # Load the model
    model = SentenceTransformer(model_name)

    # Encode the words into embeddings
    embedding1 = model.encode(word1, convert_to_tensor=True)
    embedding2 = model.encode(word2, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_score = util.cos_sim(embedding1, embedding2)

    return float(cosine_score)


if __name__ == "__main__":
    wv_model = KeyedVectors.load_word2vec_format("../systematic-review-datasets/data/word2vec/bio_embedding_intrinsic", binary=True)
    
    vocab = list(wv_model.key_to_index.keys())
    print("loaded wv model")
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    term_expansions = load_synonym_map(**BOW_PARAMS)
    qg_results_path = "data/statistics/optuna/best0"

    entries, all_terms = analyze_qg_results(root_folder=qg_results_path)
    
    sim_stats = compute_similarity_stats(
        feature_names=feature_names,
        entries=entries,
        max_queries=10_000_000,
        random_pairs=100_000,
        use_word2vec=True,
        wv_model=wv_model
    )
    print(sim_stats)

    stats = document_count_stats(entries, all_terms, X, feature_names)
    print(stats)
