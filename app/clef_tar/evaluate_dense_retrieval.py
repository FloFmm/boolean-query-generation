import os
import re
import torch
import numpy as np
from typing import Dict, List, Set
from app.pubmed.retrieval import search_pubmed_med_cpt, load_pubmed_embeddings

QRELS_PATH = "tar/testing/qrels/qrel_abs_test.txt"
TOPICS_DIR = "tar/testing/topics"
EMBEDDINGS_DIR = "boolean-query-generation/data/pubmed/embeddings"


def filter_qrels_to_embeddings(qrels, valid_pmids):
    """
    Keep only qrel entries whose PMIDs exist in the embeddings.
    Skip topics where all positives are removed.
    Print statistics on positives and negatives before and after filtering.
    """
    filtered_qrels = {}
    for topic, docs in qrels.items():
        # Count positives/negatives before filtering
        pos_before = sum(1 for rel in docs.values() if rel)
        neg_before = sum(1 for rel in docs.values() if not rel)

        # Filter PMIDs to only those in embeddings
        filtered_docs = {pmid: rel for pmid, rel in docs.items() if pmid in valid_pmids}

        # Count positives/negatives after filtering
        pos_after = sum(1 for rel in filtered_docs.values() if rel)
        neg_after = sum(1 for rel in filtered_docs.values() if not rel)

        if pos_after == 0:
            print(
                f"⚠️  Topic {topic} skipped: lost all positives after filtering "
                f"(Pos: {pos_before} → 0, Neg: {neg_before} → {neg_after})"
            )
            continue  # Skip this topic entirely

        # Otherwise, include topic and show stats
        filtered_qrels[topic] = filtered_docs
        print(
            f"ℹ️  Topic {topic}: Positives {pos_before} → {pos_after}, "
            f"Negatives {neg_before} → {neg_after}"
        )
    return filtered_qrels


def parse_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            topic, _, docid, rel = line.strip().split()
            rel = int(rel)
            if topic not in qrels:
                qrels[topic] = {}
            qrels[topic][docid] = rel
    return qrels


def parse_topics(dir_path):
    topics = {}
    for filename in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, filename)
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
                topic_match = re.search(r"Topic:\s*(\S+)", content)
                title_match = re.search(r"Title:\s*(.*)\n*Query", content)
                if topic_match and title_match:
                    topic_id = topic_match.group(1)
                    topic_text = title_match.group(1).strip().replace("\n", " ")
                    topics[topic_id] = topic_text
    return topics


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(retrieved_k) if len(retrieved_k) > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0


def evaluate_topics(
    topics: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    all_pmids: List[str],
    all_embs: np.ndarray,
    search_fn,
    top_k: int = 1000,
    device: str = "cuda",
):
    """
    Evaluate retrieval results for a set of topics without using pytrec_eval.

    Args:
        topics: Dict of topic_id -> title
        qrels: Dict of topic_id -> {pmid: relevance}
        all_pmids: List of PMIDs available in embeddings
        all_embs: Corresponding embeddings
        search_fn: Function to search PMIDs given a query, embeddings, top_k, device
        top_k: Number of top results to consider
        device: Device for search_fn ("cuda"/"cpu")

    Returns:
        metrics_all: Dict of topic_id -> metrics dict
        mean_metrics: Dict of metric -> mean value across topics
    """

    metrics_all = {}

    for i, (topic_id, title) in enumerate(topics.items(), start=1):
        relevant_ids_set = set(id for id, rel in qrels.get(topic_id, {}).items() if rel)
        if not relevant_ids_set:
            continue
        else:
            print(f"\n[{i}/{len(topics)}] Evaluating topic {topic_id}: {title}...")

        retrieved_ids = search_fn(title, all_pmids, all_embs, top_k, device)
        print(f"  Retrieved {len(retrieved_ids)} results")
        true_positives = sum(
            1 for doc in retrieved_ids[:top_k] if doc in relevant_ids_set
        )
        retrieved_at_k = len(retrieved_ids[:top_k])
        relevant_total = len(relevant_ids_set)

        precision = true_positives / retrieved_at_k if retrieved_at_k > 0 else 0
        recall = true_positives / relevant_total if relevant_total > 0 else 0

        # Precision@K for docs that are also in qrels[topic_id]
        retrieved_qrels = [doc for doc in retrieved_ids if doc in qrels[topic_id]]
        true_positives_qrels = sum(
            1 for doc in retrieved_qrels[:top_k] if doc in relevant_ids_set
        )
        precision_qrels = (
            true_positives_qrels / len(retrieved_qrels[:top_k])
            if retrieved_qrels[:top_k]
            else 0
        )

        metrics = {
            f"P_{top_k}": precision,
            f"recall_{top_k}": recall,
            f"P_{top_k}_qrels": precision_qrels,
            "true_positives": true_positives,
            "relevant_total": relevant_total,
        }
        metrics_all[topic_id] = metrics

        print("  Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

    # Compute mean metrics across topics
    mean_metrics = {
        metric: np.mean([m.get(metric, 0.0) for m in metrics_all.values()])
        for metric in metrics_all[next(iter(metrics_all))]
    }

    print("\n✅ Evaluation complete.")
    print("\n=== Final Mean Metrics ===")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")
    print(
        f"global_P_{top_k}: {sum(metric['true_positives'] for metric in metrics_all.values()) / (len(metrics_all) * top_k):.4f}"
    )
    print(
        f"global_recall_{top_k}: {sum(metric['true_positives'] for metric in metrics_all.values()) / sum(metric['relevant_total'] for metric in metrics_all.values()):.4f}"
    )

    return metrics_all, mean_metrics


def main():
    # 1️⃣ Load topics and qrels
    topics = parse_topics(TOPICS_DIR)  # topic_id -> title
    qrels = parse_qrels(QRELS_PATH)  # topic_id -> {pmid: relevance}

    # 2️⃣ Load all PMIDs and embeddings
    print("Loading valid PMIDs from embeddings...")
    all_pmids, all_embs = load_pubmed_embeddings(EMBEDDINGS_DIR)
    print(f"Found {len(all_pmids):,} PMIDs in embeddings.\n")

    # 3️⃣ Filter qrels to PMIDs present in embeddings
    qrels = filter_qrels_to_embeddings(qrels, set(all_pmids))
    # 4️⃣ Set evaluation parameters
    top_k = 10
    device = "cuda"

    # 5️⃣ Define a wrapper for your search function
    def search_fn(query, pmids, embeddings, top_k, device):
        return search_pubmed_med_cpt(query, pmids, embeddings, top_k, device)

    # 6️⃣ Evaluate
    metrics_all, mean_metrics = evaluate_topics(
        topics=topics,
        qrels=qrels,
        all_pmids=all_pmids,
        all_embs=all_embs,
        search_fn=search_fn,
        top_k=top_k,
        device=device,
    )

    # 7️⃣ Optionally, save results
    # import json
    # with open("metrics_all.json", "w") as f:
    #     json.dump(metrics_all, f, indent=2)
    # with open("mean_metrics.json", "w") as f:
    #     json.dump(mean_metrics, f, indent=2)


if __name__ == "__main__":
    main()
