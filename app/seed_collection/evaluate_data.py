from collections import defaultdict


def analyze_qrels(file_path):
    query_docs = defaultdict(list)

    # Parse file lines
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # skip malformed lines
            query_id, doc_id, relevance = parts[0], parts[2], int(parts[3])
            query_docs[query_id].append(relevance)

    # Compute stats
    num_queries = len(query_docs)
    total_pairs = sum(len(docs) for docs in query_docs.values())

    positives = [sum(docs) for docs in query_docs.values()]
    negatives = [len(docs) - sum(docs) for docs in query_docs.values()]
    percent_positives = [
        p / len(docs) * 100 for p, docs in zip(positives, query_docs.values())
    ]

    def describe(values):
        return min(values), max(values), sum(values) / len(values)

    min_pos, max_pos, avg_pos = describe(positives)
    min_neg, max_neg, avg_neg = describe(negatives)
    min_pct, max_pct, avg_pct = describe(percent_positives)

    # Print results
    print("=== Qrels File Statistics ===")
    print(f"Number of unique query IDs: {num_queries}")
    print(f"Total number of query–document pairs: {total_pairs}\n")
    print(f"Positive docs per query: min={min_pos}, max={max_pos}, avg={avg_pos:.2f}")
    print(f"Negative docs per query: min={min_neg}, max={max_neg}, avg={avg_neg:.2f}")
    print(
        f"Percent positive per query: min={min_pct:.2f}%, max={max_pct:.2f}%, avg={avg_pct:.2f}%"
    )


if __name__ == "__main__":
    analyze_qrels(
        "boolean-query-generation/data/seed_collection/candidate_documents.res"
    )
