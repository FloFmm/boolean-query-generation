from collections import defaultdict

def analyze_clef_ip_qrels(file_path):
    query_docs = defaultdict(set)  # use set to count unique doc IDs per query

    # Parse file lines
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # skip malformed lines
            query_id, _, doc_id, _ = parts
            query_docs[query_id].add(doc_id)

    # Stats
    print(query_docs)
    num_queries = len(query_docs)
    total_pairs = sum(len(docs) for docs in query_docs.values())
    min_docs = min(len(docs) for docs in query_docs.values())
    max_docs = max(len(docs) for docs in query_docs.values())
    avg_docs = total_pairs / num_queries

    print("=== CLEF-IP 2012 QREL Statistics ===")
    print(f"Number of unique query IDs: {num_queries}")
    print(f"Total number of query–document pairs: {total_pairs}")
    print(f"Docs per query: min={min_docs}, max={max_docs}, avg={avg_docs:.2f}")

if __name__ == "__main__":
    analyze_clef_ip_qrels("data/clef-ip/02_topics/train_clms_psg/training-corrected/qrels-corrected.txt")  # replace with your actual file path
