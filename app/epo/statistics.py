import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import json
from prettytable import PrettyTable
load_dotenv()
import math

def get_client() -> Elasticsearch:
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    api_key_id = os.getenv("ELASTICSEARCH_API_KEY_ID")

    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    return Elasticsearch(url)


es = get_client()

total = es.count(index="patents")["count"]
abstract_count = es.count(
    index="patents",
    body={"query": {"exists": {"field": "abstracts.text"}}}
)["count"]
claim_count = es.count(
    index="patents",
    body={"query": {"exists": {"field": "claims.text"}}}
)["count"]
table1 = PrettyTable()
table1.field_names = ["Document Category", "Documents", "Percentage"]
table1.add_row(["Total Entries", total, f"100%"])
table1.add_row(["Entries With Abstract", abstract_count, f"{abstract_count/total*100:.2f}%"])
table1.add_row(["Entries With Claims", claim_count, f"{claim_count/total*100:.2f}%"])
print(table1)

abstract_langs = es.search(
    index="patents",
    body={
        "size": 0,
        "aggs": {
            "langs": {
                "terms": {"field": "abstracts.lang.keyword", "size": 20}
            }
        }
    }
)
lang_buckets = abstract_langs["aggregations"]["langs"]["buckets"]
total_lang_docs = sum(bucket["doc_count"] for bucket in lang_buckets)
table2 = PrettyTable()
table2.field_names = ["Abstract Language", "Documents", "Percentage"]
sum_count = 0
for bucket in lang_buckets:
    lang = bucket["key"]
    count = bucket["doc_count"]
    sum_count += count
    pct = (count / total_lang_docs) * 100
    table2.add_row([lang, count, f"{pct:.2f}%"])
table2.add_row(["sum", sum_count, f"{sum_count/total_lang_docs*100:.2f}%"])
print(table2)

claim_langs = es.search(
    index="patents",
    body={
        "size": 0,
        "aggs": {
            "langs": {
                "terms": {"field": "claims.lang.keyword", "size": 20}
            }
        }
    }
)
lang_buckets = abstract_langs["aggregations"]["langs"]["buckets"]
total_lang_docs = sum(bucket["doc_count"] for bucket in lang_buckets)
table3 = PrettyTable()
table3.field_names = ["Claims Language", "Documents", "Percentage"]
sum_count = 0
for bucket in lang_buckets:
    lang = bucket["key"]
    count = bucket["doc_count"]
    sum_count += count
    pct = (count / total_lang_docs) * 100
    table3.add_row([lang, count, f"{pct:.2f}%"])
table3.add_row(["sum", sum_count, f"{sum_count/total_lang_docs*100:.2f}%"])
print(table3)

def get_field_size(es, index, field, total_count, sample_size=10000, num_buckets=10):
    """Estimate disk size of a field by sampling documents and print histogram, including zero-size bucket."""
    res = es.search(
        index=index,
        body={
            "_source": [field],
            "size": sample_size
        }
    )

    doc_sizes = []
    zero_count = 0
    for hit in res["hits"]["hits"]:
        value = hit["_source"].get(field)
        if isinstance(value, list):
            size = sum(len(str(v.get("text")).encode("utf-8")) for v in value)
        elif value is not None:
            size = len(str(value).encode("utf-8"))
        else:
            size = 0

        if size == 0:
            zero_count += 1
        else:
            doc_sizes.append(size)

    total_bytes = sum(doc_sizes)
    avg_size = total_bytes / sample_size
    estimated_total = avg_size * total_count

    # Histogram buckets for non-zero sizes
    histogram_buckets = [0] * num_buckets
    if doc_sizes:
        min_size = min(doc_sizes)
        max_size = max(doc_sizes)
        bucket_size = math.ceil((max_size - min_size) / num_buckets) if max_size != min_size else 1

        for size in doc_sizes:
            idx = min((size - min_size) // bucket_size, num_buckets - 1)
            histogram_buckets[int(idx)] += 1
    else:
        min_size = max_size = bucket_size = 0

    # Print histogram
    print(f"\nHistogram of '{field}' sizes (bytes, sample size: {sample_size}):")
    print(f"      0 (empty) | {'#' * (zero_count * 40 // sample_size)} ({zero_count})")
    for i, count in enumerate(histogram_buckets):
        lower = min_size + i * bucket_size
        upper = lower + bucket_size - 1
        bar = "#" * (count * 40 // sample_size)  # scale to max 40 chars
        print(f"{lower:6} - {upper:6} | {bar} ({count})")

    return estimated_total

abstract_size = get_field_size(es, "patents", "abstracts", total)
claims_size = get_field_size(es, "patents", "claims", total)

# Get total index storage size from ES stats
index_stats = es.indices.stats(index="patents")
total_index_size = index_stats["_all"]["primaries"]["store"]["size_in_bytes"]

table4 = PrettyTable()
table4.field_names = ["Part", "Size (MB)", "Percentage"]
table4.add_row(["Total Index", f"{total_index_size/1_000_000:.2f}", "100%"])
table4.add_row(["Abstracts", f"{abstract_size/1_000_000:.2f}", f"{abstract_size/total_index_size*100:.2f}%"])
table4.add_row(["Claims", f"{claims_size/1_000_000:.2f}", f"{claims_size/total_index_size*100:.2f}%"])
print(table4)

table5 = PrettyTable()
table5.field_names = ["Part", "Size (TB)", "Percentage"]
table5.add_row(["Prediction Total Index (20mio)", f"{total_index_size/1_000_000_000_000*20_000_000/count:.2f}", "100%"])
table5.add_row(["Prediction Abstracts", f"{abstract_size/1_000_000_000_000*20_000_000/count:.2f}", f"{abstract_size/total_index_size*100:.2f}%"])
table5.add_row(["Prediction Claims", f"{claims_size/1_000_000_000_000*20_000_000/count:.2f}", f"{claims_size/total_index_size*100:.2f}%"])
print(table5)



