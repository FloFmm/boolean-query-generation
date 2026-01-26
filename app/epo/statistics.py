import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import json
from prettytable import PrettyTable

load_dotenv()
import math
import warnings

warnings.filterwarnings("ignore")


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
    index="patents", body={"query": {"exists": {"field": "abstracts.text"}}}
)["count"]
claim_count = es.count(
    index="patents", body={"query": {"exists": {"field": "claims.text"}}}
)["count"]
table1 = PrettyTable()
table1.field_names = ["Document Category", "Documents", "Percentage"]
table1.add_row(["Total Entries", total, f"100%"])
table1.add_row(
    ["Entries With Abstract", abstract_count, f"{abstract_count / total * 100:.2f}%"]
)
table1.add_row(
    ["Entries With Claims", claim_count, f"{claim_count / total * 100:.2f}%"]
)
print(table1)

abstract_langs = es.search(
    index="patents",
    body={
        "size": 0,
        "aggs": {"langs": {"terms": {"field": "abstracts.lang.keyword", "size": 20}}},
    },
)
lang_buckets = abstract_langs["aggregations"]["langs"]["buckets"]
table2 = PrettyTable()
table2.field_names = ["Abstract Language", "Documents", "Percentage"]
sum_count = 0
for bucket in lang_buckets:
    lang = bucket["key"]
    count = bucket["doc_count"]
    sum_count += count
    pct = (count / abstract_count) * 100
    table2.add_row([lang, count, f"{pct:.2f}%"])
table2.add_row(["sum", sum_count, f"{sum_count / abstract_count * 100:.2f}%"])
print(table2)

claim_langs = es.search(
    index="patents",
    body={
        "size": 0,
        "aggs": {"langs": {"terms": {"field": "claims.lang.keyword", "size": 20}}},
    },
)
lang_buckets = claim_langs["aggregations"]["langs"]["buckets"]
table3 = PrettyTable()
table3.field_names = ["Claims Language", "Documents", "Percentage"]
sum_count = 0
for bucket in lang_buckets:
    lang = bucket["key"]
    count = bucket["doc_count"]
    sum_count += count
    pct = (count / claim_count) * 100
    table3.add_row([lang, count, f"{pct:.2f}%"])
table3.add_row(["sum", sum_count, f"{sum_count / claim_count * 100:.2f}%"])
print(table3)


def get_field_size(
    es, index, field, total_count, sample_size=50000, batch_size=10000, num_buckets=20
):
    """Estimate disk size of a field by sampling documents with scrolling."""
    doc_sizes = []
    zero_count = 0
    fetched = 0

    # Initial search with scroll
    res = es.search(
        index=index, body={"_source": [field]}, size=batch_size, scroll="2m"
    )
    scroll_id = res["_scroll_id"]
    hits = res["hits"]["hits"]

    while hits and fetched < sample_size:
        for hit in hits:
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

            fetched += 1
            if fetched >= sample_size:
                break

        # Fetch next batch
        res = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = res["_scroll_id"]
        hits = res["hits"]["hits"]

    # Clear scroll
    es.clear_scroll(scroll_id=scroll_id)

    total_bytes = sum(doc_sizes)
    avg_size = total_bytes / fetched
    estimated_total = avg_size * total_count

    # Histogram buckets for non-zero sizes
    histogram_buckets = [0] * num_buckets
    if doc_sizes:
        min_size = min(doc_sizes)
        max_size = max(doc_sizes)
        bucket_size = (
            math.ceil((max_size - min_size) / num_buckets)
            if max_size != min_size
            else 1
        )

        for size in doc_sizes:
            idx = min((size - min_size) // bucket_size, num_buckets - 1)
            histogram_buckets[int(idx)] += 1
    else:
        min_size = max_size = bucket_size = 0

    # Print histogram
    print(f"\nHistogram of '{field}' sizes (bytes):")
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
table4.add_row(["Total Index", f"{total_index_size / 1_000_000:.2f}", "100%"])
table4.add_row(
    [
        "Abstracts",
        f"{abstract_size / 1_000_000:.2f}",
        f"{abstract_size / total_index_size * 100:.2f}%",
    ]
)
table4.add_row(
    [
        "Claims",
        f"{claims_size / 1_000_000:.2f}",
        f"{claims_size / total_index_size * 100:.2f}%",
    ]
)
print(table4)

table5 = PrettyTable()
total_number_of_patents = 100_000_000
table5.field_names = ["Part", "Size (TB)", "Percentage"]
table5.add_row(
    [
        f"Prediction Total Index ({total_number_of_patents / 1000_000}mio)",
        f"{total_index_size / 1_000_000_000_000 * total_number_of_patents / count:.2f}",
        "100%",
    ]
)
table5.add_row(
    [
        "Prediction Abstracts",
        f"{abstract_size / 1_000_000_000_000 * total_number_of_patents / count:.2f}",
        f"{abstract_size / total_index_size * 100:.2f}%",
    ]
)
table5.add_row(
    [
        "Prediction Claims",
        f"{claims_size / 1_000_000_000_000 * total_number_of_patents / count:.2f}",
        f"{claims_size / total_index_size * 100:.2f}%",
    ]
)
print(table5)


# Get language counts
counts = es.search(
    index="patents",
    body={
        "size": 0,
        "aggs": {"pub_date": {"terms": {"field": "pub_date.keyword", "size": 20}}},
    },
)
lang_buckets = counts["aggregations"]["pub_date"]["buckets"]
table6 = PrettyTable()
table6.field_names = ["Language", "Patent Count", "Percentage"]
sum = 0
for bucket in lang_buckets:
    lang = bucket["key"] or "(missing)"
    count = bucket["doc_count"]
    percentage = count / total * 100
    table6.add_row([lang, f"{count:,}", f"{percentage:.2f}%"])
    sum += count
# Add total row
table6.add_row(["Total", f"{sum:,}", f"{sum / total * 100:.2f}%"])

print(table6)
