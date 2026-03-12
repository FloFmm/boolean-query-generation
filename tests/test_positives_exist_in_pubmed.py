from app.pubmed.retrieval import search_pubmed_dynamic
from app.dataset.utils import get_dataset_details

dd = get_dataset_details()
print(len(dd))
ids = []
for k,v in dd.items():
    ids += set(v["positives"])
ids = list(set(ids))
print("len(ids)", len(ids))

batch_sze = 1000
for i in range(0, len(ids), batch_sze):
    batch_ids = ids[i:i+batch_sze]
    query = " OR ".join(batch_ids)
    # print(query)
    print(f"Batch {i//batch_sze + 1}: Querying {len(batch_ids)} IDs")
    result = search_pubmed_dynamic(query, end_year=2026, min_retrieved=0, max_retrieved=10000000)
    print(f"Batch {i//batch_sze + 1}: Retrieved {len(result)} results")
    # print(sorted(batch_ids))
    # print(sorted(result))
    assert sorted(batch_ids) == sorted(result), f"Mismatch in batch {i//batch_sze + 1}: expected {sorted(batch_ids)}, got {sorted(result)}"
    assert set(batch_ids) - set(result) == set(), f"Missing IDs in batch {i//batch_sze + 1}: {set(batch_ids) - set(result)}"
