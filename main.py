from Bio import Entrez
from app.pubmed.retrieval import search_pubmed,fetch_pubmed_records, sample_jsonl_files, search_pubmed_med_cpt
from logical_query_generation import train_text_classifier
import time

# Always provide your email
Entrez.email = "florian_maurus.mueller@mailbox.tu-dresden.de"

# Search PubMed
start_time = time.time()
# Using Keywords
# search_term = '"CRISPR-Cas Systems"[MeSH]'
# relevant_ids = search_pubmed(search_term, retmax=1000)
# relevant_records = [entry for entry in fetch_pubmed_records(relevant_ids) if entry["abstract"]]

# Using dense Retreival
query = "studies on ketoprofen"
result = search_pubmed_med_cpt(query, top_k=100)
query_node, relevant_records, edges = result["query_node"], result["nodes"], result["edges"]
relevant_ids = [record["pmid"] for record in relevant_records]
print("time to find relevant records:", time.time()-start_time)

# Find negative Records
start_time = time.time()
random_records= sample_jsonl_files("./data/pubmed/baseline", 60, 300)
negative_records = [entry for entry in random_records if entry["pmid"] not in set(relevant_ids)]
print("time to find negative records:", time.time()-start_time)

# Generate boolean Query
print(f"{len(relevant_records)} relevant records")
print(f"{len(negative_records)} negative records")
start_time = time.time()
result = train_text_classifier([entry["title"] + entry["abstract"] for entry in relevant_records], [entry["title"] + entry["abstract"] for entry in negative_records])
print("time to train classifier:", time.time()-start_time)
print("accuracy:", result["accuracy"])
print(result["decision_tree"])
print("query:", result["boolean_function_set1"])
