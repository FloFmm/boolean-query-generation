from Bio import Entrez
from Bio import Medline
from pubmed_retrieval import search_pubmed,fetch_pubmed_records,get_random_pubmed_documents
from logical_query_generation import train_text_classifier
import time
import random
# Always provide your email
Entrez.email = "florian_maurus.mueller@mailbox.tu-dresden.de"

# Search PubMed
search_term = '"CRISPR-Cas Systems"[MeSH]'
relevant_ids = search_pubmed(search_term, retmax=100)
random_ids = random.sample(range(1, 40000001), 3000)
negative_ids = list(set(random_ids) - set(relevant_ids))
relevant_records = fetch_pubmed_records(relevant_ids)
negative_records = fetch_pubmed_records(negative_ids)
print(f"{len(negative_records)} negative records retrived")
result = train_text_classifier([entry["abstract"] for entry in relevant_records], [entry["abstract"] for entry in negative_records])
print(result["accuracy"])
print(result["decision_tree"])
print(result["boolean_function_set1"])
