from Bio import Entrez
from Bio import Medline
import random

Entrez.email = "florian_maurus.mueller@mailbox.tu-dresden.de"

def search_pubmed(term, retmax=1000):
    """Search PubMed for a term and return a list of PubMed IDs."""
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_pubmed_records(id_list):
    """Fetch PubMed records (title, abstract, MeSH) given a list of IDs."""
    records_list = []
    BATCH_SIZE = 200  # fetch in batches

    for start in range(0, len(id_list), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(id_list))
        batch_ids = id_list[start:end]

        handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
        records = Medline.parse(handle)

        for record in records:
            records_list.append({
                "title": record.get("TI", "No title available"),
                "abstract": record.get("AB", "No abstract available"),
                "mesh_terms": record.get("MH", [])
            })
        handle.close()

    return records_list

def get_random_pubmed_documents(search_term, n=9999):
    """
    Fetch n random PubMed documents from the corpus.
    """
    id_list = search_pubmed(search_term, retmax=1_000_000) # return 9999 ids at most (capped by pubmed)
    random_ids = random.sample(id_list, n)

    # Fetch article details in MEDLINE format
    records = fetch_pubmed_records(random_ids)
    return records