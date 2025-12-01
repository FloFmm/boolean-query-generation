from Bio import Entrez, Medline
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from sklearn.manifold import TSNE  # or UMAP if you prefer
from tqdm import tqdm
from collections import defaultdict
import random
import json
import os
import torch
import subprocess
import json
import time
Entrez.email = "florian_maurus.mueller@mailbox.tu-dresden.de"

Entrez.tool = "YearMonthSplitter"

def fetch_pmids(query, mindate=None, maxdate=None):
    """Fetch PMIDs for a query with date range (max 9999)."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        mindate=mindate,
        maxdate=maxdate,
        datetype="pdat",
        retmax=1000000,
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def search_pubmed_year_month(query, start_year=1800, end_year=2025):
    """Retrieve all PMIDs by splitting per year and month if needed."""
    if not query or not str(query).strip():
        print("Empty query — nothing to search.")
        return set()
    
    all_pmids = set()
    handle = Entrez.esearch(db="pubmed", term=query)
    record = Entrez.read(handle)
    handle.close()
    expected_pmids = int(record["Count"])
    all_pmids = set(record["IdList"])
    print(f"expected PMIDs: {expected_pmids}")
    if expected_pmids == len(all_pmids):
        print(f"retrived PMIDs: {len(all_pmids)}")
        return all_pmids
    
    last = False
    for year in range(end_year, start_year - 1, -1):
        count_remaining = expected_pmids - len(all_pmids)
        print("reamining:", count_remaining)
        if count_remaining == 0:
            break
        if count_remaining < 9999:
            last = True
            mindate = f"{start_year}/01/01"
        else:
            mindate = f"{year}/01/01"
        maxdate = f"{year}/12/31"

        pmids = fetch_pmids(query, mindate, maxdate)
        count_year = len(pmids)
        print(f"{mindate}-{maxdate}: {count_year} results")

        # If year contains <10k, retrieve directly
        if count_year < 9999:
            all_pmids.update(pmids)
            if last:
                print(f"expected PMIDs: {expected_pmids} retrived PMIDs: {len(all_pmids)}")
                return all_pmids
            time.sleep(0.34)  # NCBI polite limit
            continue

        # Split into months if year has 10k results
        print(f" Splitting year {year} into months...")
        for month in range(1, 13):
            mindate = f"{year}/{month:02d}/01"
            if month == 12:
                maxdate = f"{year}/12/31"
            else:
                maxdate = f"{year}/{month+1:02d}/01"

            month_pmids = fetch_pmids(query, mindate, maxdate)
            print("month", len(month_pmids))
            if len(month_pmids) >= 9995:
                exit(0)
            all_pmids.update(month_pmids)
            time.sleep(0.34)

    # Deduplicate (some overlap may happen)
    all_pmids = list(dict.fromkeys(all_pmids))
    print(f"expected PMIDs: {expected_pmids} retrived PMIDs: {len(all_pmids)}")


    return all_pmids

def search_pubmed(term, retmax=1000):
    """Search PubMed for a term and return a list of PubMed IDs."""
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record



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
    id_list = search_pubmed(search_term, retmax=1_000_000)["IdList"] # return 9999 ids at most (capped by pubmed)
    random_ids = random.sample(id_list, n)

    # Fetch article details in MEDLINE format
    records = fetch_pubmed_records(random_ids)
    return records

def sample_jsonl_files(folder_path, n_files, n_lines):
    folder = Path(folder_path)
    all_files = list(folder.glob("*.jsonl"))

    if n_files < len(all_files):
        sampled_files = random.sample(all_files, n_files)
    else:
        sampled_files = all_files
    result = []

    for file_path in sampled_files:
        # First pass: count lines
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        # Randomly choose which lines to read
        chosen_indices = set(random.sample(range(total_lines), min(n_lines * 3, total_lines)))
        # Oversample a bit (3x) to account for invalid lines (missing fields)

        # Second pass: extract chosen lines
        collected = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(collected) >= n_lines:
                    break
                if i in chosen_indices:
                    try:
                        data = json.loads(line)
                        if all(k in data and data[k] for k in ("title", "abstract", "pmid")):
                            collected.append(data)
                    except json.JSONDecodeError:
                        continue

        result.extend(collected)

    return result

def load_pubmed_embeddings(
    embeddings_dir: str = "data/pubmed/embeddings"
) -> Tuple[List[str], torch.Tensor, Dict[str, str]]:
    """
    Load all PubMed embeddings from disk.

    Args:
        embeddings_dir (str): Directory containing saved .pt embedding files.

    Returns:
        Tuple:
            - all_pmids (List[str]): List of PubMed IDs.
            - all_embs (torch.Tensor): Tensor of embeddings (N x D).
            - file_map (Dict[str, str]): Maps PMID -> corresponding .jsonl file.
    """
    all_pmids, all_embs = [], []

    for fname in os.listdir(embeddings_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(embeddings_dir, fname)
        emb_dict = torch.load(path, map_location="cpu")  # dict: pmid -> tensor

        for pmid, emb in emb_dict.items():
            all_pmids.append(pmid)
            all_embs.append(emb)

    if not all_embs:
        raise ValueError(f"No embeddings found in directory: {embeddings_dir}")

    all_embs = torch.stack(all_embs)
    print(f"✅ Loaded {len(all_pmids):,} PubMed embeddings.")
    return all_pmids, all_embs


def load_valid_pmids(embeddings_dir: str):
    """Collect all PMIDs present in the stored embeddings."""
    valid_pmids = set()
    for fname in os.listdir(embeddings_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(embeddings_dir, fname)
        emb_dict = torch.load(path, map_location="cpu")
        valid_pmids.update(emb_dict.keys())
    return valid_pmids

def search_pubmed_med_cpt(
    query: str,
    all_pmids: List[str],
    all_embs: torch.Tensor,
    top_k: int = 10,
    device: str = None,
) -> List[str]:
    """
    Retrieve PubMed PMIDs most relevant to a query using preloaded embeddings.

    Args:
        query (str): The search query.
        all_pmids (List[str]): PMIDs corresponding to the embeddings.
        all_embs (torch.Tensor): Preloaded embedding tensor (N x D).
        top_k (int): Number of top results to return.
        device (str or torch.device): 'cuda' or 'cpu'. Defaults to auto-detect.

    Returns:
        List[str]: List of retrieved PMIDs.
    """
    # Device setup
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model & tokenizer once per process
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    model.eval()

    # Encode query
    encoded_query = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        query_emb = model(**encoded_query).last_hidden_state[:, 0, :]
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1).squeeze(0).cpu()

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), all_embs)
    values, indices = similarities.topk(top_k)

    # Collect PMIDs in ranked order
    selected_pmids = [all_pmids[idx] for idx in indices.tolist()]

    return selected_pmids

def search_pubmed_med_cpt_graph(
    query: str,
    embeddings_dir: str = "data/pubmed/embeddings",
    records_dir: str = "data/pubmed/baseline",
    top_k: int = 10,
    device: str = None,
    similarity_threshold: float = 0.3,  # only keep strong neighbors
    reduce_to_2d: bool = True
) -> Dict[str, List[Dict]]:
    """
    Retrieve PubMed records most relevant to a query and prepare them for visualization.

    Args:
        query (str): The query string to search.
        embeddings_dir (str): Directory containing saved .pt embeddings.
        records_dir (str): Directory containing original JSONL records.
        top_k (int): Number of top records to return.
        device (str or torch.device): 'cuda' or 'cpu'. Defaults to auto-detect.
        similarity_threshold (float): Drop neighbors below this similarity.
        reduce_to_2d (bool): Whether to compute 2D positions of embeddings.

    Returns:
        Dict: {
            "nodes": List of dicts {pmid, title, abstract, pos: [x,y]},
            "edges": List of dicts {source: pmid, target: pmid, weight: similarity}
        }
    """

    # Device setup
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
    model.eval()

    # Encode the query
    encoded_query = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        query_emb = model(**encoded_query).last_hidden_state[:, 0, :]
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1).squeeze(0).cpu()

    # Load all embeddings
    all_pmids, all_embs, file_map = [], [], {}
    for fname in os.listdir(embeddings_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(embeddings_dir, fname)
        emb_dict = torch.load(path)  # dict: pmid -> tensor
        for pmid, emb in emb_dict.items():
            all_pmids.append(pmid)
            all_embs.append(emb)
            file_map[pmid] = fname.replace(".pt", ".jsonl")

    if not all_embs:
        raise ValueError("No embeddings found in directory.")

    all_embs = torch.stack(all_embs)

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), all_embs)
    values, indices = similarities.topk(top_k)

    # Collect results (apply threshold)
    selected_pmids, selected_embs, selected_sims = [], [], []
    for sim, idx in zip(values.tolist(), indices.tolist()):
        if sim < similarity_threshold:
            continue
        pmid = all_pmids[idx]
        selected_pmids.append(pmid)
        selected_embs.append(all_embs[idx])
        selected_sims.append(sim)

    # Build node list (query + neighbors)
    nodes, pmid_to_pos = [], {}
    emb_matrix = torch.vstack([query_emb] + selected_embs).numpy()

    if reduce_to_2d:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_matrix)-1))
        pos_2d = reducer.fit_transform(emb_matrix)
    else:
        pos_2d = emb_matrix[:, :2]  # just take first 2 dims

    # Query node
    query_node = {
        "pmid": "QUERY",
        "title": query,
        "abstract": None,
        "pos": pos_2d[0].tolist()
    }

    # Neighbor nodes
    nodes = []
    for i, pmid in enumerate(selected_pmids, start=1):
        record = None
        record_path = os.path.join(records_dir, file_map[pmid])
        with open(record_path, "r") as f:
            for line in f:
                r = json.loads(line)
                if r.get("pmid") == pmid:
                    record = r
                    break
        if record is None:
            continue
        nodes.append({
            "pmid": pmid,
            "title": record.get("title"),
            "abstract": record.get("abstract"),
            "pos": pos_2d[i].tolist()
        })

    # Edges (query -> neighbor)
    edges = []
    for pmid, sim in zip(selected_pmids, selected_sims):
        edges.append({
            "source": "QUERY",
            "target": pmid,
            "weight": sim
        })

    return {"query_node": query_node, "nodes": nodes, "edges": edges}

def classify_by_mesh(folder_path, n_docs = 1_000_000_000_000):
    folder = Path(folder_path)
    all_files = list(folder.glob("*.jsonl"))
    docs_by_pmid = {}
    pmids_by_mesh = defaultdict(list)
    count = 0
    fail_count = 0
    # Outer progress bar for files
    for file_path in tqdm(all_files, desc="Processing files", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            # Inner progress bar for lines in each file
            for line in f:
                try:
                    data = json.loads(line)
                    pmid, abstract, title, mesh_terms, bag_of_words = data["pmid"], data["abstract"], data["title"], data["mesh_terms"], data["bag_of_words"]
                    docs_by_pmid[pmid] = {"title": title, "abstract": abstract, "mesh_terms": mesh_terms, "bag_of_words": bag_of_words}
                    for mesh in mesh_terms:
                        pmids_by_mesh[mesh].append(pmid)
                    count += 1

                    if count >= n_docs:
                        print(f"Failed to extract {fail_count} jsonl lines")
                        return docs_by_pmid, pmids_by_mesh
                except Exception:
                    fail_count += 1
                    continue
    print(f"Failed to extract {fail_count} jsonl lines")
    return docs_by_pmid, pmids_by_mesh

