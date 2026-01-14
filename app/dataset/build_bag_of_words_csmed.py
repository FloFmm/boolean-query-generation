import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here
import re
import sys
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import json
from joblib import Parallel, delayed
from collections import defaultdict
from app.preprocessing.text_preprocessing import bag_of_words, nlp
from app.pubmed.mesh_term import strip_mesh_term
from app.preprocessing.synonyms import build_dominating_map, transitive_closure
from app.dataset.utils import bag_of_words_path, synonym_map_path
from app.pubmed.mesh_term import download_mesh_xml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, build_global_corpus

def process_doc(doc, conf, mesh_ancestor_data):
    mesh_terms = [strip_mesh_term(m) for m in doc["mesh_terms"]]
    bow, synonym_map = bag_of_words(doc["text"], mesh_terms, conf, mesh_ancestor_data)
    return {
        "id": doc["id"],
        "title": doc["title"],
        "abstract": doc["abstract"],
        "bow": bow,
        "synonym_map": synonym_map
    }
    
def process_doc_batch(doc_batch, conf, mesh_ancestor_data):
    batch_results = []
    for doc in doc_batch:
        batch_results.append(process_doc(doc, conf, mesh_ancestor_data))
    return batch_results

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
def create_bow_file(output_dir = "../systematic-review-datasets/data/bag_of_words/", conf=None):
    """Loads dataset, computes bag-of-words for each unique document, and writes them to a JSONL file."""
    if conf["mesh_ancestors"]:
        mesh_ancestor_data = download_mesh_xml(2025)
    else:
        mesh_ancestor_data = None
    
    dataset = load_dataset()
    global_corpus = build_global_corpus(dataset)[:1000]
    # texts = [d["text"] for d in global_corpus]
    # doc_meta = [((d["id"], d["title"], d["abstract"], d["mesh_terms"])) for d in global_corpus]
    print("Finished building global corpus", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    conf["total_docs"] = len(global_corpus)
    bow_output_path = bag_of_words_path(**conf)
    synonym_map_output_path = synonym_map_path(**conf)

    # doc_ids = set()
    global_synonym_map = defaultdict(set)
    records = []
    n_cpus = 32  # or you can use -1 to use all available CPUs
    bucket_size = 100
    batches = list(chunks(global_corpus, bucket_size))
    # Run in parallel
    results = Parallel(n_jobs=n_cpus)(
        delayed(process_doc_batch)(batch, conf, mesh_ancestor_data) 
        for batch in tqdm(batches, total=len(batches), desc="Processing docs in batches")
    )
    # Flatten results
    results = [doc for batch_result in results for doc in batch_result]

    records = []
    for r in results:
        for lemma, synonyms in r["synonym_map"].items():
            global_synonym_map[lemma].update(synonyms)
        records.append({
            "id": r["id"],
            "title": r["title"],
            "abstract": r["abstract"],
            "bow": r["bow"]
        })

    if conf["related_words"]:
        all_lemmas = global_synonym_map.keys()
        dom_map, reverse_map = build_dominating_map(all_lemmas, transitive_closure)
        for record in records:
            record["bow"] = sorted(set([w if w.endswith("[mh]") else dom_map[w] for w in record["bow"]]))
        
        result_map = defaultdict(set)
        for lemma, forms in global_synonym_map.items():
            if lemma in dom_map:
                result_map[dom_map[lemma]].update(forms)
            else:
                result_map[lemma].update(forms)
        global_synonym_map = result_map
     
    # Convert sets to sorted lists
    global_synonym_map = {
        lemma: sorted(list(forms))
        for lemma, forms in global_synonym_map.items()
    }
    
    # Write all records at once to JSONL
    with open(bow_output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    with open(synonym_map_output_path, "w", encoding="utf-8") as f:
        json.dump(global_synonym_map, f, indent=2)

def test():
    from app.preprocessing.text_preprocessing import expand_mesh_terms
    mesh_ancestor_data = download_mesh_xml(2025)
    print(expand_mesh_terms([strip_mesh_term("Classical swine Fever Virus")], mesh_ancestor_data))


if __name__ == "__main__":
    config = {
        "lower_case": True,
        "mesh_ancestors": True,
        "rm_numbers": True,
        "rm_punct": True,
        "related_words": True,
    }
    
    output_dir = "../systematic-review-datasets/data/bag_of_words/"
    create_bow_file(output_dir, config)