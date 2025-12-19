import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here
import re
import sys
import os
import json
from collections import defaultdict
from app.preprocessing.text_preprocessing import bag_of_words
from app.preprocessing.synonyms import build_dominating_map, transitive_closure
from app.dataset.utils import bag_of_words_path, synonym_map_path
from app.pubmed.mesh_term import download_mesh_xml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, build_global_corpus

def create_bow_file(output_dir = "../systematic-review-datasets/data/bag_of_words/", conf={}):
    """Loads dataset, computes bag-of-words for each unique document, and writes them to a JSONL file."""
    if conf["mesh_ancestors"]:
        mesh_ancestor_data = download_mesh_xml(2025)
    else:
        mesh_ancestor_data = None
    
    dataset = load_dataset()
    global_corpus = build_global_corpus(dataset)
   
    os.makedirs(output_dir, exist_ok=True)

    conf["total_docs"] = len(global_corpus)
    bow_output_path = bag_of_words_path(**conf)
    synonym_map_output_path = synonym_map_path(**conf)

    doc_ids = set()
    global_synonym_map = defaultdict(set)
    records = []
    with open(bow_output_path, "w", encoding="utf-8") as f:
        for split, reviews in dataset.items():
            for review_name, review_data in reviews.items():
                for split_name in review_data["data"].keys():  # train/val/test
                    for doc in review_data["data"][split_name]:
                        doc_id = doc["pmid"]

                        # Avoid duplicates
                        if doc_id in doc_ids:
                            continue
                        doc_ids.add(doc_id)

                        text = doc["title"] + "\n\n" + doc["abstract"]
                        mesh_terms = doc["mesh_terms"]
                        
                        bow, synonym_map = bag_of_words(text, mesh_terms, conf, mesh_ancestor_data)
                        for lemma, synonym in synonym_map.items():
                            global_synonym_map[lemma].update(synonym)

                        records.append({
                            "id": doc_id,
                            "title": doc["title"],
                            "abstract": doc["abstract"],
                            "bow": bow
                        })

    if conf["related_words"]:
        all_lemmas = global_synonym_map.keys()
        dom_map, reverse_map = build_dominating_map(all_lemmas, transitive_closure)
        for record in records:
            record["bow"] = sorted(set([dom_map[w] for w in record["bow"]]))
        
        result_map = defaultdict(set)
        for lemma, forms in global_synonym_map:
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