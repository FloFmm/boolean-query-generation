import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here

import sys
import os
import json
from app.tree_learning.text_preprocessing import bag_of_words

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, build_global_corpus


def create_bow_file(output_dir = "../systematic-review-datasets/data/bag_of_words/"):
    """Loads dataset, computes bag-of-words for each unique document, and writes them to a JSONL file."""
    
    dataset = load_dataset()
    global_corpus = build_global_corpus(dataset)
   
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"bag_of_words_docs={len(global_corpus)}.jsonl"
    )

    doc_ids = set()

    with open(output_path, "w", encoding="utf-8") as f:
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

                        bow = bag_of_words(text, mesh_terms)

                        record = {
                            "id": doc_id,
                            "title": doc["title"],
                            "abstract": doc["abstract"],
                            "bow": bow
                        }

                        f.write(json.dumps(record) + "\n")

    return output_path


def extract_unique_words(bow_file_path):
    """Reads the bag-of-words file and extracts all unique words into a separate file."""
    
    unique_words = set()

    with open(bow_file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            unique_words |= set(entry["bow"])

    # Output file
    output_dir = os.path.dirname(bow_file_path)

    base = os.path.basename(bow_file_path)            # e.g. "bag_of_words_docs=12345.jsonl"
    new_base = base.replace("bag_of_words", "unique_words")

    output_path = os.path.join(output_dir, new_base)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted(unique_words), f)

    return output_path


if __name__ == "__main__":
    output_dir = "../systematic-review-datasets/data/bag_of_words/"
    # bow_file_path = create_bow_file(output_dir)
    extract_unique_words(bow_file_path="/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/bag_of_words/bag_of_words_docs=433660.jsonl")