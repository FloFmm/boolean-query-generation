import os
CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH # has to be up here

import sys
import os
import json
from app.tree_learning.text_preprocessing import bag_of_words

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "../systematic-review-datasets")))
from csmed.experiments.csmed_cochrane_retrieval import load_dataset, build_global_corpus

if __name__ == "__main__":
    dataset = load_dataset()
    global_corpus = build_global_corpus(dataset)

    output_dir = "../systematic-review-datasets/data/bag_of_words/"
    os.makedirs(output_dir, exist_ok=True)   # Create folder if missing

    output_path = os.path.join(
        output_dir,
        f"bag_of_words_docs={len(global_corpus)}.jsonl"
    )

    doc_ids = set()
    with open(output_path, "w", encoding="utf-8") as f:
        for split, reviews in dataset.items():
            for review_name, review_data in reviews.items():
                for split_name in review_data["data"].keys():  # 'train', 'val', 'test' etc.
                    for doc in review_data["data"][split_name]:
                        doc_id = doc["pmid"]
                        if doc_id not in doc_ids:
                            doc_ids.add(doc_id)
                            text = doc["title"] + "\n\n" + doc["abstract"]
                            mesh_terms = doc["mesh_terms"]
                            bow = bag_of_words(text, mesh_terms)

                            record = {
                                "id": doc_id,
                                "text": text,
                                "bow": bow
                            }

                            f.write(json.dumps(record) + "\n")