from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import json
import time

from app.dataset.utils import get_dataset_details, review_id_to_dataset
from app.pubmed.retrieval import evaluate_query

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here
model_name = "ielabgroup/Autobool-Qwen4b-No-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH)

def get_autobool_query(topic):
    # Define your systematic review topic
    # topic = "Thromboelastography (TEG) and rotational thromboelastometry (ROTEM) for trauma-induced coagulopathy"
    # Construct the prompt with system and user messages
    messages = [
        {"role": "system", "content": "You are an expert systematic review information specialist. You are tasked to formulate a systematic review Boolean query in response to a research topic. The final Boolean query must be enclosed within <answer> </answer> tags. Do not include any explanation or reasoning."},
        {"role": "user", "content": f'You are given a systematic review research topic, with the topic title "{topic}". \n     Your task is to formulate a highly effective Boolean query in MEDLINE format for PubMed.\nThe query should balance **high recall** (capturing all relevant studies) with **reasonable precision** (avoiding irrelevant results):\n- Use both free-text terms and MeSH terms (e.g., chronic pain[tiab], Pain[mh]).\n- **Do not wrap terms or phrases in double quotes**, as this disables automatic term mapping (ATM).\n- Combine synonyms or related terms within a concept using OR.\n- Combine different concepts using AND.\n- Use wildcards (*) to capture word variants (e.g., vaccin* → vaccine, vaccination):\n  - Terms must have ≥4 characters before the * (e.g., colo*)\n  - Wildcards work with field tags (e.g., breastfeed*[tiab]).\n- Field tags limit the search to specific fields and disable ATM.\n- Do not include date limits.\n- Tag term using term field (e.g., covid-19[ti] vaccine[ti] children[ti]) when needed.\n**Only use the following allowed field tags:**\nTitle: [ti], Abstract: [ab], Title/Abstract: [tiab]\nMeSH: [mh], Major MeSH: [majr], Supplementary Concept: [nm]\nText Words: [tw], All Fields: [all]\nPublication Type: [pt], Language: [la]\n\nOutput and only output the formulated Boolean query inside <answer></answer> tags. Do not include any explanation or content outside or inside the <answer> tags.'}
    ]
    # Generate the query
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=2048)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the query from <answer> tags
    import re
    matches = re.findall(r'<answer>([^<]*)</answer>', response, re.DOTALL)
    query = None
    if matches:
        # Filter matches that contain " or " or " and " (case insensitive)
        valid_matches = [m for m in matches if re.search(r'\s(or|and)\s', m, re.IGNORECASE)]
        if valid_matches:
            # Take the shortest match
            query = min(valid_matches, key=len).strip()
        else:
            # Fallback to last match if none contain " or " or " and "
            query = matches[-1].strip()
        print(query)
    return query

if __name__ == "__main__":
    max_trials = 5
    output_file = "data/examples/autobool_results.jsonl"
    
    dataset_details = get_dataset_details()
    
    with open(output_file, "w") as f:
        for review_id, data in dataset_details.items():
            positives = set(data["positives"])
            _, _, end_year = review_id_to_dataset(review_id)
            
            # Retry logic: try up to max_trials times to get a non-None query
            query = None
            start_time = time.time()
            for trial in range(max_trials):
                query = get_autobool_query(data["title"])
                if query is not None:
                    break
                print(f"Trial {trial + 1}/{max_trials} failed for {review_id}, query is None. Retrying...")
            
            qg_time_seconds = time.time() - start_time
            
            # Raise error if query is still None after all trials
            if query is None:
                raise ValueError(f"Failed to generate query for {review_id} after {max_trials} attempts")
            
            # Evaluate the query
            precision, recall, retrieved_count, TP = evaluate_query(
                query,
                positives,
                end_year=end_year,
                max_retrieved=1000_000
            )
            
            # Write result to JSONL file
            result = {
                "query_id": review_id,
                "num_positive": len(positives),
                "pubmed_retrieved": retrieved_count,
                "pubmed_precision": precision,
                "pubmed_recall": recall,
                "qg_time_seconds": qg_time_seconds,
                "pubmed_query": query
            }
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure the line is written immediately
            print(f"Completed {review_id}: Precision={precision:.4f}, Recall={recall:.4f}") 