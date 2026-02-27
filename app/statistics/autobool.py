import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
import json
import time

from app.dataset.utils import get_dataset_details, review_id_to_dataset
from app.pubmed.retrieval import evaluate_query

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here
model_name = "ielabgroup/Autobool-Qwen4b-No-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH).to(device)
print(f"Model loaded on {device}")

def check_logic(bool_query: str) -> bool:
    """
    Validate the logical structure of a Boolean query.

    Args:
        bool_query: The Boolean query string to validate

    Returns:
        True if the query has valid logical structure, False otherwise
    """
    if not bool_query or not bool_query.strip():
        return False

    # Normalize query
    query = bool_query.strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'\b(and|or|not)\b', lambda m: m.group(1).upper(), query, flags=re.IGNORECASE)

    # Check balanced parentheses and detect empty ()
    depth = 0
    for i, char in enumerate(query):
        if char == '(':
            depth += 1
            if i + 1 < len(query) and query[i + 1] == ')':
                return False  # Empty parentheses
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False  # Unbalanced

    if depth != 0:
        return False

    # Tokenize and validate sequence
    token_pattern = r'\".*?\"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
    tokens = [t.upper() if t.upper() in {'AND', 'OR', 'NOT'} else t for t in tokens]

    if not tokens:
        return False

    # Validate token sequence
    valid_ops = {'AND', 'OR', 'NOT'}
    prev = None

    for i, token in enumerate(tokens):
        if token in {'AND', 'OR'}:
            if prev is None or prev in valid_ops or prev == '(':
                return False
        elif token == 'NOT':
            if i == len(tokens) - 1:
                return False
            if tokens[i + 1] in valid_ops or tokens[i + 1] == ')':
                return False
        elif token == '(':
            if prev and prev not in valid_ops and prev != '(':
                return False
        elif token == ')':
            if prev in valid_ops or prev == '(':
                return False
        prev = token

    return tokens[-1] not in valid_ops

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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=2048, do_sample=True, temperature=0.6)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the query from <answer> tags
    import re
    matches = re.findall(r'<answer>([^<]*)</answer>', response, re.DOTALL)
    print("matches")
    for i, m in enumerate(matches):
        print(f"Match {i}: '{m[:15] + '...' + m[-15:] if len(m) > 30 else m}'", flush=True)
        print()
    query = None
    if matches:
        # Filter matches that contain " or " or " and " (case insensitive)
        valid_matches = [m for m in matches if re.search(r'\s(or|and)\s', m, re.IGNORECASE)]
        if valid_matches:
            # Take the shortest match
            query = min(valid_matches, key=len).strip()
        print("query", query)
    if query is None:
        print("No valid query found in the response. Response was:")
        print(response)
    return query

if __name__ == "__main__":
    max_trials = 10
    output_file = "data/examples/autobool_results.jsonl"
    priority_query_ids = ["CD007394", "CD009579", "CD010438", "CD008170"]
    dataset_details = get_dataset_details()
    
    # Read existing results to skip already processed queries
    existing_query_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    existing_query_ids.add(result["query_id"])
        print(f"Found {len(existing_query_ids)} already processed queries, skipping them...")
    
    # Sort review_ids so priority ones come first
    sorted_review_ids = sorted(
        dataset_details.keys(),
        key=lambda x: (x not in priority_query_ids, x)
    )
    
    with open(output_file, "a") as f:
        for review_id in sorted_review_ids:
            # Skip if already processed
            if review_id in existing_query_ids:
                print(f"Skipping {review_id} (already processed)")
                continue
            data = dataset_details[review_id]
            positives = set(data["positives"])
            _, _, end_year = review_id_to_dataset(review_id)
            
            # Retry logic: try up to max_trials times to get a non-None query and successful evaluation
            query = None
            precision = recall = retrieved_count = TP = None
            start_time = time.time()
            
            for trial in range(max_trials):
                query = get_autobool_query(data["title"])
                
                if query is None:
                    print(f"Trial {trial + 1}/{max_trials} failed for {review_id}, query is None. Retrying...")
                    continue
                
                # Validate the query logic
                if not check_logic(query):
                    print(f"Trial {trial + 1}/{max_trials} failed for {review_id}, query has invalid logic: {query}")
                    query = None
                    continue
                
                # Try to evaluate the query
                try:
                    precision, recall, retrieved_count, TP = evaluate_query(
                        query,
                        positives,
                        end_year=end_year,
                        min_retrieved=1,
                        max_retrieved=200_000
                    )
                    # If evaluation succeeds, break out of the retry loop
                    break
                except Exception as e:
                    print(f"Trial {trial + 1}/{max_trials} failed for {review_id}, evaluation error: {e}. Retrying...")
                    query = None  # Reset query to trigger retry
            
            qg_time_seconds = time.time() - start_time
            
            # Raise error if query is still None after all trials
            if query is None or precision is None:
                raise ValueError(f"Failed to generate and evaluate query for {review_id} after {max_trials} attempts")
            
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