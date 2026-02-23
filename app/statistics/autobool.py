from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import random

CUSTOM_HF_PATH = "../systematic-review-datasets/data/huggingface"
os.environ["HF_HOME"] = CUSTOM_HF_PATH  # has to be up here
model_name = "ielabgroup/Autobool-Qwen4b-No-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CUSTOM_HF_PATH)

# Define your systematic review topic
topic = "Thromboelastography (TEG) and rotational thromboelastometry (ROTEM) for trauma-induced coagulopathy"

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
match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
if match:
    query = match.group(1).strip()
    print(query)