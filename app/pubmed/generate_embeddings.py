import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
model.eval()

# Input/output directories
input_dir = "data/pubmed/baseline"
output_dir = "data/pubmed/embeddings"
os.makedirs(output_dir, exist_ok=True)


def encode_batch(texts, batch_size):
    """Return normalized embeddings for a list of texts in batches with progress."""
    all_embeddings = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding batches",
        unit="batch",
        leave=False,
    ):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded)
            emb = model_output.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


# Process files
files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

for fname in tqdm(files, desc="Files", unit="file"):
    out_path = os.path.join(output_dir, fname.replace(".jsonl", ".pt"))
    if os.path.exists(out_path):
        print(f"Skipping {fname} (embeddings already exist)")
        continue

    in_path = os.path.join(input_dir, fname)

    pmids, texts = [], []
    with open(in_path, "r") as f:
        for line in f:
            record = json.loads(line)
            pmid = record.get("pmid")
            abstract = record.get("abstract", "").strip()
            if not abstract:
                continue
            title = record.get("title", "").strip()
            text = (title + " " + abstract).strip()
            if text:
                pmids.append(pmid)
                texts.append(text)

    if not texts:
        print(f"Skipping {fname} (no valid abstracts)")
        continue

    embeddings = encode_batch(texts, batch_size=1)
    emb_dict = {pmid: emb for pmid, emb in zip(pmids, embeddings)}

    torch.save(emb_dict, out_path)
    print(f"Saved embeddings for {fname} → {out_path}")
