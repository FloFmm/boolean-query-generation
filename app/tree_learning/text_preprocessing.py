import json
from pathlib import Path
from tqdm import tqdm
import spacy
from pubmed.mesh_term import expand_mesh_terms
# Load spaCy model (English, small is usually enough)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def lemmatize_unique(text: str):
    """Return a set of unique lemmatized words (lowercased, alphabetic only)."""
    doc = nlp(text)
    return list({
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.isalpha()
    })


def bag_of_words(text: str, mesh_terms: list[str]):
    """
    Create a bag-of-words containing:
    - Lemmatized words from text
    - Normalized MeSH terms (not lemmatized)
    """
    expanded_mesh = expand_mesh_terms(mesh_terms)
    
    # BOW from text
    bow_words = lemmatize_unique(text)

    # BOW from MeSH terms
    bow_mesh = [f'"{term}"[mh]' for term in expanded_mesh]

    # Combine both with uniqueness
    bag = sorted(bow_words) + sorted(bow_mesh)

    return bag


def process_jsonl_file(file_path: Path, skip_existing: bool):
    # Read all lines
    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in tqdm(lines, desc=f"Processing {file_path.name}"):
        data = json.loads(line)
        if skip_existing and data.get("bag_of_words"):
            continue
        # Combine title, abstract, and MeSH terms
        combined_text = " ".join([data.get("title", ""), data.get("abstract", "")])
        # Add bag_of_words field
        data["bag_of_words"] = " ".join(lemmatize_unique(combined_text))
        updated_lines.append(json.dumps(data, ensure_ascii=False))
    
    # Write back in place
    with file_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

def process_folder(folder_path: str, skip_existing: bool):
    folder = Path(folder_path)
    jsonl_files = list(folder.glob("*.jsonl"))
    for file_path in jsonl_files:
        process_jsonl_file(file_path, skip_existing)

# Usage
process_folder("data/pubmed/baseline", skip_existing = True)
