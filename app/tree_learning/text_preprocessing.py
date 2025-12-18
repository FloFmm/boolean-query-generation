import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import spacy
import re
import string
from app.pubmed.mesh_term import expand_mesh_terms
# Load spaCy model (English, small is usually enough)
nlp = spacy.load("../systematic-review-datasets/data/spacy/en_core_web_lg-3.7.1/en_core_web_lg/en_core_web_lg-3.7.1", disable=["ner", "parser"])

def lemmatize_unique(text: str):
    """Return a set of unique lemmatized words (lowercased, alphabetic only)."""
    doc = nlp(text)
    return list({
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    })

def lemmatize_with_synonyms(text: str, conf: dict):
    """
    Return:
      - unique lemmas
      - lemma -> set(original forms)
    """
    doc = nlp(text)

    lemma_to_synonyms = defaultdict(set)

    punct_to_remove = string.punctuation.replace('-', '') # minus may stay
    PUNCT_RE = re.compile(f"[{re.escape(punct_to_remove)}]")
    MULTIPLE_RE = re.compile(r"[-\s]{2,}")

    for token in doc:
        if token.is_stop:
            continue
        if token.lemma_ == "\n\n":
            continue
        if conf.get("rm_numbers", False) and token.like_num:
            continue
        if conf.get("rm_punct", False) and token.is_punct:
            continue
            
        lemma = token.lemma_.lower()
        synonym = token.text.lower()
        if conf.get("rm_punct", False):
            if conf.get("rm_numbers", False):
                if re.fullmatch(r"[\d\s\W]+", lemma):
                    continue
                elif re.fullmatch(r"[\s\W]+", lemma):
                    continue
            lemma = PUNCT_RE.sub(" ", lemma).strip(" -")
            synonym = PUNCT_RE.sub(" ", synonym).strip(" -")
            lemma = MULTIPLE_RE.sub(" ", lemma)
            synonym = MULTIPLE_RE.sub(" ", synonym)
        if len(lemma) <= 1:
            continue
        
        lemma_to_synonyms[lemma].add(synonym)

    return lemma_to_synonyms


def bag_of_words(text: str, mesh_terms: list[str], conf: dict, mesh_ancestor_data=None):
    """
    Create a bag-of-words containing:
    - Lemmatized words from text
    - Normalized MeSH terms (not lemmatized)
    """
    expanded_mesh = expand_mesh_terms(mesh_terms, mesh_ancestor_data)
    
    # BOW from text
    synonym_map = lemmatize_with_synonyms(text, conf)
    bow_words = list(synonym_map.keys())
    bow_words = [
        f'"{w}"[tiab]' if " " in w else w
        for w in bow_words
    ]
    # BOW from MeSH terms
    bow_mesh = [f'"{term}"[mh]' for term in expanded_mesh]

    # Combine both with uniqueness
    bag = sorted(bow_words) + sorted(bow_mesh)

    return bag, synonym_map


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
