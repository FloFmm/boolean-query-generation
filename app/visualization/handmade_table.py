import datetime
import os
from pathlib import Path
import re
import time
import random
from Bio import Entrez
from app.dataset.utils import (
    get_dataset_details,
    review_id_to_dataset,
)
from app.pubmed.retrieval import evaluate_query, search_pubmed_date_range
from app.visualization.helper import escape_typst

Entrez.email = "florian_maurus.mueller@mailbox.tu-dresden.de"

def get_term_counts(query: str, count_cache: dict, end_year) -> dict:
    """
    Extract terms from the query, search each on PubMed, return a dict of term -> count.
    Uses count_cache to avoid searching the same term multiple times.
    """
    # Split query into tokens to find terms
    # Delimiters: AND, OR, NOT, (, )
    tokens = re.split(r'(\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\))', query)
    
    term_counts = {}
    
    for token in tokens:
        stripped = token.strip()
        # Skip delimiters and empty strings
        if not stripped or stripped in ['AND', 'OR', 'NOT', '(', ')']:
            continue
            
        # It is a term
        term = stripped
        
        # Check cache first
        if term in count_cache:
            term_counts[term] = count_cache[term]
            continue
            
        # Search PubMed
        print(f"    Searching PubMed for term: '{term}'...")
        try:
            # We use a simple search for the count
            record = search_pubmed_date_range(term, maxdate=datetime.date(end_year, 12, 31).strftime("%Y/%m/%d"), mindate="1800/01/01")
            count = int(record["Count"])
            
            time.sleep(0.5) # Be nice to PubMed
        except Exception as e:
            print(f"    Error searching for '{term}': {e}")
            count = -1
            
        term_counts[term] = count
        count_cache[term] = count
        
    return term_counts

def format_count(count: int) -> str:
    """Format count using k/m notation."""
    if count < 1000:
        return str(count)
    elif count < 10000:
        # e.g. 1923 -> 1.9k
        return f"{count / 1000:.1f}k"
    elif count < 1000000:
        # e.g. 12345 -> 12k, 512345 -> 512k
        return f"{int(round(count / 1000))}k"
    else:
        val_m = count / 1000000
        if val_m < 10:
             return f"{val_m:.2f}m"
        elif val_m < 100:
             return f"{val_m:.1f}m"
        else:
             return f"{int(round(val_m))}m"

def annotate_query_with_counts(query: str, term_counts: dict) -> str:
    """
    Reconstruct the query with [count] appended to each term.
    """
    tokens = re.split(r'(\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\))', query)
    annotated_parts = []
    
    for token in tokens:
        stripped = token.strip()
        if not stripped or stripped in ['AND', 'OR', 'NOT', '(', ')']:
            annotated_parts.append(token)
        else:
            term = stripped
            count = term_counts.get(term, "?")
            
            if isinstance(count, int) and count >= 0:
                count_str = format_count(count)
            else:
                count_str = str(count)
                
            annotated_parts.append(f"{term}[{count_str}]")
            
    return "".join(annotated_parts)

def steps_to_typst_table(steps_data: list, output_path: str) -> None:
    """
    Convert a list of query steps to a Typst table.

    Expected steps_data list of dicts with keys:
    - query: the query string
    - precision: precision value
    - recall: recall value
    - description: description of the step

    Output format:
    - Columns: Query Step, Precision, Recall, Description
    """

    typst_lines = []
    typst_lines.append('#import "../../thesis/assets/assets.typ": *')
    typst_lines.append("#let handmade_steps() = [")
    typst_lines.append("#table(")
    typst_lines.append("  columns: (3fr, auto, auto, auto, 1fr),")
    typst_lines.append("  table.header([Query], [Retrieved], [Precision], [Recall], [Step]),")

    for step in steps_data:
        # Use annotated query
        query = escape_typst(step["annotated_query"])
        precision = step["precision"]
        recall = step["recall"]
        retrieved = step["retrieved_count"]
        description = escape_typst(step["description"])

        typst_lines.append(f"  [{query}], [{retrieved}], [{precision:.4f}], [{recall:.4f}], [{description}],")

    typst_lines.append(")")
    typst_lines.append("]")

    with open(output_path, "w") as f:
        f.write("\n".join(typst_lines))


if __name__ == "__main__":
    dataset_details = get_dataset_details()
    # Review: Galactomannan detection for invasive aspergillosis in immunocompromised patients
    review_id = "CD007394"
        
    details = dataset_details[review_id]
    
    # Extract positives and end_year
    positives = set(dataset_details[review_id]["positives"])
    dataset, _, end_year = review_id_to_dataset(review_id)

    handmade_steps = [
        {
            "query": "Galactomannan detection for invasive aspergillosis in immunocompromised patients",
            "description": "Original review title."
        },
        {
            "query": "(galactomannan detection) AND (invasive aspergillosis) AND (immunocompromised patients)",
            "description": "Extracted key concepts combined with AND."
        },
        {
            "query": "(galactomannan OR antigen detection OR biomarker detection OR fungal biomarker OR GM test OR GM assay) AND (invasive aspergillosis) AND (immunocompromised patients)",
            "description": "Added synonyms for Galactomannan."
        },
        {
            "query": "(galactomannan OR antigen detection OR biomarker detection OR fungal biomarker OR GM test OR GM assay) AND (invasive aspergillosis OR aspergillus OR fungal infection OR invasive fungal) AND (immunocompromised patients)",
            "description": "Added synonyms for Invasive Aspergillosis."
        },
        {
            "query": "(galactomannan OR antigen detection OR biomarker detection OR fungal biomarker OR GM test OR GM assay) AND (invasive aspergillosis OR aspergillus OR fungal infection OR invasive fungal) AND (immunocompromised patients OR immunodeficiency OR neutropenia OR neutropenic OR immunosuppressed OR immunosuppressive)",
            "description": "Added synonyms for immunocompromised patients."
        },
        {
            "query": "(galactomannan OR antigen detection OR biomarker detection OR fungal biomarker OR GM test OR GM assay) AND (invasive aspergillosis OR aspergillus OR fungal infection OR invasive fungal) AND (immunocompromised patients OR immunodeficiency OR neutropenia OR neutropenic OR immunosuppressed OR immunosuppressive OR hematologic malignancy OR transplant OR HIV OR AIDS OR leukemia OR myeloma OR lymphoma OR myelodysplastic syndrome OR autoimmune disease)",
            "description": "Added illnesses related to immunocompromised patients."
        }
    ]

    
    # Store counts to avoid re-searching
    count_cache = {}

    results = []
    print(f"Evaluating queries for review {review_id} (End Year: {end_year})...")
    
    for i, step in enumerate(handmade_steps):
        query_str = step["query"]
        print(f"Evaluating Step {i+1}: {query_str}")

        # Fetches term counts and updates cache only if not the first step
        if i > 0:
            _ = get_term_counts(query_str, count_cache, end_year)
            annotated_query = annotate_query_with_counts(query_str, count_cache)
        else:
            annotated_query = query_str
        
        print(f"  Annotated Query: {annotated_query}")

        try:
            precision, recall, retrieved_count, TP = evaluate_query(
                query_str,
                positives,
                end_year=end_year,
            )
            print(f"  P: {precision:.4f}, R: {recall:.4f}, Retrieved: {retrieved_count}, TP: {TP}")
            
            results.append({
                "annotated_query": annotated_query,
                "precision": precision,
                "recall": recall,
                "retrieved_count": retrieved_count,
                "description": step["description"]
            })
        except Exception as e:
            print(f"  Failed to evaluate query: {query_str}")
            print(f"  Error: {e}")

    output_file = "../master-thesis-writing/writing/tables/handmade_steps/handmade_steps.typ"
    os.makedirs(Path(output_file).parent, exist_ok=True)
    steps_to_typst_table(results, output_file)
    print(f"Table generated at {output_file}")
