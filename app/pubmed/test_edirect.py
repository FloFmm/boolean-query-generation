#!/usr/bin/env python3
"""
fetch_pubmed_pmids.py

Fetch all PMIDs for a given PubMed query using EDirect (esearch + efetch)
and save them to a text file. Works for very large queries (>10,000 PMIDs).
"""

import subprocess

# SUPER SLOW


def fetch_all_pmids(query, output_file="pmids.txt", batch_size=100000):
    """Fetch all PMIDs for a PubMed query using EDirect."""
    esearch_cmd = f'esearch -db pubmed -query "{query}" | efetch -format uilist'
    esearch_out = subprocess.run(
        esearch_cmd, shell=True, capture_output=True, text=True
    )
    print(len(esearch_out.stdout.split("\n")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch all PMIDs for a PubMed query using EDirect"
    )
    parser.add_argument("query", help="PubMed query string, e.g. 'cancer'")
    parser.add_argument(
        "-o", "--output", default="pmids.txt", help="Output file (default: pmids.txt)"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=10000,
        help="Batch size per efetch (default: 100000)",
    )
    args = parser.parse_args()

    fetch_all_pmids(args.query, args.output, args.batch)
