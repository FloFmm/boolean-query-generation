import pandas as pd
import glob
import ast
from collections import Counter
from Bio import Entrez
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def expand_mesh_terms(mesh_list):
    """
    Expands compact MeSH-style notations like
    'Down Syndrome/blood/*prevention & control'
    into full term combinations.
    """
    expanded = []
    for mesh_str in mesh_list:
        mesh_str = mesh_str.replace('&', 'and')
        mesh_str = mesh_str.replace('*', '')
        parts = mesh_str.split('/')
        expanded.append(parts[0])
        for p in parts[1:]:
            expanded.append(f"{parts[0]}/{p}")

    return expanded

# ==========================
# CONFIG
# ==========================
folder_path = '/home/florian/.cache/huggingface/datasets/tar2018//*.csv'  # CSV folder
Entrez.email = 'florian.maurus.mueller@gmail.com'    # Required by NCBI API
pause_seconds = 0.35                        # Avoid hitting API rate limit

# ==========================
# STEP 1: Load all CSVs
# ==========================
files = glob.glob(folder_path)
dfs = []

for f in files:
    df = pd.read_csv(f, usecols=['PMID', 'Label', 'MH'])
    
    # Replace NaN with empty list string
    df['MH'] = df['MH'].fillna('[]')
    
    # Convert string representation of list to actual list
    df['MH'] = df['MH'].apply(ast.literal_eval)
    df['expanded_MH'] = df['MH'].apply(expand_mesh_terms)
    dfs.append(df)


df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset='PMID')
df_pos = df_all[df_all['Label']==1]
df_neg = df_all[df_all['Label']!=1]
print("Docs:", len(df_all))
print("Positive Docs:", len(df_pos))
print("Negative Docs:", len(df_neg))
# ==========================
# STEP 2: Count MeSH terms
# ==========================
# Positive and negative label counters
mh_counter_pos = Counter()
mh_counter_neg = Counter()
pos_empty_mesh_count = 0
neg_empty_mesh_count = 0
for pmid, mesh_list, label in zip(df_all['PMID'],df_all['expanded_MH'], df_all['Label']):
    
    if label == 1:
        if not mesh_list:
            pos_empty_mesh_count +=1
            continue
        mh_counter_pos.update(set(mesh_list))
    else:
        if not mesh_list:
            neg_empty_mesh_count +=1
            continue
        mh_counter_neg.update(set(mesh_list))
print("Positive Docs without MESH: ", pos_empty_mesh_count)
print("Negative Docs without MESH: ", neg_empty_mesh_count)
print()
# Get all unique MeSH terms
all_mesh_terms = set(mh_counter_pos.keys())
print("all MESH terms:", len(all_mesh_terms))
min_occ = 10
all_mesh_terms = {mh for mh in all_mesh_terms if mh_counter_pos[mh] >= min_occ}
print(f">={min_occ} occurence MESH terms:", len(all_mesh_terms), "e.g.", list(all_mesh_terms)[:3])

# ==========================
# STEP 3: Fetch PubMed total counts for each MeSH term
# ==========================
pubmed_counts = {}
remove_set = set()
for mh in tqdm(all_mesh_terms, desc="Fetching PubMed counts"):
    # Search PubMed for this MeSH term
    handle = Entrez.esearch(db='pubmed', term=f'"{mh}"[MeSH Terms]', retmax=0)
    record = Entrez.read(handle)
    handle.close()
    
    # Total number of PubMed docs with this MeSH term
    pubmed_counts[mh] = int(record['Count'])
    if pubmed_counts[mh] == 0:
        print("pubmed_count == 0 for: ", mh)
        remove_set.add(mh)
    # Sleep to avoid hitting NCBI rate limits
    time.sleep(pause_seconds)
all_mesh_terms = all_mesh_terms - remove_set

# ==========================
# STEP 5: Greedy selection using dynamic ratios
# ==========================
covered_pos_pmids_acc, covered_neg_pmids_acc = set(), set()
greedy_mh_list = []
stats = []

i = 0

while True:
    best_mh = None
    best_ratio = 0
    
    # Iterate over all candidate MeSH terms
    for mh in (all_mesh_terms - set(greedy_mh_list)):
        # Find uncovered pmids this MH covers
        covered_pos_pmids = {pmid for pmid, mesh_list in zip(df_pos['PMID'], df_pos['expanded_MH'])
                           if mh in mesh_list and pmid not in covered_pos_pmids_acc}
        num_covered_pos_pmids = len(covered_pos_pmids)
        if num_covered_pos_pmids == 0:
            continue  # this term covers nothing new
        
        covered_neg_pmids = {pmid for pmid, mesh_list in zip(df_neg['PMID'], df_neg['expanded_MH'])
                           if mh in mesh_list and pmid not in covered_neg_pmids_acc}
        num_covered_neg_pmids = len(covered_neg_pmids)
        

        
        # Dynamic ratio: # uncovered positive docs / total PubMed docs for this MH
        total_pubmed = pubmed_counts.get(mh) 
        ratio = num_covered_pos_pmids / total_pubmed
        
        # Keep the term with the best ratio
        if ratio > best_ratio:
            best_covered_pos_pmids = covered_pos_pmids
            best_covered_neg_pmids = covered_neg_pmids
            best_total_pubmed = total_pubmed
            best_num_covered_pos_pmids = num_covered_pos_pmids
            best_num_covered_neg_pmids = num_covered_neg_pmids
            best_ratio = ratio
            best_mh = mh

    if best_mh is None:
        break  # no term improves coverage
    
    # Add the best MeSH term to greedy list
    greedy_mh_list.append(best_mh)

    # ==========================
    # Cumulative PubMed query as OR of all selected MeSH terms
    # ==========================
    query = " OR ".join([f'"{mh}"[MeSH Terms]' for mh in greedy_mh_list])
    handle = Entrez.esearch(db='pubmed', term=query, retmax=0)
    record = Entrez.read(handle)
    handle.close()
    covered_pos_pmids_acc = covered_pos_pmids_acc | best_covered_pos_pmids
    covered_neg_pmids_acc = covered_neg_pmids_acc | best_covered_neg_pmids


    # Accumulate stats for this iteration
    stats_dict = {
        "step": i + 1,
        "mesh": best_mh,
        "num_covered_pos_pmids": best_num_covered_pos_pmids,
        "num_covered_neg_pmids": best_num_covered_neg_pmids,
        "num_covered_pos_pmids_acc": len(covered_pos_pmids_acc),
        "num_covered_neg_pmids_acc": len(covered_neg_pmids_acc),
        "num_covered_pubmed_acc": int(record['Count']),
        "ratio": best_ratio,
        "query": query
    }

    stats.append(stats_dict)
    print(f"Step {stats_dict['step']}: {stats_dict['mesh']}, "
      f"pos_acc={stats_dict['num_covered_pos_pmids_acc']}, "
      f"neg_acc={stats_dict['num_covered_neg_pmids_acc']}, "
      f"pubmed_acc={stats_dict['num_covered_pubmed_acc']}, "
      f"ratio={stats_dict['ratio']:.4f}")

    # Save this row immediately to CSV
    pd.DataFrame([stats_dict]).to_csv("data/pubmed/statistics/mesh.csv", mode='a', index=False, header=(i==0))

    i += 1