import pandas as pd
from app.dataset.utils import get_qg_results
from app.config.config import CURRENT_BEST_RUN_FOLDER

# function which takes a dataframe removes all lines with less than min_positives (parameter) in the field num_positive and then calcualtes for n buckets of pubmed_recall the average subset_recall and pseudo recall 
def compute_pseudo_recall_by_bucket(df, num_buckets=5, metric="precision"):
    
    # Create buckets based on pubmed_recall
    df["recall_bucket"] = pd.qcut(df[f"pseudo_{metric}"], q=num_buckets, labels=False)
    
    # Calculate average subset_recall and pseudo recall for each bucket
    bucket_stats = df.groupby("recall_bucket").agg(
        avg_subset=(f"subset_{metric}", "mean"),
        avg_pubmed=(f"pubmed_{metric}", "mean"),
        count=(f"pseudo_{metric}", "size")
    ).reset_index()
    
    return bucket_stats

# function that calulcates for a given df with subset recall/recall, pubmed p/r the average mismatch between the two. hence. average absolute mismatch for p and for r, i do not want positive and rneagtive mismatches to cancel each other-> abs
def compute_average_mismatch(df, metric="precision"):
    df["abs_mismatch"] = abs(df[f"pubmed_{metric}"] - df[f"subset_{metric}"])
    avg_abs_mismatch = df["abs_mismatch"].mean()
    df["mismatch"] = df[f"pubmed_{metric}"] - df[f"subset_{metric}"]
    avg_mismatch = df["mismatch"].mean()
    # additionally give me only the positive mistmach and only the negative mismatch
    
    return avg_mismatch, avg_abs_mismatch

if __name__ == "__main__":
    # dataframe = get_qg_results(CURRENT_BEST_RUN_FOLDER, min_positive_threshold=50)
    # print(compute_pseudo_recall_by_bucket(dataframe, num_buckets=20, metric="recall"))
    # print(compute_pseudo_recall_by_bucket(dataframe, num_buckets=20, metric="precision"))
    
    dataframe = get_qg_results(CURRENT_BEST_RUN_FOLDER, min_positive_threshold=None, top_k_types=["cosine"], restrict_betas=["50"])
    r_results = compute_average_mismatch(dataframe, metric="recall")
    print("average subset recall", dataframe["subset_recall"].mean())
    print("average pubmed recall", dataframe["pubmed_recall"].mean())
    print("average mismatch", r_results[0])
    print("average absolute mismatch", r_results[1])
    print()
    p_results = compute_average_mismatch(dataframe, metric="precision")
    print("average subset precision", dataframe["subset_precision"].mean())
    print("average pubmed precision", dataframe["pubmed_precision"].mean())
    print("average mismatch", p_results[0])
    print("average absolute mismatch", p_results[1])