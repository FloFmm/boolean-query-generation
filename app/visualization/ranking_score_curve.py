import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from app.dataset.utils import ranking_file_path, get_dataset_details, get_positives
from app.parameter_tuning.compute_top_k import BUCKETS

def find_bucket(n_pos: int, buckets: list[tuple[int, int]]):
    for start, end in buckets:
        if start <= n_pos <= end:
            return f"{start}-{end}"
    return None

def plot_metric_score_curve_by_bucket(
    bucket_scores: dict,
    out_folder: str,
    title: str | None = None,
):
    bucket_mean_scores = {}

    for bucket, score_lists in bucket_scores.items():
        stacked = np.vstack(score_lists)  # shape: (num_queries, K)
        bucket_mean_scores[bucket] = np.nanmean(stacked, axis=0)

    ks = np.arange(1, MAX_K + 1)

    plt.figure(figsize=(8, 6))

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=len(bucket_mean_scores) - 1)

    def bucket_key(bucket_str: str):
        start, end = bucket_str.split("-")
        return int(start), int(end)

    for idx, (bucket, mean_scores) in enumerate(
        sorted(bucket_mean_scores.items(), key=lambda x: bucket_key(x[0]))
    ):
        plt.plot(
            ks,
            mean_scores,
            label=bucket,
            color=cmap(norm(idx)),
        )

    # DEBUG: print k where each bucket reaches certain score thresholds
        # DEBUG: for each score threshold, print list of ks ordered by bucket
    debug_scores = [0.98, 0.97, 0.96, 0.955, 0.95]

    sorted_buckets = sorted(bucket_mean_scores.keys(), key=bucket_key)

    print("\n=== DEBUG: ks per score threshold (bucket-ordered) ===")
    print(f"Buckets: {sorted_buckets}\n")

    for s in debug_scores:
        ks_for_score = []
        for bucket in sorted_buckets:
            mean_scores = bucket_mean_scores[bucket]
            idx = np.where(mean_scores <= s)[0]
            if len(idx) > 0:
                ks_for_score.append(idx[0] + 1)  # k is 1-based
            else:
                ks_for_score.append(None)
        print(f"score <= {s}: {ks_for_score}")


    plt.xscale("log")
    plt.xlabel("Rank position (k)")
    plt.ylabel("Average retrieval score")
    plt.title("Average ranking score vs rank (grouped by positives)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(title="Number of positives")
    plt.tight_layout()
    plt.show()

def plot_positive_score_stats_by_bucket(bucket_scores: dict):
    """
    Plots min / max / mean / median positive score per bucket.
    """

    def bucket_key(bucket_str: str):
        start, end = bucket_str.split("-")
        return int(start), int(end)

    buckets_sorted = sorted(bucket_scores.keys(), key=bucket_key)

    min_scores = []
    max_scores = []
    mean_scores = []
    median_scores = []

    for bucket in buckets_sorted:
        # stack: (num_queries, K)
        stacked = np.vstack(bucket_scores[bucket])

        # consider only positive scores (> 0)
        positives = stacked[stacked > 0]

        min_scores.append(np.min(positives))
        max_scores.append(np.max(positives))
        mean_scores.append(np.mean(positives))
        median_scores.append(np.median(positives))

    x = np.arange(len(buckets_sorted))

    plt.figure(figsize=(9, 6))

    plt.plot(x, min_scores, marker="o", label="Min")
    plt.plot(x, max_scores, marker="o", label="Max")
    plt.plot(x, mean_scores, marker="o", label="Mean")
    plt.plot(x, median_scores, marker="o", label="Median")

    plt.xticks(x, buckets_sorted, rotation=45)
    plt.xlabel("Bucket (number of positives)")
    plt.ylabel("Positive retrieval score")
    plt.title("Positive score statistics by bucket")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


MAX_K = 10000  # how far you want the curve
if __name__ == "__main__":
    bucket_scores = defaultdict(list)
    bucket_positives_scores = defaultdict(list)
    npz_files = ranking_file_path(
        retriever_name="pubmedbert",
        query_type="title_abstract",
        total_docs=503679,
    )
    dataset_details = get_dataset_details()

    for npz_path in sorted(npz_files):
        review_id = npz_path.stem  # removes .npz
        # positives = get_positives(review_id=review_id)
        n_pos = dataset_details[review_id]["real_num_positives"]

        bucket = find_bucket(n_pos, BUCKETS)
        if bucket is None:
            continue

        with np.load(npz_path) as data:
            scores = data["scores"][:MAX_K]
            pmids = data["ids"][:MAX_K]

            # pad if ranking shorter than MAX_K
            if len(scores) < MAX_K:
                scores = np.pad(scores, (0, MAX_K - len(scores)), constant_values=np.nan)

            bucket_scores[bucket].append(scores)
            
            # extract scores corresponding to positives
            # mask = np.isin(pmids, positives)
            # bucket_positives_scores[bucket].append(scores[mask])
                
    plot_metric_score_curve_by_bucket(bucket_scores=bucket_scores, out_folder=None)
    # plot_positive_score_stats_by_bucket(bucket_positives_scores)
    
    
