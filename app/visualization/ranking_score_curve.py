import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from app.dataset.utils import ranking_file_path, get_positives
from app.parameter_tuning.compute_top_k import BUCKETS

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../..", "../systematic-review-datasets"
        )
    )
)
from csmed.experiments.csmed_cochrane_retrieval import load_dataset

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

    for idx, (bucket, mean_scores) in enumerate(sorted(bucket_mean_scores.items())):
        plt.plot(
            ks,
            mean_scores,
            label=bucket,
            color=cmap(norm(idx)),
        )

    plt.xscale("log")
    plt.xlabel("Rank position (k)")
    plt.ylabel("Average retrieval score")
    plt.title("Average ranking score vs rank (grouped by positives)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(title="Number of positives")
    plt.tight_layout()
    plt.show()

MAX_K = 10000  # how far you want the curve
if __name__ == "__main__":
    dataset = load_dataset()
    
    bucket_scores = defaultdict(list)
    npz_files = ranking_file_path(
        retriever_name="pubmedbert",
        query_type="title_abstract",
        total_docs=503679,
    )

    for npz_path in sorted(npz_files):
        review_id = npz_path.stem  # removes .npz

        positives = get_positives(review_id, dataset)
        n_pos = len(positives)

        bucket = find_bucket(n_pos, BUCKETS)
        if bucket is None:
            continue

        with np.load(npz_path) as data:
            scores = data["scores"][:MAX_K]

            # pad if ranking shorter than MAX_K
            if len(scores) < MAX_K:
                scores = np.pad(scores, (0, MAX_K - len(scores)), constant_values=np.nan)

            bucket_scores[bucket].append(scores)
            
    plot_metric_score_curve_by_bucket(bucket_scores=bucket_scores, out_folder=None)