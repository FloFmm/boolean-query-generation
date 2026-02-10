import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from app.dataset.utils import ranking_file_path, get_dataset_details, select_k_positive_dependent, select_k_cosine_threshold
from app.parameter_tuning.compute_top_k import BUCKETS
from app.helper.helper import f_beta
from app.config.config import apply_matplotlib_style, COLORS, COLORMAPS

apply_matplotlib_style()


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

    plt.figure()

    def bucket_key(bucket_str: str):
        start, end = bucket_str.split("-")
        return int(start), int(end)

    sorted_buckets = sorted(bucket_mean_scores.items(), key=lambda x: bucket_key(x[0]))
    num_buckets = len(sorted_buckets)
    cmap = cm.get_cmap(COLORMAPS["spectrum"])
    norm = mcolors.Normalize(vmin=0, vmax=num_buckets - 1)

    for idx, (bucket, mean_scores) in enumerate(sorted_buckets):
        plt.plot(
            ks,
            mean_scores,
            label=bucket,
            color=cmap(norm(idx)),
        )

    # DEBUG: print k where each bucket reaches certain score thresholds
    # DEBUG: for each score threshold, print list of ks ordered by bucket
    # debug_scores = [0.98, 0.97, 0.96, 0.955, 0.95]

    # sorted_buckets = sorted(bucket_mean_scores.keys(), key=bucket_key)

    # print("\n=== DEBUG: ks per score threshold (bucket-ordered) ===")
    # print(f"Buckets: {sorted_buckets}\n")

    # for s in debug_scores:
    #     ks_for_score = []
    #     for bucket in sorted_buckets:
    #         mean_scores = bucket_mean_scores[bucket]
    #         idx = np.where(mean_scores <= s)[0]
    #         if len(idx) > 0:
    #             ks_for_score.append(idx[0] + 1)  # k is 1-based
    #         else:
    #             ks_for_score.append(None)
    #     print(f"score <= {s}: {ks_for_score}")

    plt.xscale("log")
    plt.xlabel("Rank position (k)")
    plt.ylabel("Average retrieval score")
    plt.title("Average ranking score vs rank (grouped by positives)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(title="Number of positives")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "metric_score_curve_by_bucket.png"), dpi=300)
    plt.close()

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
        arrays = [arr for arr in bucket_scores[bucket] if arr.size > 0]

        if len(arrays) == 0:
            min_scores.append(np.nan)
            max_scores.append(np.nan)
            mean_scores.append(np.nan)
            median_scores.append(np.nan)
            continue

        positives = np.concatenate(arrays)
        positives = positives[positives > 0]

        min_scores.append(np.min(positives))
        max_scores.append(np.max(positives))
        mean_scores.append(np.mean(positives))
        median_scores.append(np.median(positives))

    x = np.arange(len(buckets_sorted))

    plt.figure()

    plt.plot(x, min_scores, marker="o", label="Min", color=COLORS["category"][0])
    plt.plot(x, max_scores, marker="o", label="Max", color=COLORS["category"][1])
    plt.plot(x, mean_scores, marker="o", label="Mean", color=COLORS["category"][2])
    plt.plot(x, median_scores, marker="o", label="Median", color=COLORS["category"][3])

    plt.xticks(x, buckets_sorted, rotation=45)
    plt.xlabel("Bucket (number of positives)")
    plt.ylabel("Positive retrieval score")
    plt.title("Positive score statistics by bucket")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "positive_score_stats_by_bucket.png"), dpi=300)
    plt.close()

def compute_precision_recall_at_k(
    pmids: np.ndarray,
    positives: set,
    k: int,
) -> tuple[float, float]:
    """
    Compute precision and recall at cutoff k.
    """
    topk_ids = pmids[:k]

    tp = sum(pid in positives for pid in topk_ids)
    fp = k - tp
    fn = len(positives) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall

def plot_precision_recall_by_bucket(
    bucket_rankings: dict,
    cosine_percentage_threshold: float,
    fixed_k=500,
):
    """
    bucket_rankings[bucket] = list of dicts:
        {
            "scores": np.ndarray,
            "pmids": np.ndarray,
            "positives": set
        }
    """

    def bucket_key(bucket_str: str):
        start, end = bucket_str.split("-")
        return int(start), int(end)

    buckets_sorted = sorted(bucket_rankings.keys(), key=bucket_key)

    methods = {
        f"Fixed k={fixed_k}": lambda npos, scores: fixed_k,
        "Positive-dependent k": lambda npos, scores: math.ceil(select_k_positive_dependent(npos)),
        f"Cosine-threshold ({cosine_percentage_threshold*100:.2f}%) k": lambda npos, scores: select_k_cosine_threshold(
            scores, cosine_percentage_threshold
        ),
    }

    # base colors per strategy using centralized config
    colors = {
        f"Fixed k={fixed_k}": COLORS["fixed_k"],
        "Positive-dependent k": COLORS["pos_count_k"],
        f"Cosine-threshold ({cosine_percentage_threshold*100:.2f}%) k": COLORS["cosine_k"],
    }

    precision_means = {name: [] for name in methods}
    recall_means = {name: [] for name in methods}
    debug_metrics = {name: {">=50": [], "<50": []} for name in methods}
    for bucket in buckets_sorted:
        per_method_p = {name: [] for name in methods}
        per_method_r = {name: [] for name in methods}

        for entry in bucket_rankings[bucket]:
            scores = entry["scores"]
            pmids = entry["pmids"]
            positives = entry["positives"]

            num_positives = len(positives)

            for name, k_fn in methods.items():
                k = max(1, k_fn(num_positives, scores))
                k = min(k, len(pmids))

                p, r = compute_precision_recall_at_k(pmids, positives, k)
                f3 = f_beta(p, r, beta=3)
                per_method_p[name].append(p)
                per_method_r[name].append(r)
                
                # Store debug info
                bucket_key_dbg = ">=50" if num_positives >= 50 else "<50"
                debug_metrics[name][bucket_key_dbg].append((p, r, f3))

        for name in methods:
            precision_means[name].append(np.mean(per_method_p[name]))
            recall_means[name].append(np.mean(per_method_r[name]))

    # --- debug print ---
    for name in methods:
        for key in [">=50", "<50"]:
            metrics = debug_metrics[name][key]
            if metrics:
                ps, rs, f3s = zip(*metrics)
                print(f"{name} | {key} positives: "
                      f"Precision={np.mean(ps):.4f}, "
                      f"Recall={np.mean(rs):.4f}, "
                      f"F3={np.mean(f3s):.4f}")


    x = np.arange(len(buckets_sorted))

    plt.figure()

    for name in methods:
        plt.plot(
            x,
            precision_means[name],
            marker="o",
            linestyle="-",
            color=colors[name],
            label=f"{name} – Precision",
        )
        plt.plot(
            x,
            recall_means[name],
            marker="o",
            linestyle="--",
            color=colors[name],
            alpha=0.6,
            label=f"{name} – Recall",
        )

    plt.xticks(x, buckets_sorted, rotation=45)
    plt.xlabel("Bucket (number of positives)")
    plt.ylabel("Score")
    plt.title("Precision and Recall by bucket for different top-k selection strategies")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "precision_recall_by_bucket.png"), dpi=300)
    plt.close()

def collect_samples_with_min_positives(
    bucket_rankings: dict,
    min_positives: int = 50,
):
    """
    Collect all samples with at least `min_positives` positives.
    Returns a flat list of entries.
    """
    samples = []

    for bucket_entries in bucket_rankings.values():
        for entry in bucket_entries:
            if len(entry["positives"]) >= min_positives:
                samples.append(entry)

    return samples

def find_min_cosine_threshold_for_recall(
    bucket_rankings: dict,
    min_positives: int = 50,
    target_recall: float = 0.7,
    step: float = 0.001,
    threshold_min: float = 0.0,
    threshold_max: float = 0.2,
):
    samples = collect_samples_with_min_positives(
        bucket_rankings, min_positives
    )

    if len(samples) == 0:
        raise ValueError("No samples with required minimum positives.")

    thresholds = np.arange(threshold_min, threshold_max + step, step)

    for cosine_threshold in thresholds:
        precisions, recalls, f3_scores = [], [], []

        for entry in samples:
            scores = entry["scores"]
            pmids = entry["pmids"]
            positives = entry["positives"]

            k = select_k_cosine_threshold(scores, cosine_threshold)
            k = max(1, min(k, len(pmids)))

            p, r = compute_precision_recall_at_k(pmids, positives, k)
            f3 = f_beta(p, r, beta=3)
            recalls.append(r)
            precisions.append(p)
            f3_scores.append(f3)

        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f3 = np.mean(f3_scores)
        
        if mean_recall >= target_recall:
            return cosine_threshold, mean_precision, mean_recall, mean_f3

    return None, None, None, None

def find_min_fixed_k_for_recall(
    bucket_rankings: dict,
    min_positives: int = 50,
    target_recall: float = 0.7,
    step: int = 1,
    k_min: int = 1,
    k_max: int = 5000,
):
    samples = collect_samples_with_min_positives(
        bucket_rankings, min_positives
    )

    if len(samples) == 0:
        raise ValueError("No samples with required minimum positives.")

    for k in range(k_min, k_max + 1, step):
        precisions, recalls, f3_scores = [], [], []

        for entry in samples:
            pmids = entry["pmids"]
            positives = entry["positives"]

            k_eff = min(k, len(pmids))
            p, r = compute_precision_recall_at_k(pmids, positives, k_eff)
            f3 = f_beta(p, r, beta=3)
            precisions.append(p)
            recalls.append(r)
            f3_scores.append(f3)
            

        
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f3 = np.mean(f3_scores)

        if mean_recall >= target_recall:
            return k, mean_precision, mean_recall, mean_f3

    return None, None, None, None

def plot_actual_topk_by_bucket(bucket_rankings: dict, cosine_percentage_threshold: float, fixed_k=500):
    """
    For each top-k selection strategy, plot the average k used per bucket.
    """
    def bucket_key(bucket_str: str):
        start, end = bucket_str.split("-")
        return int(start), int(end)

    buckets_sorted = sorted(bucket_rankings.keys(), key=bucket_key)

    methods = {
        f"Fixed k={fixed_k}": lambda npos, scores: fixed_k,
        "Positive-dependent k": lambda npos, scores: math.ceil(select_k_positive_dependent(npos)),
        f"Cosine-threshold ({cosine_percentage_threshold*100:.2f}%) k": lambda npos, scores: select_k_cosine_threshold(
            scores, cosine_percentage_threshold
        ),
    }

    # base colors per strategy using centralized config
    colors = {
        f"Fixed k={fixed_k}": COLORS["fixed_k"],
        "Positive-dependent k": COLORS["pos_count_k"],
        f"Cosine-threshold ({cosine_percentage_threshold*100:.2f}%) k": COLORS["cosine_k"],
    }

    avg_ks = {name: [] for name in methods}

    for bucket in buckets_sorted:
        for name, k_fn in methods.items():
            ks_in_bucket = []
            for entry in bucket_rankings[bucket]:
                scores = entry["scores"]
                num_positives = len(entry["positives"])
                k = max(1, min(k_fn(num_positives, scores), len(entry["pmids"])))
                ks_in_bucket.append(k)
            avg_ks[name].append(np.mean(ks_in_bucket))

    x = np.arange(len(buckets_sorted))

    plt.figure()
    for name in methods:
        plt.plot(
            x,
            avg_ks[name],
            marker="o",
            linestyle="-",
            color=colors[name],
            label=name,
        )

    plt.xticks(x, buckets_sorted, rotation=45)
    plt.xlabel("Bucket (number of positives)")
    plt.ylabel("Average top-k used")
    plt.title("Average top-k per bucket for different selection strategies")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "actual_topk_by_bucket.png"), dpi=300)
    plt.close()

SAVE_DIR = "../master-thesis-writing/writing/thesis/images/graphs"
MAX_K = 10000  # how far you want the curve
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    bucket_scores = defaultdict(list)
    bucket_positives_scores = defaultdict(list)
    bucket_rankings = defaultdict(list)
    npz_files = ranking_file_path(
        retriever_name="pubmedbert",
        query_type="title_abstract",
        total_docs=503679,
    )
    dataset_details = get_dataset_details()
    assert len(dataset_details) == 295

    for npz_path in sorted(npz_files):
        review_id = npz_path.stem  # removes .npz
        positives = dataset_details[review_id]["positives"]

        bucket = find_bucket(len(positives), BUCKETS)
        if bucket is None:
            continue

        with np.load(npz_path) as data:
            scores = data["scores"][:MAX_K]
            pmids = data["ids"][:MAX_K]

            # pad if ranking shorter than MAX_K
            if len(scores) < MAX_K:
                scores = np.pad(
                    scores, (0, MAX_K - len(scores)), constant_values=np.nan
                )

            bucket_scores[bucket].append(scores)

            # extract scores corresponding to positives
            mask = np.isin(pmids, positives)
            bucket_positives_scores[bucket].append(scores[mask])
            bucket_rankings[bucket].append(
                {
                    "scores": scores,
                    "pmids": pmids,
                    "positives": positives,
                }
            )

    cos_thr, cos_prec, cos_recall, cos_f3 = find_min_cosine_threshold_for_recall(
        bucket_rankings,
        min_positives=50,
        target_recall=0.7,
        threshold_min=0.02,
        step=0.0001,
    )

    
    print(f"Cosine-threshold method:")
    print(f"  min threshold = {cos_thr}")
    print(f"  mean precision = {cos_prec:.4f}")
    print(f"  mean recall    = {cos_recall:.4f}")
    print(f"  mean F3-score  = {cos_f3:.4f}\n")


    fixed_k, cos_prec, cos_recall, cos_f3 = find_min_fixed_k_for_recall(
        bucket_rankings,
        min_positives=50,
        target_recall=0.7,
        step=1,
        k_min=700,
        k_max=5000,
    )

    print(f"Fixed-k method:")
    print(f"  min k       = {fixed_k}")
    print(f"  mean precision = {cos_prec:.4f}")
    print(f"  mean recall    = {cos_recall:.4f}")
    print(f"  mean F3-score  = {cos_f3:.4f}\n")
    
    plot_metric_score_curve_by_bucket(bucket_scores=bucket_scores, out_folder=None)
    plot_positive_score_stats_by_bucket(bucket_positives_scores)
    plot_precision_recall_by_bucket(
        bucket_rankings, cosine_percentage_threshold=cos_thr, fixed_k=fixed_k
    )
    plot_actual_topk_by_bucket(bucket_rankings, cosine_percentage_threshold=cos_thr, fixed_k=fixed_k)

    
    

    
    
    
