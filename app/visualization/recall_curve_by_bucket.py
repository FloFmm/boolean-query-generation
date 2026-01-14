import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from app.parameter_tuning.compute_top_k import compute_weighted_metric_curve, compute_top_ks, BUCKETS, CSV_PATH

def plot_metric_curve_by_bucket(
    csv_path: str,
    buckets: list[tuple],
    metric: str = "recall",
    title: str | None = None,
):
    ks, bucket_labels, bucket_vals = compute_weighted_metric_curve(csv_path, buckets, metric)

    plt.figure(figsize=(8,6))
    for bucket in bucket_labels:
        plt.plot(
            ks,
            bucket_vals[bucket],
            marker="o",
            label=f"{bucket} positives"
        )

    plt.xlabel("k")
    plt.ylabel(f"{metric}@k")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    if title:
        plt.title(title)

    plt.tight_layout()
    out_img = os.path.splitext(csv_path)[0] + f"_{metric}_at_k_curve.jpg"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {metric}@k curve plot to: {out_img}")

def plot_k_at_recall_thresholds_buckets(
    csv_path: str,
    ps: list[float],
    buckets: list[tuple],
    title: str | None = None,
):
    """
    Plot k@p-recall per bucket.

    Parameters
    ----------
    csv_path : str
        Path to CSV with recall@k values.
    ps : list[float]
        List of recall thresholds, e.g., [0.7, 0.95].
    buckets : list[tuple]
        List of buckets as (lo, hi) tuples.
    title : str | None
        Plot title.
    """

    # Compute k@p for all p
    k_at_ps = {}
    for p in ps:
        k_at_ps[p] = compute_top_ks(csv_path, p, buckets)

    # Prepare plot
    plt.figure(figsize=(10, 6))
    bucket_labels = [f"{lo}-{hi}" for lo, hi in buckets]

    for p, ks_dict in k_at_ps.items():
        ys = [ks_dict.get(f"{lo}-{hi}", float('nan')) for lo, hi in buckets]
        plt.plot(
            bucket_labels,
            ys,
            marker="o",
            label=f"k@{int(p*100)}% recall"
        )

    plt.xlabel("Number of positives (bucket)")
    plt.ylabel("k")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    if title:
        plt.title(title)

    plt.tight_layout()
    out_img = os.path.splitext(csv_path)[0] + "_k_at_recall_per_bucket.jpg"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved k@recall per bucket plot to: {out_img}")

if __name__ == "__main__":
    
    plot_metric_curve_by_bucket(
        csv_path=CSV_PATH,
        title="Recall by Number of Positives",
        metric="recall",
        buckets=BUCKETS
    )
    plot_metric_curve_by_bucket(
        csv_path=CSV_PATH,
        title="Precision by Number of Positives",
        metric="precision",
        buckets=BUCKETS
    )
    plot_k_at_recall_thresholds_buckets(
        csv_path=CSV_PATH,
        ps=[0.7, 0.3],
        buckets=BUCKETS,
        title="k@p-recall per bucket"
    )