import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_recall_at_k_by_bucket(
    buckets: list[tuple],
    csv_path: str,
    bucket_size: int = 5,
    title: str | None = None,
):
    """
    Plot recall@k curves grouped by number-of-positives buckets.
    X-axis: k
    Y-axis: weighted average recall@k
    """

    df = pd.read_csv(csv_path)

    # --------------------------------------------------
    # extract recall@k columns
    # --------------------------------------------------
    rec_cols = [c for c in df.columns if c.startswith("recall@")]

    def get_k(col):
        return int(col.split("@")[1])

    rec_cols = sorted(rec_cols, key=get_k)
    ks = [get_k(c) for c in rec_cols]

    # --------------------------------------------------
    # assign buckets
    # --------------------------------------------------
    # def bucket_label(n):
    #     lo = ((n - 1) // bucket_size) * bucket_size + 1
    #     hi = lo + bucket_size - 1
    #     return f"{lo}-{hi}"
    def assign_bucket(n):
        for lo, hi in buckets:
            if lo <= n <= hi:
                return f"{lo}-{hi}"
        return None

    df["bucket"] = df["n_positives"].apply(assign_bucket)

    # --------------------------------------------------
    # prepare sorted buckets
    # --------------------------------------------------
    def bucket_start(bucket):
        return int(bucket.split("-")[0])

    grouped = sorted(
        df.groupby("bucket"),
        key=lambda x: bucket_start(x[0]),
    )

    # --------------------------------------------------
    # plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))

    for bucket, g in grouped:
        weights = g["n_queries"].values
        n_examples = weights.sum()

        recall_vals = [
            np.average(g[col].values, weights=weights)
            for col in rec_cols
        ]

        plt.plot(
            ks,
            recall_vals,
            marker="o",
            label=f"{bucket} positives (n={n_examples})",
        )

    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(title="Query buckets")

    if title:
        plt.title(title)

    plt.tight_layout()

    # --------------------------------------------------
    # save image
    # --------------------------------------------------
    out_img = os.path.splitext(csv_path)[0] + "_recall_at_k.jpg"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved recall@k plot to: {out_img}")

if __name__ == "__main__":
    plot_recall_at_k_by_bucket(
        csv_path="data/reports/title_and_abstract/pubmedbert_abstract_docs=433660_by_pos_count.csv",
        bucket_size=20,
        title="recall–Recall by Number of Positives",
        buckets=[(1,1),(2,3),(4,6),(7,10),(11,15),(16,20),(21,30),(31,50),(51,75),(76,100),(101,150),(151,250),(251,500),(501,750),(751,1000),(1001,1500)]
    )