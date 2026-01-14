import pandas as pd
import numpy as np

# BUCKETS = [(1,3),(4,10),(11,30),(31,100),(101,1500)]
BUCKETS = [(1,1),(2,3),(4,6),(7,10),(11,15),(16,20),(21,30),(31,50),(51,75),(76,100),(101,150),(151,250),(251,500),(501,750),(751,1000),(1001,1500)]
# BUCKETS = list(zip(range(1,500), range(1,500)))
CSV_PATH = "data/reports/title_and_abstract/pubmedbert_abstract_docs=433660_by_pos_count.csv"
def compute_weighted_metric_curve(
    csv_path: str,
    buckets: list[tuple],
    metric: str = "recall",
):
    """
    Compute weighted average {metric}@k per bucket.
    
    Returns:
        ks: list of k values
        bucket_labels: list of bucket labels
        bucket_vals: dict[bucket_label] = weighted average metric@k array
    """
    df = pd.read_csv(csv_path)

    # --------------------------------------------------
    # extract {metric}@k columns
    # --------------------------------------------------
    metric_cols = [c for c in df.columns if c.startswith(f"{metric}@")]
    def get_k(col): return int(col.split("@")[1])
    metric_cols = sorted(metric_cols, key=get_k)
    ks = [get_k(c) for c in metric_cols]

    # --------------------------------------------------
    # assign buckets
    # --------------------------------------------------
    def assign_bucket(n):
        for lo, hi in buckets:
            if lo <= n <= hi:
                return (lo, hi)#f"{lo}-{hi}"
        return None

    df["bucket"] = df["n_positives"].apply(assign_bucket)

    grouped = df.groupby("bucket")

    bucket_labels = []
    bucket_vals = {}

    for bucket, g in grouped:
        weights = g["n_queries"].values
        if len(weights) == 0:
            continue

        vals = np.array([
            np.average(g[col].values, weights=weights)
            for col in metric_cols
        ])
        bucket_labels.append(bucket)
        bucket_vals[bucket] = vals

    return ks, bucket_labels, bucket_vals

def compute_k_at_recall_threshold(
    ks: list[int],
    bucket_vals: dict[str, np.ndarray],
    p: float,
):
    """
    Compute interpolated k where metric reaches threshold p for each bucket.
    Linear interpolation between surrounding points.
    
    Returns:
        dict[bucket_label] = k_at_p
    """
    k_at_p = {}

    for bucket, vals in bucket_vals.items():
        vals = np.array(vals)
        # If p <= smallest value
        if p <= vals[0]:
            k_at_p[bucket] = ks[0]
            continue
        # If p >= largest value
        if p >= vals[-1]:
            k_at_p[bucket] = ks[-1]
            continue

        # Find indices around p
        idx = np.where(vals >= p)[0][0]
        k_hi = ks[idx]
        k_lo = ks[idx-1]
        v_hi = vals[idx]
        v_lo = vals[idx-1]

        # Linear interpolation
        k_interp = k_lo + (p - v_lo) / (v_hi - v_lo) * (k_hi - k_lo)
        k_at_p[bucket] = k_interp

    return k_at_p

def compute_top_ks(csv_path, p, buckets):
    # 1. Compute curve
    ks, bucket_labels, bucket_vals = compute_weighted_metric_curve(csv_path, buckets, metric="recall")

    # 3. Compute exact k for 70% and 95% recall
    ks = compute_k_at_recall_threshold(ks, bucket_vals, p)
    return ks

def approximate_y(x_vals, y_vals, x_query):
    """
    Approximate y for a given x using linear interpolation or extrapolation.

    Parameters
    ----------
    x_vals : list[float]
        Sorted x values
    y_vals : list[float]
        Corresponding y values
    x_query : float
        x value to query

    Returns
    -------
    float
        Approximated y value
    """
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have same length")
    if len(x_vals) < 2:
        raise ValueError("Need at least 2 points for interpolation/extrapolation")

    # Extrapolate if x_query is smaller than first x
    if x_query < x_vals[0]:
        x0, x1 = x_vals[0], x_vals[1]
        y0, y1 = y_vals[0], y_vals[1]
        return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)

    # Extrapolate if x_query is larger than last x
    if x_query > x_vals[-1]:
        x0, x1 = x_vals[-2], x_vals[-1]
        y0, y1 = y_vals[-2], y_vals[-1]
        return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)

    # Interpolate if inside the x range
    for i in range(1, len(x_vals)):
        if x_query <= x_vals[i]:
            x0, x1 = x_vals[i-1], x_vals[i]
            y0, y1 = y_vals[i-1], y_vals[i]
            return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)

    # Should never reach here
    raise RuntimeError("Interpolation failed")

def compute_top_k(n_positive, csv_path, p , buckets):
    ks = compute_top_ks(csv_path, p, buckets)
    return approximate_y(x_vals=[(k[1]+k[0])/2 for k in ks.keys()], y_vals=list(ks.values()), x_query=n_positive)

if __name__ == "__main__":
    ks = compute_top_ks(CSV_PATH, 0.7, BUCKETS)
    print([(k[1]+k[0])/2 for k in ks.keys()])
    print(list(ks.values()))
    print("k@70% recall:", ks)
    print()
    for num_pos in range(1, 200, 1):
        print(num_pos, compute_top_k(num_pos, CSV_PATH, 0.7, BUCKETS))