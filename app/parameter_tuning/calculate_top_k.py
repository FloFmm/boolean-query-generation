import pandas as pd

def rank_at_recall_threshold(csv_path: str, p: float):
    """
    Given a CSV with recall@k columns (named 'recall@k'), return
    the smallest k for each row where recall >= p.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    p : float
        Recall threshold (0 to 1)

    Returns
    -------
    pandas.Series
        Series with the k value for each row where recall >= p.
    """
    df = pd.read_csv(csv_path)

    # Select only recall@k columns
    recall_cols = [c for c in df.columns if c.startswith("recall@")]
    
    # Extract k as float from column names: recall@10 -> 10
    k_values = [float(c.split("@")[1]) for c in recall_cols]

    recall_df = df[recall_cols].astype(float)

    # Boolean DataFrame: True where recall >= p
    mask = recall_df.ge(p)

    # For each row, get the first k where recall >= p
    first_k = mask.apply(lambda row: next((k for k, m in zip(k_values, row) if m), float('inf')), axis=1)

    result = {}
    for n_pos, group_idx in df.groupby("n_positives").indices.items():
        k_values_group = first_k.iloc[group_idx]
        n_samples = len(k_values_group)
        avg_k = k_values_group.mean()  # you can also take median if you prefer
        result[n_pos] = {"k_at_p": avg_k, "n_samples": n_samples}

    return result

if __name__ == "__main__":
    csv_path = "boolean-query-generation/data/reports/title_and_abstract/pubmedbert_abstract_docs=433660_by_pos_count.csv"
    p = 0.7  # 80% recall

    k_at_80pct = rank_at_recall_threshold(csv_path, p)
    print(k_at_80pct)