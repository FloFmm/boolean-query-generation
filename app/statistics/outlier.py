from app.config.config import CURRENT_BEST, CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import calc_missing_columns_in_result_df, get_qg_results
from app.helper.helper import f_beta
from app.visualization.tables.result_table import aggregate_results

import os
import json

def filter_jsonl_outliers(folder_path, outlier_save_path):
    outliers = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                filtered_lines = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue
                        pubmed_retrieved = data.get('pubmed_retrieved')
                        if pubmed_retrieved == 0 or (isinstance(pubmed_retrieved, (int, float)) and pubmed_retrieved > 200_000):
                            data['__source_path'] = file_path
                            outliers.append(data)
                        else:
                            filtered_lines.append(line)
                # Overwrite file with filtered lines
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(filtered_lines)
    # Save outliers
    if outliers:
        os.makedirs(os.path.dirname(outlier_save_path), exist_ok=True)
        with open(outlier_save_path, 'w', encoding='utf-8') as f:
            for item in outliers:
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    folder_path = CURRENT_BEST_RUN_FOLDER
    # Remove outlier lines from all jsonl files and collect them
    filter_jsonl_outliers(folder_path, "data/examples/outlier.jsonl")
    # Load and prepare DataFrame once
    base_df = get_qg_results(folder_path, query_ids=None)
    base_df = calc_missing_columns_in_result_df(base_df)
    print("original", len(base_df), "total samples in results dataframe")
    # fitler out all lines wher "pubmed_retrieved" is smaller than "top_k"/2 and bigger than top_k*10
    zero_results = base_df[
        (base_df["pubmed_retrieved"] == 0)
    ].copy()
    more_than_200_000 = base_df[
        (base_df["pubmed_retrieved"] > 200_000)
    ].copy()
    
    print(">200k results")
    print("avg precision", more_than_200_000["pubmed_precision"].mean(),)
    print("avg recall", more_than_200_000["pubmed_recall"].mean())
    print("avg pubmed_retrieved", more_than_200_000["pubmed_retrieved"].mean())
    print("num samples", len(more_than_200_000)/len(base_df)*100, "%")
    
    print("==0 retrieved")
    print("avg precision", zero_results["pubmed_precision"].mean(),)
    print("avg recall", zero_results["pubmed_recall"].mean())
    print("avg pubmed_retrieved", zero_results["pubmed_retrieved"].mean())
    print("num samples", len(zero_results)/len(base_df)*100, "%")
    
    # agg_df = aggregate_results(base_df)
    # print fromd ataframe all rows wher dataset is tar2018 and bucket is \>\=50 and only print the columns dataset, bucket, pubmed_precisison and pubmed_recall
    # print(agg_df[["dataset", "pubmed_retrieved", "num_positive_bucket", "pubmed_precision", "pubmed_recall", "pubmed_f50"]])