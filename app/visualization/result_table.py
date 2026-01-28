import json
import csv
import os
from collections import defaultdict
from app.helper.helper import f_beta
from app.dataset.utils import review_id_to_dataset, dataset_names

def process_jsonl_folder(folder_path, output_csv):
    # Dictionary to store sum of values and counts
    # Key is now (dataset, bucket, file)
    stats = defaultdict(lambda: defaultdict(list))

    # Walk through folder and subfolders
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith("qg_results.jsonl"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)

                meta_path = os.path.join(root, "qg_meta_data.json")
                betas_str = ""

                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        meta = json.load(mf)
                        betas = sorted(meta.get("betas", {}).keys(), key=int)
                        betas_str = ",".join(map(str, betas))
                else: 
                    print("warning meta file missing, value get lost if thats the case")
                    continue

                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        dataset, _, _ = review_id_to_dataset(data["query_id"])
                        if dataset == "tar2017":
                            dataset = "tar2018"  # tar2017 is part of 2018

                        num_positive_bucket = "<50" if data["num_positive"] < 50 else ">=50"
                        key = (dataset, num_positive_bucket, relative_path, betas_str)  # include filename in key

                        # Top-level numerical values
                        for field in [
                            "num_positive",
                            "top_k",
                            "pubmed_retrieved",
                            "pubmed_precision",
                            "pubmed_recall",
                            "subset_retrieved",
                            "subset_precision",
                            "subset_recall",
                            "pseudo_precision",
                            "pseudo_recall",
                            "optimization_score",
                            "qg_time_seconds",
                        ]:
                            value = data.get(field, 0)
                            if value is None:
                                value = 0
                            stats[key][field].append(value)

                        # query_size values
                        for field, value in data.get("query_size", {}).items():
                            if value is not None:
                                stats[key][f"query_size_{field}"].append(value)

                        # Calculate F1 and F3 for pubmed
                        p = data.get("pubmed_precision", 0) or 0
                        r = data.get("pubmed_recall", 0) or 0
                        stats[key]["pubmed_f1"].append(f_beta(precision=p, recall=r, beta=1))
                        stats[key]["pubmed_f3"].append(f_beta(precision=p, recall=r, beta=3))

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        field_keys = list(next(iter(stats.values())).keys())
        header = ["dataset", "num_positive_bucket", "source_file", "selection_betas"] + field_keys
        writer.writerow(header)

        for (dataset, bucket, file_path, beta_str), fields in stats.items():
            row = [dataset, bucket, file_path, beta_str]
            for field in field_keys:
                values = [v for v in fields.get(field, []) if v is not None]
                avg = sum(values) / len(values) if values else 0
                row.append(avg)
            writer.writerow(row)

def generate_typst_table(csv_file, typst_file, baseline_dict, betas=None):
    """
    Generates a Typst table from a CSV containing metrics and a baseline dictionary.

    Args:
        csv_file (str): Path to the input CSV file.
        typst_file (str): Path where the Typst table will be written.
        baseline_dict (dict): Dictionary of baselines for each dataset.
                              Format: {dataset: [(name, p, f1, f3, r), ...]}
    """
    # Read CSV into nested dict: stats[dataset][bucket] = row_dict
    stats = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            bucket = row["num_positive_bucket"]
            
            # If betas is set, skip rows that don't contain any of the selected betas
            if betas is not None:
                row_betas = set(row["selection_betas"].split(","))
                if not row_betas & betas:
                    continue  # skip this row entirely
            
            stats.setdefault(dataset, {}).setdefault(bucket, []).append(row)
            
    def config_name(row, betas):
        parts = row["selection_betas"].split(",")
        if betas is not None:
            c_name = "F" + ", ".join(sorted(set(parts) & betas))
        else:
            c_name = f"{parts[0]}-{parts[-1]}" if len(parts) > 1 else parts[0]
        
        return c_name
    
    with open(typst_file, "w") as f:
        # Table preamble
        f.write("#set table(\n")
        f.write("  stroke: (x, y) => (top: 0.5pt, bottom: 0.5pt),\n")
        f.write("  align: horizon + center\n")
        f.write(")\n\n")

        f.write("#table(\n")
        f.write("  columns: 7,\n")
        f.write("  table.header([], [], [Prompt], table.vline(start:0, stroke:(thickness:0.5pt)), [Precision], [F1], [F3], [Recall]),\n")
        seperator = "  table.cell(colspan: 7, inset: (top: 1pt, bottom: 1pt))[],\n"
        for i, dataset in enumerate(["tar2018", "tar2019", "sigir2017", "sr_updates"]):#enumerate(stats.items())]
            buckets = stats[dataset]
            if i != 0:
                f.write(seperator)
            # Determine rowspans
            # baseline_count = len(baseline_dict.get(dataset, []))
            # config_rows_ge50 = 1 if ">=50" in buckets else 0
            # config_rows_le50 = 1 if "<50" in buckets else 0
            # config_rows_all = 1 if "all" in buckets else 0
            baseline_count = len(baseline_dict.get(dataset, []))
            config_rows_ge50 = len(buckets.get(">=50", []))
            config_rows_le50 = len(buckets.get("<50", []))
            config_rows_all = len(buckets.get("all", []))
            total_rows = baseline_count + config_rows_ge50 + config_rows_le50 + config_rows_all

            # Rotated dataset label
            f.write(f"  table.cell(rowspan: {total_rows}, rotate(-90deg, reflow:true)[{dataset_names(dataset)}]),\n")

            # Compute best values for bolding
            best_metrics = {"Precision": -1, "F1": -1, "F3": -1, "Recall": -1}
            for bucket in buckets.values():
                for row in bucket:
                    for m, key in zip(["Precision","F1","F3","Recall"],
                                      ["pubmed_precision","pubmed_f1","pubmed_f3","pubmed_recall"]):
                        best_metrics[m] = max(best_metrics[m], float(row[key]))
            for m, key in zip(["Precision","F1","F3","Recall"],
                              ["pubmed_precision","pubmed_f1","pubmed_f3","pubmed_recall"]):
                if baseline_count > 0:
                    for _, p, f1, f3, r in baseline_dict[dataset]:
                        val = {"Precision":p,"F1":f1,"F3":f3,"Recall":r}[m]
                        best_metrics[m] = max(best_metrics[m], val)

            def fmt(x, metric=None):
                x = float(x)
                if metric is not None and x == best_metrics[metric]:
                    return f"*{x:.4f}*"
                return f"{x:.4f}"

            # Baselines
            if baseline_count > 0:
                f.write(f"  table.cell(rowspan: {baseline_count})[Baselines],\n")
                for name, p, f1, f3, r in baseline_dict[dataset]:
                    f.write(f"    [{name}], [{fmt(p,'Precision')}], [{fmt(f1,'F1')}], [{fmt(f3,'F3')}], [{fmt(r,'Recall')}],\n")

            # Buckets
            for bucket_name in [">=50","<50","all"]:
                if bucket_name in buckets:
                    rows = sorted(buckets[bucket_name], key=lambda r: int(r["selection_betas"].split(",")[0]))
                    f.write(f"  table.cell(rowspan:{len(rows)})[{bucket_name} pos],\n".replace('<', '\<'))
                    for row in rows:
                        c_name = config_name(row, betas)
                        f.write(
                            f"    [{c_name}], "
                            f"[{fmt(row['pubmed_precision'],'Precision')}], "
                            f"[{fmt(row['pubmed_f1'],'F1')}], "
                            f"[{fmt(row['pubmed_f3'],'F3')}], "
                            f"[{fmt(row['pubmed_recall'],'Recall')}],\n"
                        )
                    
        f.write(")\n")

    print(f"Typst table written to {typst_file}")

if __name__ == "__main__":
    baseline_dict = { # TODO find real data
        "tar2018": [
            ("Original", 0.0217, 0.0407, 0.1439, 0.9338), # original, conceptional and obj all from https://bevankoopman.github.io/papers/irj2020-comparison.pdf (same as in other verions of that paper, was also chosen as source from ChatGPT paper)
            ("Conceptual", 0.0021, 0.0037, 0.0114, 0.6286), # highest recall, highest f3
            ("Objective", 0.0002, 0.0005, 0.0022, 0.8780), # highest recall (since highest f3 has very low recall)
            ("ChatGPT", 0.0752, 0.0642, 0.0847, 0.5035), # https://arxiv.org/pdf/2302.03495, highest recall, highest F3, with example q4
        ],
        "tar2019": [ # no value found
            # ("Original", "≤0.012\*", "", "", ""),
        ],
        "sr_updates": [
            # ("Original", "≤0.004\*", "", "", ""),
        ],
        "sigir2017": [
            # ("Original", "≤0.089\*", "", "", ""),
        ],
    }
    
    csv_path = "data/statistics/final/best/best_average.csv"
    process_jsonl_folder(
        folder_path="data/statistics/optuna/best_old", #TODO remove "_old" 
        output_csv=csv_path,
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file="data/statistics/final/best/best_average.typ",
        baseline_dict=baseline_dict,
        betas={"3","15","30","50"},
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file="data/statistics/final/best/best_average_all.typ",
        baseline_dict=baseline_dict,
    )
    print("done")
