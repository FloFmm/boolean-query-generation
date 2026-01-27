import json
import csv
import os
from collections import defaultdict
from app.helper.helper import f_beta
from app.dataset.utils import review_id_to_dataset


def process_jsonl(file_path, output_csv, typst_txt_path=None):
    # Dictionary to store sum of values and counts
    stats = defaultdict(lambda: defaultdict(list))

    # Read file
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset, _, _ = review_id_to_dataset(data["query_id"])
            if dataset == "tar2017":
                dataset = "tar2018" #tar 2017 is aprt of 2018
            num_positive_bucket = "<50" if data["num_positive"] < 50 else ">=50"
            keys = [(dataset, num_positive_bucket), (dataset, "all")]  # add "all"

            for key in keys:
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
                    stats[key][field].append(data.get(field, 0))

                # query_size values
                for field, value in data.get("query_size", {}).items():
                    stats[key][f"query_size_{field}"].append(value)

                # Calculate F1 and F3 for pubmed
                p = data.get("pubmed_precision", 0)
                r = data.get("pubmed_recall", 0)
                stats[key]["pubmed_f1"].append(f_beta(precision=p, recall=r, beta=1))
                stats[key]["pubmed_f3"].append(f_beta(precision=p, recall=r, beta=3))

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ["dataset", "num_positive_bucket"] + list(
            next(iter(stats.values())).keys()
        )
        writer.writerow(header)

        for (dataset, bucket), fields in stats.items():
            row = [dataset, bucket]
            for field in header[2:]:
                values = fields.get(field, [])
                avg = sum(values) / len(values) if values else 0
                row.append(avg)
            writer.writerow(row)

def generate_typst_table(csv_file, typst_file, baseline_dict):
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
            stats.setdefault(dataset, {})[bucket] = row

    
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
        for i, (dataset, buckets) in enumerate(stats.items()):
            if i != 0:
                f.write(seperator)
            # Determine rowspans
            baseline_count = len(baseline_dict.get(dataset, []))
            config_rows_ge50 = 1 if ">=50" in buckets else 0
            config_rows_le50 = 1 if "<50" in buckets else 0
            config_rows_all = 1 if "all" in buckets else 0
            total_rows = baseline_count + config_rows_ge50 + config_rows_le50 + config_rows_all

            # Rotated dataset label
            f.write(f"  table.cell(rowspan: {total_rows}, rotate(-90deg, reflow:true)[{dataset.upper()}]),\n")

            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) else f"'{x}'"

            # Baselines
            if baseline_count > 0:
                f.write(f"  table.cell(rowspan: {baseline_count})[Baselines],\n")
                for name, p, f1, f3, r in baseline_dict[dataset]:
                    f.write(
                        f"    [{name}], [{fmt(p)}], [{fmt(f1)}], [{fmt(f3)}], [{fmt(r)}],\n"
                    )

            # >=50 bucket
            if ">=50" in buckets:
                f.write(f"  table.cell(rowspan:{config_rows_ge50})[>=50 pos],\n")
                row = buckets[">=50"]
                f.write(f"    [Config1], [{float(row['pubmed_precision']):.4f}], [{float(row['pubmed_f1']):.4f}], [{float(row['pubmed_f3']):.4f}], [{float(row['pubmed_recall']):.4f}],\n")

            # <=50 bucket
            if "<50" in buckets:
                f.write(f"  table.cell(rowspan:{config_rows_le50})[<=50 pos],\n")
                row = buckets["<50"]
                f.write(f"    [Config1], [{float(row['pubmed_precision']):.4f}], [{float(row['pubmed_f1']):.4f}], [{float(row['pubmed_f3']):.4f}], [{float(row['pubmed_recall']):.4f}],\n")

            # all bucket
            if "all" in buckets:
                f.write(f"  table.cell(rowspan:{config_rows_all})[all],\n")
                row = buckets["all"]
                f.write(f"    [Config1], [{float(row['pubmed_precision']):.4f}], [{float(row['pubmed_f1']):.4f}], [{float(row['pubmed_f3']):.4f}], [{float(row['pubmed_recall']):.4f}],\n")

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
    typst_txt_path="data/statistics/final/best/best_average.typst.txt"
    process_jsonl(
        "data/statistics/optuna/run_2_nodes_10tasks_1cpu_per_task/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.5,maxd=5,maxf=0.11,mof=10,maxs=None,midre=0.0255,midrs=0.0255,mwfl=0.0012,ne=50,pfs=1.1,rmf=2.4,rmidr=2.4,rweight=2.6,k=0.7,tkoc=1500/cost_factor=0.002,cover_beta=1.1,mh_noexp=True,min_rule_occ=0.03,min_rule_precision=0.01,min_tree_occ=0.05,pruning_beta=0.15,te=True,tiab=True/qg_results.jsonl",
        csv_path,
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file=typst_txt_path,
        baseline_dict=baseline_dict,
    )
    print("done")
