import json
import csv
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

    # Write Typst table
    datasets = sorted(set(k[0] for k in stats.keys()))
    with open(typst_txt_path, "w") as f:
        for dataset in datasets:
            f.write(f"# {dataset}\n\n")

            f.write("#table(\n")
            f.write("  columns: 5,\n")
            f.write("  table.header(\n")
            f.write("    [], [Precision], [F1], [F3], [Recall]\n")
            f.write("  ),\n")

            # Baselines
            baselines = [
                ("Original", 0.0207, 0.0290, 0.0481, 0.8317),
                ("Conceptual", 0.0015, 0.0027, 0.0101, 0.6997),
                ("Objective", 0.0002, 0.0005, 0.0023, 0.9128),
            ]
            for name, p, f1, f3, r in baselines:
                f.write(f"  [{name}], [{p:.4f}], [{f1:.4f}], [{f3:.4f}], [{r:.4f}],\n")

            # Our results (average of "all")
            key = (dataset, "all")
            fields = stats[key]
            avg_p = sum(fields["pubmed_precision"]) / len(fields["pubmed_precision"])
            avg_r = sum(fields["pubmed_recall"]) / len(fields["pubmed_recall"])
            avg_f1 = sum(fields["pubmed_f1"]) / len(fields["pubmed_f1"])
            avg_f3 = sum(fields["pubmed_f3"]) / len(fields["pubmed_f3"])
            f.write(f"  [Our], [{avg_p:.4f}], [{avg_f1:.4f}], [{avg_f3:.4f}], [{avg_r:.4f}],\n")

            f.write(")\n\n")


if __name__ == "__main__":
    process_jsonl(
        "data/statistics/optuna/run_2_nodes_10tasks_1cpu_per_task/lc=True,maxdf=0.5,mesh=True,ma=True,mindf=100,rw=True,rmn=True,rmp=True,d=503679/boot,=True,cw=0.5,maxd=5,maxf=0.11,mof=10,maxs=None,midre=0.0255,midrs=0.0255,mwfl=0.0012,ne=50,pfs=1.1,rmf=2.4,rmidr=2.4,rweight=2.6,k=0.7,tkoc=1500/cost_factor=0.002,cover_beta=1.1,mh_noexp=True,min_rule_occ=0.03,min_rule_precision=0.01,min_tree_occ=0.05,pruning_beta=0.15,te=True,tiab=True/qg_results.jsonl",
        "data/statistics/final/best/best_average.csv",
        typst_txt_path="data/statistics/final/best/best_average.typst.txt",
    )
    print("done")
