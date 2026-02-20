import json
import csv
import os
import statistics
from collections import defaultdict
from app.config.config import CURRENT_BEST_RUN_FOLDER, RESULT_TABLE_OPERATOR_METRICS_ORDERED, RESULT_TABLE_PERFORMANCE_METRICS_ORDERED, RESULT_TABLE_PERFORMANCE_METRICS, RESULT_TABLE_OPERATOR_METRICS
from app.helper.helper import f_beta
from app.dataset.utils import review_id_to_dataset, dataset_names
from app.tree_learning.query_generation import query_size_value



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
                print(betas_str, sum(1 for _ in open(file_path, "r")))
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        dataset, _, _ = review_id_to_dataset(data["query_id"])
                        if dataset == "tar2017":
                            dataset = "tar2018"  # tar2017 is part of 2018

                        num_positive_bucket = (
                            "\<50" if data["num_positive"] < 50 else "\>\=50"
                        )
                        key = (
                            dataset,
                            num_positive_bucket,
                            relative_path,
                            betas_str,
                        )  # include filename in key

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
                        # stats[key]["logical_operators"].append(query_size_value(data["query_size"])) wrong for old data
                        check = (
                            sum(
                                data["query_size"][k]
                                for k in [
                                    "paths",
                                    "ANDs",
                                    "NOTs",
                                    "added_ORs",
                                    "synonym_ORs",
                                ]
                            )
                            - 1
                        )
                        count_value = (
                            data["pubmed_query"].count("AND")
                            + data["pubmed_query"].count("OR")
                            + data["pubmed_query"].count("NOT")
                        )
                        assert check == count_value
                        stats[key]["logical_operators"].append(count_value)
                        stats[key]["all_ORs"].append(data["pubmed_query"].count("OR"))
                        # Calculate F1 and F3 for pubmed
                        p = data.get("pubmed_precision", 0) or 0
                        r = data.get("pubmed_recall", 0) or 0
                        stats[key]["pubmed_f1"].append(
                            f_beta(precision=p, recall=r, beta=1)
                        )
                        stats[key]["pubmed_f3"].append(
                            f_beta(precision=p, recall=r, beta=3)
                        )

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header - include stddev columns alongside each metric
        field_keys = list(next(iter(stats.values())).keys())
        header = [
            "dataset",
            "num_positive_bucket",
            "source_file",
            "selection_betas",
        ]
        for field in field_keys:
            header.append(field)
            header.append(f"{field}_stddev")
        writer.writerow(header)

        for (dataset, bucket, file_path, beta_str), fields in stats.items():
            row = [dataset, bucket, file_path, beta_str]
            for field in field_keys:
                values = [v for v in fields.get(field, []) if v is not None]
                avg = sum(values) / len(values) if values else 0
                # Calculate standard deviation if we have more than one value
                stddev = statistics.stdev(values) if len(values) > 1 else 0
                row.append(avg)
                row.append(stddev)
            writer.writerow(row)


def generate_typst_table(
    csv_file,
    typst_file,
    baseline_dict,
    betas=None,
    metrics=None,
    text_size=10.3,
    min_positive_buckets=["\<50", "\>\=50"],
    used_datasets=["tar2018", "tar2019", "sigir2017", "sr_updates"],
    show_performance=True,
    show_operators=True,
    table_name="result_table",
):
    """
    Generates a Typst table from a CSV containing metrics and a baseline dictionary.

    Args:
        csv_file (str): Path to the input CSV file.
        typst_file (str): Path where the Typst table will be written.
        baseline_dict (dict): Dictionary of baselines for each dataset.
                              Format: {dataset: [(name, p, f1, f3, r, ops), ...]}
        metrics (dict | None): Ordered mapping of display name to metric config.
                                Each config supports:
                                  - key (str): CSV field name
                                  - direction ("max" | "min"): for best-value bolding
                                  - fmt (str): format string for values
                                  - baseline_index (int | None): index in baseline tuple
                                  - vline_before (bool, optional): add vline before column
    """
    typst_file = os.path.join(typst_file, f"{table_name}.typ")
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

    # Filter metrics based on show_performance and show_operators flags
    filtered_metrics = {}
    performance_metrics = RESULT_TABLE_PERFORMANCE_METRICS_ORDERED
    operator_metrics = RESULT_TABLE_OPERATOR_METRICS_ORDERED
    
    for name, cfg in metrics.items():
        include = True
        if name in performance_metrics and not show_performance:
            include = False
        if name in operator_metrics and not show_operators:
            include = False
        if include:
            filtered_metrics[name] = cfg
    
    metrics = filtered_metrics

    def config_name(row, betas):
        parts = row["selection_betas"].split(",")
        source_file = row["source_file"]
        if "ktype=cosine" in source_file:
            ktype = "cosine"
        elif "ktype=pos_count" in source_file:
            ktype = "\#pos"
        elif "ktype=fixed" in source_file:
            ktype = "fixed"
        algo_name = "#algo-name-short\-F"
        if betas is not None:
            algo_name += ", ".join(sorted(set(parts) & betas))
        else:
            algo_name += f"{parts[0]}-{parts[-1]}" if len(parts) > 1 else parts[0]
        return algo_name, ktype

    with open(typst_file, "w") as f:
        # Table preamble
        f.write('#import "../../thesis/assets/assets.typ": *\n')

        f.write(f"#let {table_name}() = [\n")
        f.write("#table(\n")
        f.write(f"  columns: {3 + len(metrics)},\n")
        header_cells = [
            "[]",
            "[]",
            "[Method]",
            "table.vline(start:0, stroke:(thickness:0.5pt))",
        ]
        for name, cfg in metrics.items():
            if cfg.get("vline_before"):
                header_cells.append("table.vline(start:0, stroke:(thickness:0.5pt))")
            header_cells.append(f"[{name}]")
        f.write(f"  table.header({', '.join(header_cells)}),\n")
        seperator = f"  table.cell(colspan: {3 + len(metrics)}, inset: (top: 1pt, bottom: 1pt))[],\n"
        for i, dataset in enumerate(
            used_datasets
        ):  # enumerate(stats.items())]
            buckets = stats.get(dataset, {})
            baseline_count = len(baseline_dict.get(dataset, []))
            
            # Pre-count displayed rows to calculate correct total_rows
            displayed_baseline_count = 0
            if baseline_count > 0:
                for baseline in baseline_dict[dataset]:
                    # Check if baseline has any non-None values for filtered metrics
                    has_value = False
                    for name, cfg in metrics.items():
                        value = None
                        
                        # Handle new dict format: {"name": "...", "Metric": [mean, stddev], ...}
                        if isinstance(baseline, dict):
                            if name in baseline:
                                metric_data = baseline[name]
                                value = metric_data[0] if isinstance(metric_data, (list, tuple)) and len(metric_data) > 0 else metric_data
                        # Handle old tuple format
                        else:
                            idx = cfg.get("baseline_index")
                            if idx is not None and len(baseline) > idx:
                                value = baseline[idx]
                        
                        if value is not None:
                            has_value = True
                            break
                    if has_value:
                        displayed_baseline_count += 1
            
            displayed_config_count = 0
            for bucket_name in min_positive_buckets:
                if bucket_name in buckets:
                    for row in buckets[bucket_name]:
                        # Check if row has any non-empty values for filtered metrics
                        has_value = False
                        for name, cfg in metrics.items():
                            key = cfg.get("key")
                            if key is not None:
                                val = row.get(key)
                                if val not in (None, ""):
                                    has_value = True
                                    break
                        if has_value:
                            displayed_config_count += 1
            
            total_rows = displayed_baseline_count + displayed_config_count
            
            # Skip dataset if there are no rows to display
            if total_rows == 0:
                continue
            
            if i != 0:
                f.write(seperator)

            # Rotated dataset label
            f.write(
                f"  table.cell(rowspan: {total_rows}, rotate(-90deg, reflow:true)[{dataset_names(dataset)}]),\n"
            )

            # Compute best values for bolding (only from displayed buckets)
            best_metrics = {}
            for name, cfg in metrics.items():
                direction = cfg.get("direction", "max")
                best_metrics[name] = (
                    float("inf") if direction == "min" else float("-inf")
                )

            # Only consider rows from buckets that will be displayed
            for bucket_name in min_positive_buckets:
                if bucket_name in buckets:
                    for row in buckets[bucket_name]:
                        for name, cfg in metrics.items():
                            key = cfg.get("key")
                            if key is None:
                                continue
                            raw_val = row.get(key)
                            if raw_val in (None, ""):
                                continue
                            try:
                                val = float(raw_val)
                            except (TypeError, ValueError):
                                continue
                            if cfg.get("direction", "max") == "min":
                                best_metrics[name] = min(best_metrics[name], val)
                            else:
                                best_metrics[name] = max(best_metrics[name], val)

            if baseline_count > 0:
                for name, cfg in metrics.items():
                    idx = cfg.get("baseline_index")
                    if idx is None:
                        continue
                    for baseline in baseline_dict[dataset]:
                        value = None
                        
                        # Handle new dict format
                        if isinstance(baseline, dict):
                            if name in baseline:
                                metric_data = baseline[name]
                                value = metric_data[0] if isinstance(metric_data, (list, tuple)) and len(metric_data) > 0 else metric_data
                        # Handle old tuple format
                        else:
                            if len(baseline) > idx:
                                value = baseline[idx]
                        
                        if value is None:
                            continue
                        if cfg.get("direction", "max") == "min":
                            best_metrics[name] = min(best_metrics[name], float(value))
                        else:
                            best_metrics[name] = max(best_metrics[name], float(value))

            def fmt(x, metric=None):
                if x is None:
                    return ""
                x = float(x)
                cfg = metrics.get(metric, {}) if metric is not None else {}
                fmt_str = cfg.get("fmt", "{:.4f}")
                formatted = fmt_str.format(x)

                best_val = best_metrics.get(metric)
                if best_val is None or best_val in (float("inf"), float("-inf")):
                    return formatted
                if metric is not None and abs(x - best_val) < 1e-6:
                    return f"*{formatted}*"
                return formatted
            
            # Baselines
            if baseline_count > 0:
                # Count baselines that will actually be displayed
                displayed_baseline_count = 0
                for baseline in baseline_dict[dataset]:
                    metric_cells = []
                    for m_name, cfg in metrics.items():
                        value = None
                        
                        # Handle new dict format: {"name": "...", "Metric": [mean, stddev], ...}
                        if isinstance(baseline, dict):
                            if m_name in baseline:
                                metric_data = baseline[m_name]
                                # Extract mean (first value)
                                value = metric_data[0] if isinstance(metric_data, (list, tuple)) and len(metric_data) > 0 else metric_data
                        # Handle old tuple format: (name, precision, f1, f3, recall, ...)
                        else:
                            idx = cfg.get("baseline_index")
                            if idx is not None and len(baseline) > idx:
                                value = baseline[idx]
                        
                        metric_cells.append(f"[{fmt(value, m_name)}]")
                    if not all(cell == "[]" for cell in metric_cells):
                        displayed_baseline_count += 1
                
                if displayed_baseline_count > 0:
                    f.write(
                        f"  table.cell(rowspan: {displayed_baseline_count}, rotate(-90deg, reflow:true)[Baselines]),\n"
                    )
                    for baseline in baseline_dict[dataset]:
                        # Get baseline name
                        if isinstance(baseline, dict):
                            name = baseline.get("name", "")
                        else:
                            name = baseline[0] if baseline else ""
                        
                        metric_cells = []
                        for m_name, cfg in metrics.items():
                            value = None
                            stddev = None
                            
                            # Handle new dict format
                            if isinstance(baseline, dict):
                                if m_name in baseline:
                                    metric_data = baseline[m_name]
                                    if isinstance(metric_data, (list, tuple)) and len(metric_data) >= 2:
                                        value = metric_data[0]
                                        stddev = metric_data[1]
                                    else:
                                        value = metric_data
                            # Handle old tuple format
                            else:
                                idx = cfg.get("baseline_index")
                                if idx is not None and len(baseline) > idx:
                                    value = baseline[idx]
                            
                            # Format as "value ± stddev" if both are present
                            if value is not None and stddev is not None:
                                formatted_value = fmt(value, m_name)
                                formatted_stddev = fmt(stddev, m_name)
                                cell = f"[{formatted_value} ± {formatted_stddev}]"
                            else:
                                cell = f"[{fmt(value, m_name)}]"
                            metric_cells.append(cell)
                        
                        # Skip baseline if all metrics are empty
                        if all(cell == "[]" for cell in metric_cells):
                            continue
                        
                        f.write(f"    [{name}], {', '.join(metric_cells)},\n")

            def config_type_order(row):
                source_file = row["source_file"]
                if "ktype=pos_count" in source_file:
                    return 0
                elif "ktype=cosine" in source_file:
                    return 1
                elif "ktype=fixed" in source_file:
                    return 2
                return 3

            # Collect all rows across all buckets and group by algo-name-Fscore with bucket label
            all_rows_by_algo = {}
            for bucket_name in min_positive_buckets:
                if bucket_name in buckets:
                    rows = sorted(
                        buckets[bucket_name],
                        key=lambda r: (
                            -int(r["selection_betas"].split(",")[0]),
                            config_type_order(r),
                        ),
                    )
                    
                    for row in rows:
                        algo_name, ktype = config_name(row, betas)
                        algo_name_with_bucket = f"{algo_name}"
                        if len(min_positive_buckets) > 1:
                            algo_name_with_bucket += f"{bucket_name}"
                        if algo_name_with_bucket not in all_rows_by_algo:
                            all_rows_by_algo[algo_name_with_bucket] = []
                        all_rows_by_algo[algo_name_with_bucket].append((row, ktype, bucket_name))
            
            # Render rows grouped by algo-name-Fscore (each spans 3)
            for algo_name in sorted(all_rows_by_algo.keys(), reverse=True, key=lambda x: int(x.split("-F")[-1].split("\\")[0])):
                rows_with_ktype = all_rows_by_algo[algo_name]
                
                # Filter to only display rows with non-empty metrics
                displayed_rows = []
                for row, ktype, bucket_name in rows_with_ktype:
                    metric_cells = []
                    for m_name, cfg in metrics.items():
                        key = cfg.get("key")
                        value = row.get(key) if key is not None else None
                        metric_cells.append(f"[{fmt(value, m_name)}]")
                    if not all(cell == "[]" for cell in metric_cells):
                        displayed_rows.append((row, ktype, bucket_name))
                
                if displayed_rows:
                    num_displayed = len(displayed_rows)
                    # Write algo_name with rowspan across all rows for this config
                    f.write(
                        f"  table.cell(rowspan: {num_displayed}, rotate(-90deg, reflow:true)[{algo_name}]),\n"
                    )
                    
                    for idx, (row, ktype, bucket_name) in enumerate(displayed_rows):
                        metric_cells = []
                        for m_name, cfg in metrics.items():
                            key = cfg.get("key")
                            value = row.get(key) if key is not None else None
                            stddev_key = f"{key}_stddev" if key else None
                            stddev = row.get(stddev_key) if stddev_key else None
                            
                            # Format as "value ± stddev" if both are present
                            if value is not None and stddev is not None:
                                formatted_value = fmt(value, m_name)
                                formatted_stddev = fmt(stddev, m_name)
                                cell = f"[{formatted_value} ± {formatted_stddev}]"
                            else:
                                cell = f"[{fmt(value, m_name)}]"
                            metric_cells.append(cell)
                        
                        f.write(f"    [{ktype}], {', '.join(metric_cells)},\n")

        f.write(")]\n")

    print(f"Typst table written to {typst_file}")


if __name__ == "__main__":

    # laod bseline dict from file
    with open("data/examples/baseline_values.json", "r") as f:
        baseline_dict = json.load(f)
    
    best_choice = CURRENT_BEST_RUN_FOLDER.split('/')[-1]
    csv_path = f"../master-thesis-writing/writing/tables/{best_choice}/best_average.csv"

    process_jsonl_folder(
        folder_path=f"data/statistics/optuna/{best_choice}",
        output_csv=csv_path,
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file=f"../master-thesis-writing/writing/tables/{best_choice}/",
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=True,
        show_operators=False,
        table_name="best_table",
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file=f"../master-thesis-writing/writing/tables/{best_choice}/",
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50", "\<50"],
        used_datasets=["tar2018", "tar2019", "sigir2017", "sr_updates"],
        show_performance=True,
        show_operators=True,
        table_name="best_table_appendix",
    )
    generate_typst_table(
        csv_file=csv_path,
        typst_file=f"../master-thesis-writing/writing/tables/{best_choice}/",
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=False,
        show_operators=True,
        table_name="best_table_operators",
    )
    # generate_typst_table(
    #     csv_file=csv_path,
    #     typst_file=f"../master-thesis-writing/writing/tables/{best_choice}/",
    #     baseline_dict=baseline_dict,
    #     metrics=metrics,
    #     text_size=10.3,
    #     show_performance=True,
    #     show_operators=True,
    #     table_name="best_all",
    # )

    print("done")
