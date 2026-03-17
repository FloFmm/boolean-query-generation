import json
import os
from app.config.config import (
    BASE_VARIATIONS,
    CURRENT_BEST,
    CURRENT_BEST_RUN_FOLDER,
    RESULT_TABLE_OPERATOR_METRICS_ORDERED,
    RESULT_TABLE_PERFORMANCE_METRICS_ORDERED,
    RESULT_TABLE_PERFORMANCE_METRICS,
    RESULT_TABLE_OPERATOR_METRICS,
    TRAIN_REVIEWS,
)
from app.dataset.utils import (
    calc_missing_columns_in_result_df,
    get_dataset_details,
    get_ktype,
    get_qg_results,
    dataset_names,
    review_id_to_dataset,
)


AGGREGATE_METRIC_COLS = [
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
    "query_size_paths",
    "query_size_ANDs",
    "query_size_NOTs",
    "query_size_added_ORs",
    "query_size_synonym_ORs",
    "logical_operators",
    "all_ORs",
    "pubmed_f1",
    "pubmed_f3",
    "pubmed_f50",
]

GROUP_COLS = ["dataset", "num_positive_bucket", "source_file", "selection_betas"]


def aggregate_results(df):
    """
    Group by (dataset, num_positive_bucket, source_file, selection_betas)
    and compute mean + stddev for each metric column.
    """
    metric_cols = [c for c in AGGREGATE_METRIC_COLS if c in df.columns]

    # not necessary: Fill NaN with 0 for metric columns (match original behavior)
    # df = df.copy()
    # for col in metric_cols:
    #     df[col] = df[col].fillna(0)

    # Add count to first metric to track how many samples were aggregated
    agg_dict = {col: ["mean", "std"] for col in metric_cols}
    if metric_cols:
        agg_dict[metric_cols[0]].append("count")
    
    agg_df = df.groupby(GROUP_COLS).agg(agg_dict)
    
    # Flatten column names: mean -> col, std -> col_stddev, count -> num_samples
    new_cols = []
    count_col_name = metric_cols[0] if metric_cols else None
    for col, stat in agg_df.columns:
        if col == count_col_name:
            if stat == "mean":
                new_cols.append(col)
            elif stat == "std":
                new_cols.append(f"{col}_stddev")
            elif stat == "count":
                new_cols.append("num_samples")
        else:
            if stat == "mean":
                new_cols.append(col)
            else:
                new_cols.append(f"{col}_stddev")
    agg_df.columns = new_cols
    print(agg_df[["num_samples"] + new_cols])

    # Fill NaN stddev (single-row groups) with 0
    stddev_cols = [c for c in agg_df.columns if c.endswith("_stddev")]
    agg_df[stddev_cols] = agg_df[stddev_cols].fillna(0)

    agg_df = agg_df.reset_index()
    return agg_df


def generate_typst_table(
    df,
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
    baseline_names=None,
    top_k_types=None,
    baseline_name="Baselines",
    show_baselines_first=True,
):
    """
    Generates a Typst table from an aggregated DataFrame and a baseline dictionary.

    Args:
        df (pd.DataFrame): Aggregated DataFrame with mean and stddev columns.
        typst_file (str): Path where the Typst table will be written.
        baseline_dict (dict): Dictionary of baselines for each dataset.
                              Format: {dataset: [(name, p, f1, f3, r, ops), ...]}
        baseline_names (list[str] | None): Optional list of baseline names to show,
                                           in display order.
        top_k_types (list[str] | None): List of top_k_types to display (e.g., ["cosine", "#pos", "fixed"]).
                                        Defaults to all types if None.
        metrics (dict | None): Ordered mapping of display name to metric config.
                                Each config supports:
                                  - key (str): DataFrame column name
                                  - direction ("max" | "min"): for best-value bolding
                                  - fmt (str): format string for values
                                  - baseline_index (int | None): index in baseline tuple
                                  - vline_before (bool, optional): add vline before column
    """
    if top_k_types is None:
        top_k_types = ["cosine", "pos_count", "fixed"]

    typst_file = os.path.join(typst_file, f"{table_name}.typ")

    # Build nested dict: stats[dataset][bucket] = [row_dict, ...]
    stats = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        dataset = row_dict["dataset"]
        bucket = row_dict["num_positive_bucket"]

        # If betas is set, skip rows that don't contain any of the selected betas
        if betas is not None:
            row_betas = set(row_dict["selection_betas"].split(","))
            if not row_betas & betas:
                continue  # skip this row entirely

        stats.setdefault(dataset, {}).setdefault(bucket, []).append(row_dict)

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

    def baseline_name_of(baseline):
        if isinstance(baseline, dict):
            return baseline.get("name", "")
        return baseline[0] if baseline else ""

    def select_baselines(baselines):
        if baseline_names is None:
            return list(baselines)
        baseline_by_name = {baseline_name_of(b): b for b in baselines}
        return [
            baseline_by_name[name]
            for name in baseline_names
            if name in baseline_by_name
        ]

    def config_name(row, betas):
        parts = row["selection_betas"].split(",")
        source_file = row["source_file"]
        ktype = get_ktype(source_file=source_file)
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
        for i, dataset in enumerate(used_datasets):  # enumerate(stats.items())]
            buckets = stats.get(dataset, {})
            selected_baselines = select_baselines(baseline_dict.get(dataset, []))
            baseline_count = len(selected_baselines)

            # Pre-count displayed rows to calculate correct total_rows
            displayed_baseline_count = 0
            if baseline_count > 0:
                for baseline in selected_baselines:
                    # Check if baseline has any non-None values for filtered metrics
                    has_value = False
                    for name, cfg in metrics.items():
                        value = None

                        # Handle new dict format: {"name": "...", "Metric": [mean, stddev], ...}
                        if isinstance(baseline, dict):
                            if name in baseline:
                                metric_data = baseline[name]
                                value = (
                                    metric_data[0]
                                    if isinstance(metric_data, (list, tuple))
                                    and len(metric_data) > 0
                                    else metric_data
                                )
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
                        # Check if row's ktype is in allowed top_k_types
                        ktype = get_ktype(row["source_file"])
                        if ktype not in top_k_types:
                            continue
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
                    for baseline in selected_baselines:
                        value = None

                        # Handle new dict format
                        if isinstance(baseline, dict):
                            if name in baseline:
                                metric_data = baseline[name]
                                value = (
                                    metric_data[0]
                                    if isinstance(metric_data, (list, tuple))
                                    and len(metric_data) > 0
                                    else metric_data
                                )
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

            # Prepare baseline rows
            baseline_lines = []
            if baseline_count > 0:
                # Count baselines that will actually be displayed
                displayed_baseline_count = 0
                for baseline in selected_baselines:
                    metric_cells = []
                    for m_name, cfg in metrics.items():
                        value = None

                        # Handle new dict format: {"name": "...", "Metric": [mean, stddev], ...}
                        if isinstance(baseline, dict):
                            if m_name in baseline:
                                metric_data = baseline[m_name]
                                # Extract mean (first value)
                                value = (
                                    metric_data[0]
                                    if isinstance(metric_data, (list, tuple))
                                    and len(metric_data) > 0
                                    else metric_data
                                )
                        # Handle old tuple format: (name, precision, f1, f3, recall, ...)
                        else:
                            idx = cfg.get("baseline_index")
                            if idx is not None and len(baseline) > idx:
                                value = baseline[idx]

                        metric_cells.append(f"[{fmt(value, m_name)}]")
                    if not all(cell == "[]" for cell in metric_cells):
                        displayed_baseline_count += 1

                if displayed_baseline_count > 0:
                    baseline_lines.append(
                        f"  table.cell(rowspan: {displayed_baseline_count}, rotate(-90deg, reflow:true)[{baseline_name}]),\n"
                    )
                    for baseline in selected_baselines:
                        # Get baseline name
                        name = baseline_name_of(baseline)

                        metric_cells = []
                        for m_name, cfg in metrics.items():
                            value = None
                            stddev = None

                            # Handle new dict format
                            if isinstance(baseline, dict):
                                if m_name in baseline:
                                    metric_data = baseline[m_name]
                                    if (
                                        isinstance(metric_data, (list, tuple))
                                        and len(metric_data) >= 2
                                    ):
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

                        baseline_lines.append(f"    [{name}], {', '.join(metric_cells)},\n")

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
                        # Filter by top_k_types
                        ktype = get_ktype(row["source_file"])
                        if ktype not in top_k_types:
                            continue
                        algo_name, _ = config_name(row, betas)
                        algo_name_with_bucket = f"{algo_name}"
                        if len(min_positive_buckets) > 1:
                            algo_name_with_bucket += f"{bucket_name}"
                        if algo_name_with_bucket not in all_rows_by_algo:
                            all_rows_by_algo[algo_name_with_bucket] = []
                        all_rows_by_algo[algo_name_with_bucket].append(
                            (row, ktype, bucket_name)
                        )

            # Collect config rows
            config_lines = []
            # Render rows grouped by algo-name-Fscore (each spans 3)
            for algo_name in sorted(
                all_rows_by_algo.keys(),
                reverse=True,
                key=lambda x: int(x.split("-F")[-1].split("\\")[0]),
            ):
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
                    config_lines.append(
                        f"  table.cell(rowspan: {num_displayed}, rotate(-90deg, reflow:true)[{algo_name if len(top_k_types) > 1 else ''}]),\n"
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
                        pretty_ktype = f"#{ktype}_k"
                        method_column = pretty_ktype if len(top_k_types) > 1 else f"{algo_name}\-{pretty_ktype}"
                        config_lines.append(f"    [{method_column}], {', '.join(metric_cells)},\n")

            # Write baselines and configs in the right order
            if show_baselines_first:
                f.writelines(baseline_lines)
                f.writelines(config_lines)
            else:
                f.writelines(config_lines)
                f.writelines(baseline_lines)

        f.write(")]\n")

    print(f"Typst table written to {typst_file}")


if __name__ == "__main__":
    # load baseline dict from file
    with open("data/examples/baseline_values.json", "r") as f:
        baseline_dict = json.load(f)

    folder_path = CURRENT_BEST_RUN_FOLDER
    typst_base = f"../master-thesis-writing/writing/tables/{CURRENT_BEST}/"

    # Load and prepare DataFrame once
    base_df = get_qg_results(folder_path, min_positive_threshold=None, query_ids=None)
    base_df = calc_missing_columns_in_result_df(base_df)
    
    outliers_0 = len(base_df[["dataset", "num_positive_bucket", "pubmed_retrieved"]]
        [
            (base_df["pubmed_retrieved"] == 0)
        ]
    )
    print("warning, found", outliers_0, "rows with 0 pubmed_retrieved")
    outliers_200k = len(base_df[["dataset", "num_positive_bucket", "pubmed_retrieved"]]
        [
            (base_df["pubmed_retrieved"] > 200_000)
        ]
    )
    print("warning, found", outliers_200k, "rows with >200k pubmed_retrieved")
    agg_df = aggregate_results(base_df)
    
    def expected_num_samples(row, dataset_details):
        dataset = row["dataset"]
        bucket = row["num_positive_bucket"]
        expected = 0
        for q_id in dataset_details.keys():
            if q_id in TRAIN_REVIEWS:
                continue
            ds, _, _ = review_id_to_dataset(q_id)
            if ds == "tar2017":
                ds = "tar2018"
            if not ds == dataset:
                continue
            num_positive = len(dataset_details[q_id]["positives"])
            if bucket == "\<50" and num_positive < 50:
                expected+=1
            elif bucket == "\>\=50" and num_positive >= 50:
                expected+=1
        return expected*10

    ## add column expected_num_samples to agg_df based on dataset_details
    dataset_details = get_dataset_details()
    agg_df["expected_num_samples"] = agg_df.apply(lambda row: expected_num_samples(row, dataset_details), axis=1)
    print(agg_df[["dataset", "num_positive_bucket", "num_samples", "expected_num_samples"]][agg_df["num_samples"] != agg_df["expected_num_samples"]])
    generate_typst_table(
        df=agg_df,
        typst_file=typst_base,
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=True,
        show_operators=False,
        baseline_names=[
            "Manual",
            "Conceptual",
            "Objective",
            "Semantic",
            "ChatGPT",
            "Fine-Tuned LLM",
            "Fine-Tuned LLM (self-evaluation)",
        ],
        top_k_types=["cosine", "pos_count", "fixed"],
        table_name="best_table",
    )
    generate_typst_table(
        df=agg_df,
        typst_file=typst_base,
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50", "\<50"],
        used_datasets=["tar2018", "tar2019", "sigir2017", "sr_updates"],
        show_performance=True,
        show_operators=True,
        baseline_names=[
            "Manual",
            "Conceptual",
            "Objective",
            "Semantic",
            "ChatGPT",
            "Fine-Tuned LLM",
            "Fine-Tuned LLM (self-evaluation)",
        ],
        top_k_types=["cosine", "pos_count", "fixed"],
        table_name="best_table_appendix",
    )
    generate_typst_table(
        df=agg_df,
        typst_file=typst_base,
        baseline_dict=baseline_dict,
        betas={"3", "15", "30", "50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=False,
        show_operators=True,
        baseline_names=[
            "Manual",
            "Conceptual",
            "Objective",
            "Semantic",
            "ChatGPT",
            "Fine-Tuned LLM",
            "Fine-Tuned LLM (self-evaluation)",
        ],
        top_k_types=["cosine", "pos_count", "fixed"],
        table_name="best_table_operators",
    )
    generate_typst_table(
        df=agg_df,
        typst_file=typst_base,
        baseline_dict=baseline_dict,
        betas={"50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=True,
        show_operators=False,
        baseline_names=["#no_Ors", "#no_variation", "#no_rule_pruning", "#no_atm_not_countered", "#no_atm"],
        top_k_types=["cosine"],
        table_name="base_variations_table",
        baseline_name="Variations",
        show_baselines_first=False,
    )
    generate_typst_table(
        df=agg_df,
        typst_file=typst_base,
        baseline_dict=baseline_dict,
        betas={"50"},
        metrics=RESULT_TABLE_PERFORMANCE_METRICS | RESULT_TABLE_OPERATOR_METRICS,
        text_size=10.3,
        min_positive_buckets=["\>\=50"],
        used_datasets=["tar2018"],
        show_performance=False,
        show_operators=True,
        baseline_names=["#no_Ors", "#no_variation", "#no_rule_pruning", "#no_atm_not_countered", "#no_atm"],
        top_k_types=["cosine"],
        table_name="base_variations_table_operators",
        baseline_name="Variations",
        show_baselines_first=False,
    )

    print("done")
