#!/bin/bash
set -e

python -m app.visualization.document_weight_graph
python -m app.visualization.precision_recall_heatmap
python -m app.visualization.top_k_type_comparison
python -m app.visualization.recall_curve_by_bucket
python -m app.visualization.compare_retrievers
python -m app.visualization.size_impact
# python -m app.visualization.f_beta
python -m app.visualization.visualize_optuna



# tables
python -m app.visualization.feature_replacement_map
python -m app.statistics.baseline_values

python -m app.visualization.tables.query_example_table
python -m app.visualization.tables.result_table
python -m app.visualization.tables.handmade_table
python -m app.visualization.tables.best_worst_table
python -m app.visualization.tables.parameter_table
python -m app.visualization.rules_vs_length

