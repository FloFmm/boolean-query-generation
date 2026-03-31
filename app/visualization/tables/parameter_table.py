import json
import os
from pathlib import Path

if __name__ == "__main__":
    prameter_kes = [
        # dense retrieval
        "top_k",
        "dont_cares",
        "rank_weight",
        
        # Learning Tree Ensembles 
        "max_depth",
        "class_weight",
        "min_weight_fraction_leaf",
        "top_k_or_candidates",
        "max_features",
        "randomize_max_feature",
        "min_impurity_decrease_range_start",
        "min_impurity_decrease_range_end",
        "randomize_min_impurity_decrease_range",
        
        # Rule Extraction and Query Generation
        "mh_noexp",
        "tiab",
        "term_expansions",
        "cover_beta",
        "pruning_beta",
        "min_rule_occ",
        "min_tree_occ",
    ]

    descriptions = {
        "max_depth": "Maximum depth of the trees, and therefore the maximum length of a rule",
        "min_weight_fraction_leaf": "Early stopping criterion for tree learning (required fraction of the total sum of document weights)",
        "top_k": "Multiplier for the top $k$ documents considered as pseudo-relevant",
        "dont_cares": "Multiplier for the number of intermediate ranked documents considered as don't cares",
        "rank_weight": "Weight of the document ranked first compared to the last pseudo-relevant document",
        "max_features": "Fraction of keywords to consider when looking for the best split",
        "min_impurity_decrease_range_start": "Minimum impurity decrease required for adding a #OR\-connected keyword to the root node",
        "min_impurity_decrease_range_end": "Minimum impurity decrease required for adding a #OR\-connected keyword to a leaf node",
        "class_weight": "Class balancing ranging from equivalent weight for both classes to full inverse frequency balancing",
        "top_k_or_candidates": "Heuristic number of top keywords to consider for #OR\-connected addition to a node",
        "randomize_max_feature": """
            Biased randomization of $#max_features\(t)$ in [#max_features, 1.0] for each tree $t$. \ 
            #set align(left)
            $#randomize_max_feature>1$: biased towards #max_features\ 
            $#randomize_max_feature<1$: biased towards 1.0
            """,
        "randomize_min_impurity_decrease_range": """
            Biased randomization of $#min_impurity_decrease _x\(t)$ in [$#min_impurity_decrease _x$, 1.0] ($x in {"root","leaf"}$) for each tree $t$. \ 
            #set align(left)
            $#randomize_max_feature>1$: biased towards $#min_impurity_decrease _x$\ 
            $#randomize_max_feature<1$: biased towards 1.0
        """,
        "min_tree_occ": "Minimum fraction of trees a keyword must appear in",
        "min_rule_occ": "Minimum fraction of rules a keyword must appear in",
        "cover_beta": "Part of optimization target\ #opt_target_cover during rule subset selection",
        "pruning_beta": "Part of optimization target #opt_target_pruning during rule pruning",
        "term_expansions": "Whether to explode keywords using our generated synonym set",
        "mh_noexp": "Whether to use no-explode for MeSH terms",
        "tiab": "Whether to deactivate #gls(\"atm\") and restrict search to Title/Abstract"
    }

    ranges = { #checked correct
        "max_depth": "[3, 10]",
        "min_weight_fraction_leaf": "[0.0, 0.002]",
        "top_k": "[0.1, 2.0]",
        "dont_cares": "[0.0, 5.0]",
        "rank_weight": "[1.0, 10.0]",
        "max_features": "[0.01, 1.0]",
        "min_impurity_decrease_range_start": "[0.001, 0.05]",
        "min_impurity_decrease_range_end": "[0.001, 0.05]",
        "class_weight": "[0.0, 1.0]",
        "top_k_or_candidates": "[500, 3000]",
        "randomize_max_feature": "[0.0, 3.0]",
        "randomize_min_impurity_decrease_range": "[0.0, 3.0]",
        "min_tree_occ": "[0.0, 0.2]",
        "min_rule_occ": "[0.0, 0.1]",
        "cover_beta": "[0.1, 2.0]",
        "pruning_beta": "[0.05, 1.0]",
        "term_expansions": "{#true, #false}",
        "mh_noexp": "{#true, #false}",
        "tiab": "{#true, #false}"
    }

    groups = [
        {
            "label": "#link(<chap:dense_retrieval>)[Dense Retrieval]",
            "keys": ["top_k", "dont_cares", "rank_weight"],
        },
        {
            "label": "#link(<chap:learning_ensemble>)[Learning Tree Ensembles]",
            "keys": [
                "max_depth",
                "class_weight",
                "min_weight_fraction_leaf",
                "top_k_or_candidates",
                "max_features",
                "randomize_max_feature",
                "min_impurity_decrease_range_start",
                "min_impurity_decrease_range_end",
                "randomize_min_impurity_decrease_range",
            ],
        },
        {
            "label": "#link(<chap:rule_extraction>)[Rule Extraction and Boolean Query Generation]",
            "keys": [
                "mh_noexp",
                "tiab",
                "term_expansions",
                "cover_beta",
                "pruning_beta",
                "min_rule_occ",
                "min_tree_occ",
            ],
        },
    ]

    # Load best params
    with open("data/final/best_params_te=False.json", "r") as f:
        data = json.load(f)

    # Extract best params for F3, F15, F30, F50
    best_params = {}
    for entry in data:
        for beta in entry.get("betas", {}):
            best_params[str(beta)] = entry["params"]

    target_betas = ["3", "15", "30", "50"]

    typst_lines = []
    typst_lines.append('#import "../../thesis/assets/assets.typ": *')
    typst_lines.append("#let parameter_table() = [")
    typst_lines.append("#table(")
    typst_lines.append("  columns: (auto, 3fr, 5fr, 2fr, 1fr, 1fr, 1fr, 1fr),")
    typst_lines.append("table.hline(stroke: table_strong_line),\n")
    typst_lines.append("  table.header([], [Parameter], [Description], [Range], [$F_3$], [$F_(15)$], [$F_(30)$], [$F_(50)$]),")
    typst_lines.append("table.hline(stroke: table_strong_line),\n")
    for group in groups:
        label = group["label"]
        keys = group["keys"]
        n = len(keys)

        for i, key in enumerate(keys):
            desc = descriptions.get(key, "")
            rng = ranges.get(key, "")

            vals = []
            for beta in target_betas:
                val = best_params.get(beta, {}).get(key, "")
                if isinstance(val, float):
                    val_str = f"{val:.4g}"
                elif isinstance(val, bool):
                    if val:
                        val_str = "#true"
                    else:
                        val_str = "#false"
                else:
                    val_str = str(val)
                vals.append(val_str)

            if i == 0:
                typst_lines.append(f"  table.cell(rowspan: {n})[#rotate(-90deg, reflow: true)[{label}]],")

            typst_lines.append(f"  [#{key}_long\ #{key}], [{desc}], [{rng}], [{vals[0]}], [{vals[1]}], [{vals[2]}], [{vals[3]}],")

    typst_lines.append("table.hline(stroke: table_strong_line),\n")
    typst_lines.append(")")
    typst_lines.append("]")

    output_file = "../master-thesis-writing/writing/tables/parameter_table/parameter_table.typ"
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Table generated at {output_file}")
