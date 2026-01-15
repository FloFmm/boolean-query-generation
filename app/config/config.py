TOP_K = {
    0.7: (
        [1.0, 2.5, 5.0, 8.5, 13.0, 18.0, 25.5, 40.5, 63.0, 88.0, 125.5, 200.5, 375.5, 625.5],
        [4.6, 15, 33, 55, 67, 72, 102, 152, 217, 417, 872, 493, 1585, 2700]
        )
    }

BOW_PARAMS = {
    "lower_case": True,
    "mesh_ancestors": True,
    "rm_numbers": True,
    "rm_punct": True,
    "related_words": True,
}

RF_PARAMS = {
    "top_k": {"recall": 0.7, "factor": 1.5},#200, # 0.7 means k where we reach 0.7 recall multipled by factor
    "rank_weight": 1.5, # how much more weighted shall rank 1 be than rank k
    "n_estimators": 32,
    "max_depth": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.00005,
    "max_features": 0.1,#"sqrt",
    "randomize_max_feature": 1,
    "min_impurity_decrease_range": (0.01, 0.01),
    "randomize_min_impurity_decrease_range": 1,
    "bootstrap": True,
    "n_jobs": 32,
    "random_state": None,
    "verbose": True,
    "class_weight": 0.2,
    "max_samples": None,
    "top_k_or_candidates": 500,
    "prefer_pos_splits": 1.1,
    "max_or_features": 10,
}

QG_PARAMS = {
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.02,
    "cost_factor": 0.002, # 50 ANDs are worth 0.1 F3 score
    "min_rule_precision": 0.01,
    "cover_beta": 2.0, # high to prefer covering training data fully
    "pruning_beta": 0.1, # low to prefer precise rules
    "synonym_expansion": True,
    "mh_noexp": True,
    "tiab": True,
    "pruning_thresholds": {
        "or": {
            False: {  # removal in negated term false -> is positive term
                "acceptance_metric": "tp_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
            True: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
        },
        "and": {
            False: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
            True: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
        },
    }
}