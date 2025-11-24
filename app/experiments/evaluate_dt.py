
import sys
from tree_learning.disjunctive_dt import GreedyORDecisionTree
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "../systematic-review-datasets")))
from csmed.csmed.csmed_cochrane import CSMeDCochrane


if __name__ == "__main__":
    dataset = CSMeDCochrane().load_dataset(
        base_path="../systematic-review-datasets/csmed/csmed_cochrane"
    )


    models = []
    for min_samples_split in [2, 5]:
        for min_impurity_d_start in [0.01, 0.1, 0.001]:
            for min_impurity_d_end in [0.03, 0.3, 0.003]:
                for top_k_or_candidates in [100, 500, 1000]:
                    for class_weight in ["balanced", {1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 4, 0: 1}, {1: 5, 0: 1}, {1: 6, 0: 1}, {1: 3, 0: 0.5}, {1: 500, 0: 0.5}, {1: 500, 0: 1}]:
                        models.append(
                            GreedyORDecisionTree(
                                max_depth=4,
                                min_samples_split=min_samples_split,
                                min_impurity_decrease_range=[
                                    min_impurity_d_start,
                                    min_impurity_d_end,
                                ],
                                top_k_or_candidates=top_k_or_candidates,
                                class_weight=class_weight,  # "balanced",
                                verbose=True,
                            )
                        )

    args = {
        "models": models,
        "baseline_folder": "./data/pubmed/baseline",
        "output_path": "data/pubmed/statistics/classifier_learning/",
        "skip_existing": True,
        "n_docs": 5_000_0,
        "min_f_occ": {0: 10, 1: 2},
        "mesh_terms": [
            "Endometriosis",
            "Rectal Neoplasms",
            "Fluorodeoxyglucose F18",
            "Cholelithiasis",
            "Antigens, Helminth",
            "Down Syndrome",
            "Antigens, Protozoan",
            "Urinary Tract Infections",
            "Chromosome Aberrations",
            "Streptococcal Infections",
            "Kidney Transplantation",
            "Cognition Disorders",
            "Alzheimer Disease",
            "Pregnancy",
        ],
    }
    # train_all_mesh_terms_jsonl(**args)