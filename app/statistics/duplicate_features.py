# function that takes a rules [[[1,2,3, True], [4,5,6, False]], [[7,8,9, True], [10,11,12, False]]] and returns how many percent of the
# features are duplicates. for that ids are first mapped to strings useing feature_names map
# after that it is checked for each feature_name in the entire rule list whether it occurs in another feature_name of the rule set (or whetehr another feature appears as substring in it)
# this is dviided by number of features in the rule set (counting duplciates)
# rules consist of a list of rules, where each rule is a list of terms, and each term is a list of [feature_id, is_positive]
from app.config.config import BOW_PARAMS, CURRENT_BEST_RUN_FOLDER
from app.dataset.utils import get_qg_results, load_vectors


def calculate_duplicate_features_percentage(rules, feature_names, exact_match=False):
    # Map feature IDs to names
    feature_occurrences = set()
    duplicates = 0
    total_features = 0
    for rule in rules:
        for term in rule:
            for feature_id in term[0]:
                feature_name = feature_names[feature_id]
                total_features += 1

                if exact_match:
                    if feature_name in feature_occurrences:
                        duplicates += 1
                else:
                    if any(
                        feature_name in existing for existing in feature_occurrences
                    ) or any(
                        existing in feature_name for existing in feature_occurrences
                    ):
                        duplicates += 1
                feature_occurrences.add(feature_name)

    duplicate_percentage = duplicates / total_features * 100
    return duplicate_percentage


if __name__ == "__main__":
    dataframe = get_qg_results(CURRENT_BEST_RUN_FOLDER, min_positive_threshold=50, top_k_types=["cosine"], restrict_betas=["50"])
    print(len(dataframe), "queries in dataframe")
    X, ordered_pmids, feature_names = load_vectors(**BOW_PARAMS)
    dataframe["duplicate_percentage_exact"] = dataframe["rules"].apply(
        lambda rules: calculate_duplicate_features_percentage(
            rules, feature_names, exact_match=True
        )
    )
    print(
        "Average duplicate percentage (exact match):",
        dataframe["duplicate_percentage_exact"].mean(),
    )

    dataframe["duplicate_percentage_exact"] = dataframe["rules"].apply(
        lambda rules: calculate_duplicate_features_percentage(
            rules, feature_names, exact_match=False
        )
    )
    print(
        "Average duplicate percentage (substring match):",
        dataframe["duplicate_percentage_exact"].mean(),
    )