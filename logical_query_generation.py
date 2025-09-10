from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
import numpy as np


def train_text_classifier(set1, set2, max_depth=4, random_state=42):
    """
    Train a decision tree classifier to distinguish between two sets of texts,
    and evaluate on the same dataset (no train/test split).

    Args:
        set1 (list[str]): First set of texts.
        set2 (list[str]): Second set of texts.
        max_depth (int): Maximum depth of the decision tree.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: {
            "accuracy": float,
            "rules": str,
            "model": DecisionTreeClassifier,
            "vectorizer": CountVectorizer
        }
    """
    # Combine texts and create labels
    texts = set1 + set2
    labels = ["set1"] * len(set1) + ["set2"] * len(set2)

    # Vectorizer: binary presence, remove stopwords
    vectorizer = CountVectorizer(binary=True, stop_words="english")
    X = vectorizer.fit_transform(texts)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, labels)

    # Accuracy on the same data
    accuracy = clf.score(X, labels)

    # Human-readable rules
    feature_names = vectorizer.get_feature_names_out()
    decision_tree = export_text(clf, feature_names=list(feature_names))

    return {
        "accuracy": accuracy,
        "decision_tree": decision_tree,
        "boolean_function_set1": tree_to_dnf_pubmed(clf, feature_names, "set1"),
        "boolean_function_set2": tree_to_dnf_pubmed(clf, feature_names, "set2"),
        "model": clf,
        "vectorizer": vectorizer,
    }


def tree_to_boolean(clf, feature_names, target_class):
    tree = clf.tree_
    classes = clf.classes_

    def recurse(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            left_expr = recurse(tree.children_left[node])
            right_expr = recurse(tree.children_right[node])

            exprs = []

            if left_expr is not None:
                if left_expr == "":
                    exprs.append(f'NOT "{name}"')
                else:
                    exprs.append(f'(NOT "{name}" AND {left_expr})')

            if right_expr is not None:
                if right_expr == "":
                    exprs.append(f'"{name}"')
                else:
                    exprs.append(f'("{name}" AND {right_expr})')

            if not exprs:
                return None
            return " OR ".join(exprs)
        else:
            # Leaf node: only include path if leaf predicts target_class
            class_idx = tree.value[node].argmax()
            if classes[class_idx] == target_class:
                return ""  # empty string, meaning "this path is enough"
            else:
                return None

    return recurse(0)


def tree_to_dnf_pubmed(
    tree: DecisionTreeClassifier, feature_names=None, target_class=None
):
    """
    Converts a boolean DecisionTreeClassifier to DNF form.
    Assumes:
      - Left child = NOT of the feature
      - Right child = feature is present
    Returns:
      DNF as a str
    """
    tree_ = tree.tree_
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(tree_.n_features)]

    def recurse(node):
        if tree_.feature[node] == -2:  # leaf
            predicted_class = tree.classes_[np.argmax(tree_.value[node][0])]
            if predicted_class == target_class:
                return [[]]  # path leads to target_class
            else:
                return []  # path does not lead to target_class
        feature = feature_names[tree_.feature[node]]
        left = recurse(tree_.children_left[node])
        right = recurse(tree_.children_right[node])

        # Left child = NOT feature
        left = [[f"NOT {feature}"] + terms for terms in left]
        # Right child = feature present
        right = [[f"{feature}"] + terms for terms in right]

        # Combine left and right disjunctions
        return left + right

    dnf_clauses = recurse(0)
    return " OR ".join(
        [
            f"({' AND '.join([l for l in term if 'NOT' not in l])}{' ' if [l for l in term if 'NOT' not in l] and [l for l in term if 'NOT' in l] else ''}{' '.join([l for l in term if 'NOT' in l])})"
            for term in dnf_clauses
        ]
    )
    # return dnf_clauses


# Example usage
if __name__ == "__main__":
    set1 = [
        "Apples and bananas are tasty.",
        "I love eating fruit in the morning.",
    ]
    set2 = [
        "My car needs a new engine.",
        "Wheels and tires are important for driving.",
    ]

    result = train_text_classifier(set1, set2)
    print("Accuracy:", result["accuracy"])
    print()
    print(result["decision_tree"])
    print()
    print(result["boolean_function_set1"])
    print()
    print(result["boolean_function_set2"])
