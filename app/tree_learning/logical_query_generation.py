from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.metrics import recall_score
from sklearn import tree
import matplotlib.pyplot as plt#plt the figure, setting a black background
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])



def train_text_classifier(
    set1,
    set2,
    max_depth=5,
    random_state=42,
    # min_samples_split=5,
    # min_samples_leaf=5,
    class_weight="balanced",
):
    # class_weight = {"set1": int(len(set2)/len(set1)*4.0), "set2": 1}
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
    vectorizer = CountVectorizer(binary=True)#, stop_words="english")
    X = vectorizer.fit_transform(texts)

    # Train decision tree
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        # min_samples_split=min_samples_split,
        class_weight=class_weight,
        # min_samples_leaf=min_samples_leaf,
    )
    clf.fit(X, labels)

    # Accuracy on the same data
    y_pred = clf.predict(X)
    accuracy = clf.score(X, labels)
    recall_set1 = recall_score(labels, y_pred, pos_label="set1")

    # Human-readable rules
    feature_names = vectorizer.get_feature_names_out()
    decision_tree = export_text(clf, feature_names=list(feature_names))

    return {
        "recall": recall_set1,
        "accuracy": accuracy,
        "decision_tree": decision_tree,
        "boolean_function_set1": tree_to_dnf_pubmed(clf, feature_names, "set1"),
        "boolean_function_set2": tree_to_dnf_pubmed(clf, feature_names, "set2"),
        "model": clf,
        "vectorizer": vectorizer,
        "feature_names": feature_names,
        "class_names": ["set1", "set2"],
    }
    # from nltk.corpus import wordnet as wn

    # synonym_map = {}

    # # Example: map 'automobile' and 'car' → 'car'
    # for syn in wn.synsets('car'):
    #     for lemma in syn.lemmas():
    #         synonym_map[lemma.name()] = 'car'

    # # Then replace in your text
    # def map_synonyms(text, synonym_map):
    #     return " ".join([synonym_map.get(word, word) for word in text.split()])


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


def plot_tree(model, feature_names, class_names):
    plt.figure(figsize=(30,10), facecolor ='k')#create the tree plot
    a = tree.plot_tree(model,
                    #use the feature names stored
                    feature_names = feature_names,
                    #use the class names stored
                    class_names = class_names,
                    rounded = True,
                    filled = True,
                    fontsize=14)#show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    set1 = [
        "biofeedback Apples and bananas are tasty.",
        "I love eating fruit in the morning. Biofeedback",
        "I love eating fruit in the morning. Biofeedback",
        "I love eating fruit in the morning. Biofeedback",
        "I love eating fruit in the morning. Biofeedback",
        "I love eating fruit in the morning. Biofeedback",
        "I love eating fruit in the morning. Biofeedback",
    ]
    set2 = [
        "My car needs a new engine.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
        "Wheels and tires are important for driving.",
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
