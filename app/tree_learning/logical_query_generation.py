from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import export_text, _tree
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn import tree
import matplotlib.pyplot as plt  # plt the figure, setting a black background
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from imodels import SkopeRulesClassifier, DecisionTreeClassifier, Rule
import re
from app.tree_learning.disjunctive_dt import GreedyORDecisionTree

nlp = spacy.load("en_core_web_lg")


def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


def build_semantic_map(words, similarity_threshold=0.7):
    """
    Group words by semantic similarity and assign each cluster a canonical lemmatized word.
    """
    # Get SpaCy docs and vectors
    # docs = [nlp(word) for word in tqdm(words, desc="Processing words with SpaCy")]
    docs = list(nlp.pipe(tqdm(words, desc="Processing words with SpaCy", unit="word")))
    vectors = np.array([doc.vector for doc in docs])

    # Compute cosine distance matrix
    tqdm.write("Computing cosine distance matrix...")
    distance_matrix = cosine_distances(vectors)

    # Cluster words using Agglomerative Clustering
    # distance_threshold controls similarity; 0 = identical, higher = more lenient
    tqdm.write("Clustering words...")
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let distance_threshold decide
        metric="precomputed",  # was affinity='precomputed'
        linkage="complete",
        distance_threshold=1 - similarity_threshold,
    )
    labels = clustering.fit_predict(distance_matrix)

    # Map each word to the canonical lemma of its cluster
    tqdm.write("Building semantic map...")
    semantic_map = {}
    for label in set(labels):
        cluster_words = [words[i] for i in range(len(words)) if labels[i] == label]
        # Pick the most common lemma as canonical (or just first one)
        canonical = nlp(cluster_words[0])[0].lemma_
        for w in cluster_words:
            semantic_map[w] = canonical

    return semantic_map


def map_synonyms(text, synonym_map):
    """
    Replace words in the text with their canonical synonym.
    """
    return " ".join([synonym_map.get(word, word) for word in text.split()])


def build_synonym_map(words):
    """
    Build a mapping of words to their canonical synonym using WordNet.
    """
    synonym_map = {}
    for word in words:
        synsets = wn.synsets(word)
        if synsets:
            # Pick the first lemma of the first synset as canonical
            canonical = synsets[0].lemmas()[0].name()
            synonym_map[word] = canonical
    return synonym_map


def train_text_classifier(
    clf,
    set1,
    set2,
    n_words
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
    labels = [1] * len(set1) + [0] * len(set2)

    # Vectorizer: binary presence, remove stopwords
    vectorizer = CountVectorizer(binary=True, max_features=n_words, stop_words="english")
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    #debug
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    least_common_count = np.min(word_counts)

    print(f"Number of features (words): {len(feature_names)}")
    print(f"Least common word count: {least_common_count}")
    #debug

    if isinstance(clf, SkopeRulesClassifier):
        X = X.toarray()
    if isinstance(clf, GreedyORDecisionTree):
        labels = np.array(labels)
    # Train classifier
    clf.fit(X, labels)

    # Accuracy on the same data
    if isinstance(clf, SkopeRulesClassifier):
        clf.rules_ = [
            rule
            for rule in clf.rules_
            if all(symbol != "==" for feature, symbol in sorted(rule.agg_dict.keys()))
        ]
        top_n_rules = min(5, len(clf.rules_))
        y_pred = clf._predict_top_rules(X, top_n_rules)
    else:
        y_pred = clf.predict(X)
    # print(y_pred)
    recall = recall_score(labels, y_pred, pos_label=1)
    precision = precision_score(labels, y_pred, pos_label=1)
    # Human-readable rules

    if isinstance(clf, SkopeRulesClassifier):
        pretty_print = [rule.rule for rule in clf.rules_]
        boolean_function_set1 = " OR ".join(
            [
                "("
                + " AND ".join(
                    [
                        ("NOT " if symbol == "<=" else "")
                        + feature_names[int(feature[1:])]
                        for feature, symbol in sorted(rule.agg_dict.keys())
                    ]
                )
                + ")"
                for rule in clf.rules_[:top_n_rules]
            ]
        )
        # for rule in clf.rules_:
        #     print(rule.rule, rule.args, rule.agg_dict)
        boolean_function_set2 = (
            "boolean_function_set2 not available for SkopeRulesClassifier"
        )
    elif isinstance(clf, GreedyORDecisionTree):
        pretty_print = clf.export_tree(feature_names=list(feature_names))
        boolean_function_set1 = ""
        boolean_function_set2 = ""
    else:
        pretty_print = export_text(clf, feature_names=list(feature_names))
        boolean_function_set1 = tree_to_dnf_pubmed(clf, feature_names, 1)
        boolean_function_set2 = tree_to_dnf_pubmed(clf, feature_names, 0)

    return {
        "recall": recall,
        "precision": precision,
        "pretty_print": pretty_print,
        "boolean_function_set1": boolean_function_set1,
        "boolean_function_set2": boolean_function_set2,
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
    plt.figure(figsize=(30, 10), facecolor="k")  # create the tree plot
    a = tree.plot_tree(
        model,
        # use the feature names stored
        feature_names=feature_names,
        # use the class names stored
        class_names=class_names,
        rounded=True,
        filled=True,
        fontsize=14,
    )  # show the plot
    plt.show()


# Example usage
if __name__ == "__main__":
    # words = ["teeth", "dental", "tooth", "smile", "orthodontic", "drawing", "draw", "painting", "art", "Sketch", "Depict", "Design", "Draft"]
    # semantic_map = build_semantic_map(words, similarity_threshold=0.5)
    # print(semantic_map)

    set1 = [
        "car mobile auto.",
        "mobile auto",
        "auto mobile",
    ]
    set2 = [
        "flora auto",
        "vegetation.",
        "greenery.",
        "seedling. mobile",
        "sapling.",
        "herb.",
        "shrub.",
        "tree.",
        "plant.",
    ]

    result = train_text_classifier(
        # DecisionTreeClassifier(
        #     max_depth=5,
        #     random_state=42,
        #     # min_samples_split=min_samples_split,
        #     class_weight="balanced",
        #     # min_samples_leaf=min_samples_leaf,
        # ),
        SkopeRulesClassifier(
            n_estimators=1000,
            precision_min=0.1,
            recall_min=0.1,
        ),
        set1,
        set2,
    )
    print("Recall:", result["recall"])
    print("Precision:", result["precision"])
    print()
    # Access raw rules with metrics
    print(result["pretty_print"])
    print()
    print(result["boolean_function_set1"])
    print()
    print(result["boolean_function_set2"])
