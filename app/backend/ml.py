# backend/ml.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class TextDecisionTree:
    def __init__(self, max_depth=4, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.vectorizer = None

    def train(self, set1, set2):
        texts = set1 + set2
        labels = ["set1"] * len(set1) + ["set2"] * len(set2)

        self.vectorizer = CountVectorizer(binary=True, stop_words="english")
        X = self.vectorizer.fit_transform(texts)

        clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        clf.fit(X, labels)

        self.model = clf

        return {
            "accuracy": clf.score(X, labels),
            "decision_tree": self.get_tree_text(),
            "boolean_function_set1": self.get_formula("set1"),
            "boolean_function_set2": self.get_formula("set2"),
        }

    def get_tree_text(self):
        from sklearn.tree import export_text
        return export_text(self.model, feature_names=list(self.vectorizer.get_feature_names_out()))

    def get_formula(self, target_class):
        return self._tree_to_dnf(self.model, list(self.vectorizer.get_feature_names_out()), target_class)

    def _tree_to_dnf(self, tree, feature_names, target_class):
        tree_ = tree.tree_

        def recurse(node):
            if tree_.feature[node] == -2:
                predicted_class = tree.classes_[np.argmax(tree_.value[node][0])]
                if predicted_class == target_class:
                    return [[]]  # path leads to target_class
                else:
                    return []  # path does not lead to target_class
            feature = feature_names[tree_.feature[node]]
            left = recurse(tree_.children_left[node])
            right = recurse(tree_.children_right[node])
            left = [[f"NOT {feature}"] + terms for terms in left]
            right = [[f"{feature}"] + terms for terms in right]
            return left + right

        dnf_clauses = recurse(0)
        return " OR ".join(
            [
                f"({' AND '.join([l for l in term if 'NOT' not in l])}{' ' if [l for l in term if 'NOT' not in l] and [l for l in term if 'NOT' in l] else ''}{' '.join([l for l in term if 'NOT' in l])})"
                for term in dnf_clauses
            ]
        )

# Singleton instance to reuse in FastAPI
text_tree_model = TextDecisionTree()
