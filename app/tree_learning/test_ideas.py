import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import random
import itertools

def gini_impurity(y):
    """Compute Gini impurity."""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)


def best_split(X, y, features):
    """Find best split feature (single feature)."""
    if not features:
        # No features left to split on → cannot improve, return None
        return None, (None, None), 0.0
    best_feature = None
    best_impurity = 1.0
    best_split = None
    current_impurity = gini_impurity(y)
    for f in features:
        left_mask = X[:, f] == 1
        right_mask = ~left_mask
        left_imp = gini_impurity(y[left_mask])
        right_imp = gini_impurity(y[right_mask])
        weighted = (
            len(y[left_mask]) * left_imp + len(y[right_mask]) * right_imp
        ) / len(y)

        if weighted < best_impurity:
            best_impurity = weighted
            best_feature = f
            best_split = (left_mask, right_mask)

    improvement = current_impurity - best_impurity
    return best_feature, best_split, improvement


def greedy_or_expand(X, y, base_features, candidate_features):
    """Try adding features with OR to current node."""
    current_mask = np.any(X[:, base_features] == 1, axis=1)
    best_impurity = (
        len(y[current_mask]) * gini_impurity(y[current_mask])
        + len(y[~current_mask]) * gini_impurity(y[~current_mask])
    ) / len(y)
    improved = True

    while improved:
        improved = False
        best_addition = None
        for f in candidate_features:
            combined_mask = np.any(X[:, base_features + [f]] == 1, axis=1)
            weighted = (
                len(y[combined_mask]) * gini_impurity(y[combined_mask])
                + len(y[~combined_mask]) * gini_impurity(y[~combined_mask])
            ) / len(y)
            if weighted < best_impurity - 1e-9:
                best_impurity = weighted
                best_addition = f
                improved = True
        if improved:
            base_features.append(best_addition)
            candidate_features.remove(best_addition)
    return base_features


class GreedyORDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = (
            feature_names.tolist()
            if feature_names is not None
            else [f"f{i}" for i in range(X.shape[1])]
        )
        self.tree = self._grow(X, y, depth=0, features=list(range(X.shape[1])))

    def _grow(self, X, y, depth, features):
        node = {}
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            node["type"] = "leaf"
            counts = Counter(y)
            total = len(y)
            # Store both predicted class and probability distribution
            node["class"] = counts.most_common(1)[0][0]
            node["prob"] = {cls: count / total for cls, count in counts.items()}
            return node

        best_f, (left_mask, right_mask), improvement = best_split(X, y, features)
        if best_f is None or improvement <= 0:
            node["type"] = "leaf"
            counts = Counter(y)
            total = len(y)
            # Store both predicted class and probability distribution
            node["class"] = counts.most_common(1)[0][0]
            node["prob"] = {cls: count / total for cls, count in counts.items()}
            return node

        # Expand with OR combinations
        or_features = greedy_or_expand(
            X, y, [best_f], [f for f in features if f != best_f]
        )
        # Update for child nodes
        new_features = [f for f in features if f not in or_features]
        combined_mask = np.any(X[:, or_features] == 1, axis=1)

        node["type"] = "node"
        node["features"] = [self.feature_names[f] for f in or_features]
        node["left"] = self._grow(
            X[combined_mask], y[combined_mask], depth + 1, new_features,
        )
        node["right"] = self._grow(
            X[~combined_mask], y[~combined_mask], depth + 1, new_features,
        )
        return node

    def predict_one(self, x, node=None):
        node = node or self.tree
        if node["type"] == "leaf":
            return node["class"]
        feature_indices = [self.feature_names.index(f) for f in node["features"]]
        if np.any(x[feature_indices] == 1):
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def print_tree(node, indent=""):
    """Pretty-print the decision tree."""
    if node["type"] == "leaf":
        print(f"{indent}class: {node['class']} ({node['prob'][node['class']]:.2f})")
        return
    features = " OR ".join(node["features"])
    print(f"{indent}if ({features}):")
    print_tree(node["left"], indent + "    ")
    print(f"{indent}else:")
    print_tree(node["right"], indent + "    ")


def generate_texts_from_cnf(formula, error=0.0, completeness=1.0, seed=None):
    """
    formula: list of clauses, each clause is list of words, e.g. [["cats","dogs"], ["house","mice"]]
    error: fraction of labels to flip
    completeness: fraction of all possible combinations to include
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Collect all words
    words = sorted(set(word for clause in formula for word in clause))
    
    # 2. Generate all combinations (truth table) of presence/absence
    all_combinations = list(itertools.product([0,1], repeat=len(words)))

    # 3. Determine which combinations satisfy the CNF formula
    def satisfies_cnf(comb):
        word_presence = dict(zip(words, comb))
        return all(any(word_presence[w] for w in clause) for clause in formula)
    
    # Filter combinations based on CNF
    valid_combinations = [(comb, satisfies_cnf(comb)) for comb in all_combinations]
    
    # 4. Sample combinations based on completeness
    k = max(1, int(len(valid_combinations) * completeness))
    sampled = valid_combinations if completeness >= 1.0 else random.sample(valid_combinations, k)
    
    texts, labels = [], []
    for comb, label in sampled:
        # Generate text string from combination
        text_words = [w for i,w in enumerate(words) if comb[i]]
        text = " ".join(text_words)
        texts.append(text)
        labels.append(label)
    
    # 5. Flip some labels according to error
    num_flip = int(error * len(sampled))
    flip_indices = random.sample(range(len(sampled)), num_flip)
    for i in flip_indices:
        labels[i] = 1 - labels[i]
    
    return texts, np.array(labels)




# --- Example Usage ---
if __name__ == "__main__":
    # --- Example usage ---
    formula = [["cats","dogs","mice"], ["house","wohnung"], ["bowl", "box"]]  # (cats OR dogs OR mice) AND (house OR wohnung)
    texts, labels = generate_texts_from_cnf(formula, error=0.0, completeness=1.0, seed=42)

    for t,l in zip(texts, labels):
        print(f"{t} → {l}")

    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts).toarray()

    tree = GreedyORDecisionTree(max_depth=3)
    tree.fit(X, labels, feature_names=vectorizer.get_feature_names_out())

    from pprint import pprint

    pprint(tree.tree)

    preds = tree.predict(X)
    print("Predictions:", preds)

    print_tree(tree.tree)
