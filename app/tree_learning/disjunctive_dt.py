import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
from scipy.sparse import csr_matrix
import random
import itertools
import math
import time
import json

def gini_impurity(y):
    """Compute Gini impurity."""
    if len(y) == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs**2)


def fast_gini(mask, y_true):
    if not np.any(mask):
        return 0.0
    p = np.mean(y_true[mask])
    print("new", np.sum(y_true[mask], dtype=np.float64)/mask.sum())
    print("old", np.mean(y_true[mask]))
    return 2 * p * (1 - p)

def fast_gini_both(mask, y_true):
    """Compute Gini impurity for mask and its complement in one pass."""
    n_total = len(y_true)

    # Left side
    n_left = mask.sum()
    if n_left == 0 or n_left == n_total:
        # One side empty → impurity 0 for both
        return 0.0

    sum_left = np.sum(y_true[mask], dtype=np.float64)  # sum of positives on left
    p_left = sum_left / n_left
    left_imp = 2 * p_left * (1 - p_left)

    # Right side
    n_right = n_total - n_left
    sum_right = y_true.sum() - sum_left
    p_right = sum_right / n_right
    right_imp = 2 * p_right * (1 - p_right)

    return (n_left * left_imp + n_right * right_imp) / n_total


def best_split(X, y, features):
    """Find best split feature (sparse version, optimized with fast_gini)."""
    if not features:
        return None, (None, None), 0.0

    best_feature = None
    best_impurity = 1.0

    # Compute current impurity once
    p_total = np.mean(y)
    current_impurity = 2 * p_total * (1 - p_total)

    improvements = []
    for f in features:
        # Sparse column slice
        col = X[:, f]
        mask = col.getnnz(axis=1) > 0  # rows where feature is present

        weighted = fast_gini_both(mask, y)
        if weighted == 0:
            continue
        improvements.append((f, current_impurity - weighted))

        if weighted < best_impurity:
            best_impurity = weighted
            best_feature = f

    # Sort features by improvement (descending)
    sorted_features = sorted(improvements, key=lambda x: x[1], reverse=True)        

    improvement = current_impurity - best_impurity
    return best_feature, improvement, sorted_features



def greedy_or_expand(X, y, base_features, candidate_features, min_impurity_decrease):
    """
    Try adding features with OR (sparse version, fully vectorized).

    Parameters
    ----------
    X : csr_matrix, shape (n_samples, n_features)
        Binary sparse feature matrix.
    y : array-like, shape (n_samples,)
        Labels.
    base_features : list[int]
        Already selected features.
    candidate_features : list[int]
        Candidate features for OR expansion.
    min_impurity_decrease : float
        Minimum required improvement to add a feature.

    Returns
    -------
    base_features : list[int]
        Selected OR features after greedy expansion.
    """
    n_samples = X.shape[0]

    # Compute initial combined mask
    if base_features:
        combined_mask = X[:, base_features].getnnz(axis=1) > 0
    else:
        combined_mask = np.zeros(n_samples, dtype=bool)

    # Initial weighted impurity
    best_impurity = (
        len(y[combined_mask]) * gini_impurity(y[combined_mask])
        + len(y[~combined_mask]) * gini_impurity(y[~combined_mask])
    ) / n_samples

    col_masks = {f: (X[:, f].getnnz(axis=1) > 0) for f in candidate_features}
    improved = True
    while improved and candidate_features:
        improved = False

        # Vectorized: compute OR mask for all candidates
        # X[:, candidate_features] returns sparse; ensure CSR for row slicing
        candidates_matrix = X[:, candidate_features].tocsr()
        # OR operation with base_features mask
        # Convert combined_mask to int (0/1) and add to each candidate column
        # result >0 gives OR
        combined_masks = candidates_matrix.copy()
        combined_masks.data = np.ones_like(combined_masks.data)  # ensure binary

        # Weighted impurities for each candidate
        weighted_impurities = []
        for f in candidate_features:
            mask = combined_mask | col_masks[f]
            weighted = fast_gini_both(mask, y)
            weighted_impurities.append(weighted)

        weighted_impurities = np.array(weighted_impurities)
        improvements = best_impurity - weighted_impurities

        # Pick best candidate
        idx = np.argmax(improvements)
        if improvements[idx] >= min_impurity_decrease:
            best_addition = candidate_features[idx]
            base_features.append(best_addition)
            candidate_features.remove(best_addition)
            best_impurity = weighted_impurities[idx]
            # update combined_mask for next round
            combined_mask = combined_mask | (X[:, best_addition].getnnz(axis=1) > 0)
            improved = True
            print("added OR feature", best_addition)

    return base_features


class GreedyORDecisionTree:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_impurity_decrease=0.01,
        verbose=False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.verbose = verbose
        self._grow_counter = 0
        self._grow_total = 2**(max_depth)-1

    def fit(self, X, y, feature_names=None):
        self._grow_counter = 0
        self.feature_names = (
            feature_names.tolist()
            if feature_names is not None
            else [f"f{i}" for i in range(X.shape[1])]
        )
        # Convert X to CSC once for fast column access
        self.tree = self._grow(X.tocsc(), y, depth=0, features=list(range(X.shape[1])))

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
        
        if self.verbose:
            self._grow_counter += 1
            progress = 100 * self._grow_counter / self._grow_total
            print(
                f"GROWING NODE {self._grow_counter}/{self._grow_total} ({progress:.1f}%) at depth {depth}"
            )

        best_f, improvement, sorted_features = best_split(X, y, features)
    

        if best_f is None or improvement <= 0:
            node["type"] = "leaf"
            counts = Counter(y)
            total = len(y)
            # Store both predicted class and probability distribution
            node["class"] = counts.most_common(1)[0][0]
            node["prob"] = {cls: count / total for cls, count in counts.items()}
            return node

        # Take top-k features to try for OR expansion
        top_k = 500  # for example
        top_candidates = [f for f, imp in sorted_features[:top_k] if f != best_f]

        # Expand with OR combinations
        or_features = greedy_or_expand(
            X,
            y,
            [best_f],
            top_candidates,
            min_impurity_decrease=self.min_impurity_decrease,
        )
        # Update for child nodes
        new_features = [f for f in features if f not in or_features]
        combined_mask = X[:, or_features].getnnz(axis=1) > 0

        node["type"] = "node"
        node["features"] = [self.feature_names[f] for f in or_features]
        node["left"] = self._grow(
            X[combined_mask],
            y[combined_mask],
            depth + 1,
            new_features,
        )
        node["right"] = self._grow(
            X[~combined_mask],
            y[~combined_mask],
            depth + 1,
            new_features,
        )
        return node

    def predict_one(self, x, node=None):
        """
        Predict a single sample for a sparse row vector x (csr_matrix, shape 1×n_features).
        """
        node = node or self.tree
        if node["type"] == "leaf":
            return node["class"]

        # Get feature indices in the node
        feature_indices = [self.feature_names.index(f) for f in node["features"]]

        # Check OR condition over sparse features
        # x[0, feature_indices] gives a sparse slice
        if x[0, feature_indices].sum() > 0:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def export_tree(self, format="pretty", feature_names=None):
        """
        Export the trained tree.

        Parameters
        ----------
        format : str
            "pretty"  -> human-readable text (like print_tree)
            "dict"    -> nested dictionary
            "json"    -> JSON string
        feature_names : list[str] or None
            If provided, use these names in pretty-print instead of internal names.

        Returns
        -------
        str or dict
        """
        if self.tree is None:
            raise ValueError("Tree has not been trained yet. Call fit() first.")

        # Helper to map internal names to user-provided names
        def get_name(f):
            if feature_names is None:
                return f
            try:
                idx = int(f[1:])  # assumes internal names are fXXX
                return feature_names[idx]
            except:
                return f

        if format == "dict":
            return self.tree
        elif format == "json":
            return json.dumps(self.tree, indent=4)
        elif format == "pretty":
            lines = []

            def recurse(node, indent=""):
                if node["type"] == "leaf":
                    lines.append(f"{indent}class: {node['class']} ({node['prob'][node['class']]:.2f})")
                else:
                    features = " OR ".join(get_name(f) for f in node["features"])
                    lines.append(f"{indent}if ({features}):")
                    recurse(node["left"], indent + "    ")
                    lines.append(f"{indent}else:")
                    recurse(node["right"], indent + "    ")

            recurse(self.tree)
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown export format '{format}'")

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


def generate_texts_from_boolean(
    func,
    variables,
    error=0.0,
    completeness=1.0,
    seed=None,
    doc_count=1000,
    word_pool_size=1000,
    average_doc_length=15,
):
    """
    Generate data samples (combinations and labels) from any Boolean function.

    Parameters
    ----------
    func : callable
        A function that takes a dict {var_name: 0/1, ...} and returns True/False.
        Example: lambda d: (d['A'] and not d['B']) or d['C']
    variables : list[str]
        List of variable names.
    error : float
        Fraction of labels to flip (simulate noise).
    completeness : float
        Fraction of all possible combinations to include (subset sampling).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    samples : list[dict]
        List of dictionaries mapping variable names to 0/1 values.
    labels : list[int]
        List of 0/1 labels computed from func (after error/noise).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Generate all combinations
    all_combinations = list(itertools.product([0, 1], repeat=len(variables)))

    # 2. Apply completeness (subsampling)
    if completeness < 1.0:
        n_keep = int(len(all_combinations) * completeness)
        all_combinations = random.sample(all_combinations, n_keep)

    # 3. Evaluate Boolean function
    texts = []
    labels = []
    for comb in all_combinations:
        mapping = dict(zip(variables, comb))
        label = int(bool(func(mapping)))
        texts.append(" ".join(var for var, val in mapping.items() if val))
        labels.append(label)

    # 4. expand with random words
    # Duplicate until we have enough
    expand_factor = math.ceil(doc_count / len(texts))
    texts = (texts * expand_factor)[:doc_count]
    labels = (labels * expand_factor)[:doc_count]

    # Prepare random word pool
    word_pool = [f"r{i}" for i in range(word_pool_size - len(variables))]

    # Append random words to each text
    new_texts = []
    for text in texts:
        # Each doc gets 0–5 random filler words (you can change range)
        n_extra = random.randint(0, max(0, average_doc_length * 2 - len(variables)))
        extra_words = random.sample(word_pool, n_extra)
        new_text = f"{text} {' '.join(extra_words)}".strip()
        new_texts.append(new_text)

    # 5. Apply label noise (error)
    n_flip = int(len(labels) * error)
    flip_indices = np.random.choice(len(labels), n_flip, replace=False)
    for i in flip_indices:
        labels[i] = 1 - labels[i]

    return new_texts, labels


# --- Example Usage ---
# if __name__ == "__main__":
def main():
    f = (
        lambda d: not (d["cats"] or d["dogs"] or d["mice"])
        and (d["house"] or d["wohnung"])
        and (d["bowl"] or d["box"])
    )
    variables = ["cats", "dogs", "mice", "house", "wohnung", "bowl", "box"]
    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        error=0.1,
        completeness=0.7,
        seed=42,
        doc_count=1_000_000,
        word_pool_size=400_000,
        average_doc_length=30,
    )

    # --- Calculate actual statistics from the result ---
    actual_doc_count = len(texts)
    all_words = [word for text in texts for word in text.split()]
    unique_words = set(all_words)
    actual_word_pool_size = len(unique_words)
    avg_doc_length = sum(len(text.split()) for text in texts) / len(texts)

    # --- Print for verification ---
    print("Actual doc_count:", actual_doc_count)
    print("Actual word_pool_size:", actual_word_pool_size)
    print("Actual average_doc_length:", round(avg_doc_length, 2))

    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts)

    tree = GreedyORDecisionTree(max_depth=3, min_impurity_decrease=0.01, verbose=True)
    start_time = time.time()
    tree.fit(X, np.array(labels), feature_names=vectorizer.get_feature_names_out())
    end_time = time.time()

    from pprint import pprint

    pprint(tree.tree)

    # preds = tree.predict(X)
    # true_preds = np.array(
    #     [f({var: int(var in text.split()) for var in variables}) for text in texts]
    # )
    # precision = precision_score(true_preds, preds)
    # recall = recall_score(true_preds, preds)
    print_tree(tree.tree)
    # print()
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall:    {recall:.4f}")
    print(f"Prediction time: {end_time - start_time:.4f} seconds")

from line_profiler import LineProfiler
if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(best_split)

    lp.run('main()')
    lp.print_stats()
