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
from tqdm import tqdm


def fast_gini_both(mask, y_true, min_samples_split):
    """Compute Gini impurity for mask and its complement in one pass."""
    n_total = len(y_true)

    # Left side
    n_left = mask.sum()
    if n_left < min_samples_split or n_left > n_total - min_samples_split:
        # return high impuriy to avoid useless splits
        return 1.0, False

    sum_left = np.sum(y_true[mask], dtype=np.float64)  # sum of positives on left
    p_left = sum_left / n_left
    left_imp = 2 * p_left * (1 - p_left)

    # Right side
    n_right = n_total - n_left
    sum_right = y_true.sum() - sum_left
    p_right = sum_right / n_right
    right_imp = 2 * p_right * (1 - p_right)

    return (n_left * left_imp + n_right * right_imp) / n_total, True


def best_split(X, y, features, min_samples_split):
    """Find best split feature (sparse version, optimized with fast_gini)."""
    if not features:
        return None, 0.0, []

    best_feature = None
    best_impurity = 1.0

    # Compute current impurity once
    p_total = np.mean(y)
    initial_impurity = 2 * p_total * (1 - p_total)

    improvements = []
    invalid_features = []
    for f in features:
        # Sparse column slice
        col = X[:, f]
        mask = col.getnnz(axis=1) > 0  # rows where feature is present

        weighted, valid_split = fast_gini_both(mask, y, min_samples_split)
        if not valid_split:
            invalid_features.append(f)
            continue
        improvements.append((f, initial_impurity - weighted))

        if weighted < best_impurity:
            best_impurity = weighted
            best_feature = f

    # Sort features by improvement (descending)
    best_sorted_features = sorted(improvements, key=lambda x: x[1], reverse=True)

    return best_feature, best_impurity, initial_impurity, best_sorted_features, invalid_features


def greedy_or_expand(X, y, base_features, candidate_features, current_impurity, min_impurity_decrease, min_samples_split):
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

    best_impurity = current_impurity

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
            weighted, valid_split = fast_gini_both(mask, y, min_samples_split)
            if valid_split:
                weighted_impurities.append(weighted)

        if weighted_impurities: # found suitable or features
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
                # print("added OR feature", best_addition)

    return base_features


class GreedyORDecisionTree:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_impurity_decrease_range=[0.01,0.03],
        top_k_or_candidates=500,
        verbose=False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.top_k_or_candidates = top_k_or_candidates
        self._tree = None
        self._verbose = verbose
        self._grow_counter = 0
        self._grow_total = 2 ** (max_depth) - 1
        self._total_docs = -1

    def fit(self, X, y, feature_names=None):
        self._total_docs = X.shape[0]
        self._grow_counter = 0
        self.feature_names = (
            feature_names.tolist()
            if feature_names is not None
            else [f"f{i}" for i in range(X.shape[1])]
        )

        # Initialize tqdm progress bar
        if self._verbose:
            self._pbar = tqdm(
                total=2 ** (self.max_depth) - 1, desc="Growing Tree", ncols=80
            )
        else:
            self._pbar = None

        # Convert X to CSC once for fast column access
        self._tree = self._grow(X.tocsc(), y, depth=0, features=list(range(X.shape[1])))

        self._prune_pure_subtrees()

    def _grow(self, X, y, depth, features):
        node = {}
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node["type"] = "leaf"
            counts = Counter(y)
            total = len(y)
            # Store both predicted class and probability distribution
            node["class"] = counts.most_common(1)[0][0]
            node["counts"] = counts
            node["prob"] = {cls: count / total for cls, count in counts.items()}
            return node

        # if self._verbose:
        #     self._grow_counter += 1
        #     progress = 100 * self._grow_counter / self._grow_total
        #     print(
        #         f"GROWING NODE {self._grow_counter}/{self._grow_total} ({progress:.1f}%) at depth {depth}"
        #     )
        if self._verbose and hasattr(self, "_pbar") and self._pbar is not None:
            self._grow_counter += 1
            self._pbar.update(1)
            self._pbar.set_postfix(depth=depth)

        best_feature, best_impurity, initial_impurity, best_sorted_features, invalid_features = best_split(X, y, features, self.min_samples_split)

        if best_feature is None or initial_impurity - best_impurity <= 0:
            node["type"] = "leaf"
            counts = Counter(y)
            total = len(y)
            # Store both predicted class and probability distribution
            node["class"] = counts.most_common(1)[0][0]
            node["counts"] = counts
            node["prob"] = {cls: count / total for cls, count in counts.items()}
            return node

        # Take top-k features to try for OR expansion
        top_candidates = [
            f for f, imp in best_sorted_features[: self.top_k_or_candidates] if f != best_feature
        ]

        # Expand with OR combinations
        or_features = greedy_or_expand(
            X,
            y,
            [best_feature],
            top_candidates,
            current_impurity=best_impurity,
            min_impurity_decrease=self.scaled_min_impurity_decrease(X.shape[0]), # min_impurity_decrease scales inversely with number of remaining docs
            min_samples_split=self.min_samples_split,
            
        )
        # Update for child nodes
        new_features = [f for f in features if f not in (or_features + invalid_features)]
        combined_mask = X[:, or_features].getnnz(axis=1) > 0

        node["type"] = "node"
        node["features"] = [self.feature_names[f] for f in or_features]
        node["feature_indices"] = or_features
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

    def scaled_min_impurity_decrease(self, n_samples):
        """
        Scale min_impurity_decrease linearly between given bounds.

        Parameters
        ----------
        n_samples : int
            Number of samples in the current node.
        n_total : int
            Total number of samples at the root.
        min_impurity_decrease_range : [float, float]
            [min_value_at_root, max_value_at_small_nodes]

        Returns
        -------
        float
            Scaled min_impurity_decrease for this node.
        """
        min_val, max_val = self.min_impurity_decrease_range
        # Linear interpolation: goes from min_val (root) → max_val (deep)
        fraction = 1 - (n_samples / self._total_docs)
        return min_val + (max_val - min_val) * fraction

    def predict(self, X):
        """Vectorized prediction for sparse CSR matrix X."""
        n_samples = X.shape[0]
        preds = np.empty(n_samples, dtype=int)

        # Stack for DFS traversal: (node, sample_indices)
        stack = [(self._tree, np.arange(n_samples))]

        while stack:
            node, idx = stack.pop()

            if node["type"] == "leaf":
                preds[idx] = node["class"]
                continue

            # Extract precomputed indices
            feature_indices = node["feature_indices"]

            # Compute OR condition efficiently for all rows in this subset
            # X[idx][:, feature_indices] is a sparse submatrix
            mask = X[idx][:, feature_indices].getnnz(axis=1) > 0

            # Send samples left or right
            if mask.any():
                stack.append((node["left"], idx[mask]))
            if (~mask).any():
                stack.append((node["right"], idx[~mask]))

        return preds

    def _prune_pure_subtrees(self):
        """
        Remove unnecessary decision nodes where all descendant leaves
        predict the same class. This simplifies the tree without changing
        its predictions.

        After pruning, leaf counts and probabilities are recalculated
        from all descendant leaves.
        """
        if self._tree is None:
            raise ValueError("Tree has not been trained yet. Call fit() first.")

        def aggregate_counts(node):
            """
            Recursively aggregate class counts for a node.
            Returns a Counter of class counts under this node.
            """
            if node["type"] == "leaf":
                return Counter(node["counts"])
            left_counts = aggregate_counts(node["left"])
            right_counts = aggregate_counts(node["right"])
            return left_counts + right_counts

        def prune(node):
            """
            Recursively prune pure subtrees.
            Returns (is_pure, pure_class).
            """
            if node["type"] == "leaf":
                return True, node["class"]

            left_pure, left_class = prune(node["left"])
            right_pure, right_class = prune(node["right"])

            # If both subtrees pure and of same class → collapse
            if left_pure and right_pure and left_class == right_class:
                # Aggregate counts from children
                total_counts = aggregate_counts(node)
                total = sum(total_counts.values())
                probs = {cls: count / total for cls, count in total_counts.items()}

                node.clear()
                node["type"] = "leaf"
                node["class"] = left_class
                node["counts"] = total_counts
                node["prob"] = probs
                return True, left_class

            return False, None

        prune(self._tree)

    def export_tree(self, format="pretty", feature_names=None, verbose=False):
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
        if self._tree is None:
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
            return self._tree
        elif format == "json":
            return json.dumps(self._tree, indent=4)
        elif format == "pretty":
            lines = []

            def recurse(node, indent=""):
                if node["type"] == "leaf":
                    leaf_text = f"{indent}class: {node['class']}"
                    if verbose:
                        leaf_text += f" ({node['prob'][node['class']]:.6g}, {node['counts']})"
                    lines.append(
                        leaf_text
                    )
                else:
                    features = " OR ".join(get_name(f) for f in node["features"])
                    lines.append(f"{indent}if ({features}):")
                    recurse(node["left"], indent + "    ")
                    lines.append(f"{indent}else:")
                    recurse(node["right"], indent + "    ")

            recurse(self._tree)
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown export format '{format}'")

    def __repr__(self):
        params = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):  # skip hidden
                params.append(f"{k}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"


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


# if __name__ == "__main__":
def main():
    def f(d):
        return (
            not (d["cats"] or d["dogs"] or d["mice"])
            and (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"])
        )

    variables = ["cats", "dogs", "mice", "house", "wohnung", "bowl", "box"]
    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        error=0.1,
        completeness=0.9,
        seed=42,
        doc_count=1_000_0,
        word_pool_size=400_0,
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

    tree = GreedyORDecisionTree(
        max_depth=4,
        min_impurity_decrease_range=[0.01, 0.03],
        top_k_or_candidates=500,
        verbose=True,
        min_samples_split=5,
    )
    start_time = time.time()
    tree.fit(X, np.array(labels), feature_names=vectorizer.get_feature_names_out())
    end_time = time.time()

    print()
    print(tree.export_tree(feature_names=vectorizer.get_feature_names_out()))
    preds = tree.predict(X)
    true_preds = np.array(
        [f({var: int(var in text.split()) for var in variables}) for text in texts]
    )
    precision = precision_score(true_preds, preds)
    recall = recall_score(true_preds, preds)
    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Fit time: {end_time - start_time:.4f} seconds")

    X_test = vectorizer.transform(
        [
            "bowl house",
            "bowl wohnung house",
            "bowl wohnung",
            "cats bowl house",
            "cats mice",
            "hello",
        ]
    )
    print("result", tree.predict(X_test))


from line_profiler import LineProfiler

if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(GreedyORDecisionTree.fit)

    lp.run("main()")
    lp.print_stats()