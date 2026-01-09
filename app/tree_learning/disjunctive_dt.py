import numpy as np
from collections import Counter
from typing import List, Tuple, FrozenSet, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, fbeta_score, fbeta_score
from app.tree_learning.query_generation import rules_to_pubmed_query
from app.helper.helper import biased_random
import random
import itertools
import math
import re
import time
import jsonpickle
import numba
from tqdm import tqdm

from app.pubmed.retrieval import search_pubmed


def compute_sample_weight(class_weight: dict, y):
    """
    Compute sample weights for binary classification (labels 0 and 1).

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels (0 or 1).
    class_weight : dict
        Precomputed class weights, e.g., {0: 1.5, 1: 0.75}

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Weight for each sample.
    """
    assert 0 in class_weight and 1 in class_weight

    y = np.asarray(y)  # ensure it's a numpy array
    if not np.all(np.isin(y, [0, 1])):
        print(y)
        raise ValueError("All labels must be 0 or 1")

    # Assign weights based on class
    sample_weight = np.vectorize(class_weight.get)(y)
    return sample_weight


def compute_class_weight(class_weight, X, y):
    _n_samples = X.shape[0]
    n_class_1 = y.sum()
    n_class_0 = _n_samples - n_class_1

    if class_weight == "balanced":
        return {
            0: _n_samples / (2 * n_class_0),
            1: _n_samples / (2 * n_class_1),
        }
    elif class_weight is None:
        return {0: 1, 1: 1}
    else:
        return class_weight


@numba.njit
def union_sorted(a, b):
    i = j = 0
    out = np.empty(len(a) + len(b), dtype=a.dtype)
    k = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out[k] = a[i]
            k += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            out[k] = a[i]
            k += 1
            i += 1
        else:
            out[k] = b[j]
            k += 1
            j += 1

    while i < len(a):
        out[k] = a[i]
        k += 1
        i += 1

    while j < len(b):
        out[k] = b[j]
        k += 1
        j += 1

    return out[:k]


def calc_prob_class_1(y, sample_weight):
    w_total = np.sum(sample_weight)
    if w_total == 0:
        print("warning w_total = 0, all samples have weight 0")
        return 0.0, 0.0, False
    weighted_class_1 = np.sum(y * sample_weight)
    p_class_1 = weighted_class_1 / w_total
    return p_class_1, weighted_class_1, True


def gini(y, sample_weight):
    p_class_1, weighted_class_1, valid = calc_prob_class_1(y, sample_weight)
    if not valid:
        return 0.0, 0.0, False
    impurity = 2 * p_class_1 * (1 - p_class_1)
    return impurity, weighted_class_1, True


@numba.njit
def fast_gini_both(
    rows,
    y_true,
    sample_weight,
    w_class_1,
    min_samples_split,
    min_weight_fraction_leaf,
    total_sample_weight,
):
    n_total = len(y_true)
    n_left = len(rows)

    if n_left < min_samples_split or n_left > n_total - min_samples_split:
        return 1.0, False, False

    w_left = 0.0
    w_class_1_left = 0.0
    for i in range(n_left):
        w = sample_weight[rows[i]]
        w_class_1_left += y_true[rows[i]] * w
        w_left += w
    w_total = np.sum(sample_weight)
    w_right = w_total - w_left
    if (
        w_left / total_sample_weight < min_weight_fraction_leaf
        or w_right / total_sample_weight < min_weight_fraction_leaf
    ):
        return 1.0, False, False

    assert w_left >= 0
    assert w_right >= 0
    if (
        w_left == 0 or w_right == 0
    ):  # weights can be 0 due to bootsrapping (samples that do not occur)
        return 1.0, False, False

    p_left = w_class_1_left / w_left
    left_imp = 2 * p_left * (1 - p_left)
    w_class_1_right = w_class_1 - w_class_1_left

    p_right = w_class_1_right / w_right
    right_imp = 2 * p_right * (1 - p_right)

    return (w_left * left_imp + w_right * right_imp) / w_total, True, p_left > p_right


def best_split(
    X,
    y,
    features,
    min_samples_split,
    min_weight_fraction_leaf,
    total_sample_weight,
    sample_weight,
    features_subset=None,
    prefer_pos_splits=None,
):
    """Find best split feature (sparse version, optimized with fast_gini)."""
    if not features:
        return None, None, 0.0, [], []

    # Compute current impurity once
    initial_impurity, weighted_class_1, valid = gini(y=y, sample_weight=sample_weight)
    if not valid:
        return None, None, 0.0, [], []

    best_feature = None
    best_impurity = 1.0
    best_impurity_gain = 0.0
    improvements = []
    invalid_features = []
    for f in features:
        # Sparse column slice
        # col = X[:, f]
        # mask = col.getnnz(axis=1) > 0  # rows where feature is present
        start = X.indptr[f]
        end = X.indptr[f + 1]
        rows_with_feature = X.indices[start:end]

        weighted, valid_split, is_pos_split = fast_gini_both(
            rows=rows_with_feature,
            y_true=y,
            sample_weight=sample_weight,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            total_sample_weight=total_sample_weight,
            w_class_1=weighted_class_1,
        )
        if not valid_split:
            invalid_features.append(f)
            continue
        impurity_gain = initial_impurity - weighted
        if prefer_pos_splits and is_pos_split:
            impurity_gain *= prefer_pos_splits
        improvements.append((f, impurity_gain))

        if impurity_gain > best_impurity_gain: #weighted < best_impurity:
            if not features_subset or f in features_subset:
                best_impurity = weighted
                best_feature = f
                best_impurity_gain = impurity_gain

    # Sort features by improvement (descending)
    best_sorted_features = sorted(improvements, key=lambda x: x[1], reverse=True)
    return (
        best_feature,
        best_impurity,
        initial_impurity,
        best_sorted_features,
        invalid_features,
    )


def greedy_or_expand(
    X,
    y,
    base_features,
    candidate_features,
    current_impurity,
    min_impurity_decrease,
    min_samples_split,
    min_weight_fraction_leaf,
    total_sample_weight,
    sample_weight,
):
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
    # n_samples = X.shape[0]
    # Compute initial combined mask
    # if base_features:
    #     combined_mask = X[:, base_features].getnnz(axis=1) > 0
    # else:
    #     combined_mask = np.zeros(n_samples, dtype=bool)
    if base_features:
        combined_rows = np.unique(
            np.concatenate(
                [X.indices[X.indptr[f] : X.indptr[f + 1]] for f in base_features]
            )
        )
    else:
        combined_rows = np.array([], dtype=int)

    best_impurity = current_impurity

    # col_masks = {f: (X[:, f].getnnz(axis=1) > 0) for f in candidate_features}
    col_indices = {
        f: X.indices[X.indptr[f] : X.indptr[f + 1]] for f in candidate_features
    }
    weighted_class_1 = np.sum(y * sample_weight)
    improved = True

    while improved and candidate_features:
        improved = False

        # Vectorized: compute OR mask for all candidates
        # X[:, candidate_features] returns sparse; ensure CSR for row slicing
        # candidates_matrix = X[:, candidate_features].tocsr()
        # OR operation with base_features mask
        # Convert combined_mask to int (0/1) and add to each candidate column
        # result >0 gives OR
        # combined_masks = candidates_matrix.copy()
        # combined_masks.data = np.ones_like(combined_masks.data)  # ensure binary

        # Weighted impurities for each candidate
        # weighted_impurities = []
        best_candidate = None
        best_weighted_impurity = best_impurity
        best_improvement = 0.0

        for f in candidate_features:
            # mask = combined_mask | col_masks[f]
            # union_rows = np.union1d(combined_rows, col_indices[f])
            union_rows = union_sorted(
                combined_rows, col_indices[f]
            )  # slowest part (can be optimized. we do not need union only the counts from it)
            weighted, valid_split, is_pos_split = fast_gini_both(
                rows=union_rows,
                y_true=y,
                sample_weight=sample_weight,
                min_samples_split=min_samples_split,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                total_sample_weight=total_sample_weight,
                w_class_1=weighted_class_1,
            )
            # print(f, valid_split)
            if not valid_split:
                continue
            improvement = best_impurity - weighted

            # Keep best instantly
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = f
                best_weighted_impurity = weighted

        if best_candidate is not None and best_improvement >= min_impurity_decrease:
            base_features.append(best_candidate)
            candidate_features.remove(best_candidate)
            best_impurity = best_weighted_impurity
            combined_rows = np.union1d(
                combined_rows,
                X.indices[X.indptr[best_candidate] : X.indptr[best_candidate + 1]],
            )
            improved = True

    return base_features


class GreedyORDecisionTree:
    def __init__(
        self,
        max_depth=3,
        min_samples_split=2,
        min_weight_fraction_leaf=0.01,
        min_impurity_decrease_range=[0.01, 0.03],
        top_k_or_candidates=500,
        class_weight=None,
        max_features=None,
        randomize_max_feature=None,
        random_state=None,
        verbose=False,
        prefer_pos_splits=1.1, # multiply impurity gain of positive splits by prefer_pos_splits
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.top_k_or_candidates = top_k_or_candidates
        self.class_weight = class_weight
        self._tree = None
        self._verbose = verbose
        self._grow_counter = 0
        self._grow_total = 2 ** (max_depth) - 1
        self._n_samples = -1
        self._optimal_threshold = 0.5
        self._optimal_metric = None
        self._optimal_score = None
        self._possible_thresholds = set()
        self.max_features = max_features
        self.randomize_max_feature = randomize_max_feature
        self.random_state = np.random.RandomState(random_state)
        self.prefer_pos_splits = prefer_pos_splits

    def fit(self, X, y, feature_names=None, sample_weight=None):
        # self._n_samples = X.shape[0]
        self._grow_counter = 0
        self._feature_names = (
            feature_names.tolist()
            if feature_names is not None
            else [f"f{i}" for i in range(X.shape[1])]
        )
        if sample_weight is not None:
            sample_weight = sample_weight
        else:
            class_weight = compute_class_weight(self.class_weight, X, y)
            sample_weight = compute_sample_weight(class_weight, np.copy(y))
        self.total_sample_weight = np.sum(sample_weight)

        n_class_1 = y.sum()
        n_class_0 = self._n_samples - n_class_1
        if n_class_0 == 0 or n_class_1 == 0:
            print("All sample are of the same class")
            return -1
        self.class_weight = compute_class_weight(self.class_weight, X, y)
        # if self.class_weight == "balanced":
        #     self.class_weight  = {
        #         0: self._n_samples / (2 * n_class_0),
        #         1: self._n_samples / (2 * n_class_1),
        #     }
        # elif self.class_weight is None:
        #     self.class_weight  = {0: 1, 1: 1}

        # Initialize tqdm progress bar
        if self._verbose:
            self._pbar = tqdm(
                total=2 ** (self.max_depth) - 1, desc="Growing Tree", ncols=80
            )
        else:
            self._pbar = None

        # Convert X to CSC once for fast column access
        relevant_sample_mask = sample_weight > 0
        self._tree = self._grow(
            X[relevant_sample_mask, :].tocsc(),
            y[relevant_sample_mask],
            depth=0,
            features=list(range(X.shape[1])),
            sample_weight=sample_weight[relevant_sample_mask],
        )

        # self._find_optimal_threshold(
        #     X,
        #     y,
        #     metric="f2",
        #     constraint="recall",
        #     constraint_value=0.7,
        # )

    def _calc_node_stats(self, y, sample_weight=None):
        n_class_1 = int(np.sum(y))
        n_class_0 = int((len(y) - n_class_1))
        counts = {0: n_class_0, 1: n_class_1}
        if sample_weight is not None:
            p_class_1, weighted_class_1, valid = calc_prob_class_1(y, sample_weight)
        else:
            weighted_class_1 = n_class_1 * self.class_weight[1]
            weighted_class_0 = n_class_0 * self.class_weight[0]
            total_weight = weighted_class_0 + weighted_class_1
            p_class_1 = weighted_class_1 / total_weight
        return counts, p_class_1

    def _create_leaf(self, y, sample_weight):
        node = {}
        node["type"] = "leaf"
        counts, prob_class_1 = self._calc_node_stats(y, sample_weight=sample_weight)
        node["counts"] = counts
        node["prob_class_1"] = prob_class_1
        self._possible_thresholds.add(prob_class_1)
        return node

    def _grow(self, X, y, depth, features, sample_weight):
        node = {}
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.sum(sample_weight) / self.total_sample_weight
            < self.min_weight_fraction_leaf
            or len(np.unique(y)) == 1
        ):
            return self._create_leaf(y, sample_weight=sample_weight)

        if self._verbose and hasattr(self, "_pbar") and self._pbar is not None:
            self._grow_counter += 1
            self._pbar.update(1)
            self._pbar.set_postfix(depth=depth)

        # --- Random feature subsampling (Random Forest style) ---
        # IMPORTANT: OR combinations are only explored among the sampled features because of top_candidates
        if self.max_features is not None:
            k = self._compute_max_features(len(features))
            features_subset = set(
                self.random_state.choice(features, size=k, replace=False)
            )
        else:
            features_subset = None

        (
            best_feature,
            best_impurity,
            initial_impurity,
            best_sorted_features,
            invalid_features,
        ) = best_split(
            X,
            y,
            features,
            self.min_samples_split,
            sample_weight=sample_weight,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            total_sample_weight=self.total_sample_weight,
            features_subset=features_subset,
            prefer_pos_splits=self.prefer_pos_splits
        )
        if best_feature is None or initial_impurity - best_impurity <= 0:
            return self._create_leaf(y, sample_weight=sample_weight)

        # Take top-k features to try for OR expansion
        top_candidates = [
            f
            for f, imp in best_sorted_features[: self.top_k_or_candidates]
            if f != best_feature
        ]

        # Expand with OR combinations
        or_features = greedy_or_expand(
            X,
            y,
            [best_feature],
            top_candidates,
            current_impurity=best_impurity,
            min_impurity_decrease=self.scaled_min_impurity_decrease(
                X.shape[0]
            ),  # min_impurity_decrease scales inversely with number of remaining docs
            min_samples_split=self.min_samples_split,
            sample_weight=sample_weight,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            total_sample_weight=self.total_sample_weight,
        )
        # Update for child nodes
        excluded_features = set(or_features)
        excluded_features.update(invalid_features)
        new_features = [f for f in features if f not in excluded_features]
        combined_mask = X[:, or_features].getnnz(axis=1) > 0

        node["type"] = "node"
        counts, prob_class_1 = self._calc_node_stats(y, sample_weight=sample_weight)
        node["counts"] = counts
        node["prob_class_1"] = prob_class_1
        node["features"] = [self._feature_names[f] for f in or_features]
        node["feature_indices"] = or_features
        node["left"] = (
            self._grow(  # TODO performance instead of passing masked X,y and sampleweight only pass rows (no copies of X,y and smapleweight -> speed?)
                X[combined_mask],
                y[combined_mask],
                depth + 1,
                new_features,
                sample_weight=sample_weight[combined_mask],
            )
        )
        node["right"] = self._grow(
            X[~combined_mask],
            y[~combined_mask],
            depth + 1,
            new_features,
            sample_weight=sample_weight[~combined_mask],
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
        fraction = 1 - (n_samples / self._n_samples)
        return min_val + (max_val - min_val) * fraction

    def predict_proba(self, X):
        """Vectorized prediction for sparse CSR matrix X."""
        n_samples = X.shape[0]
        preds = np.empty(n_samples, dtype=float)

        # Stack for DFS traversal: (node, sample_indices)
        stack = [(self._tree, np.arange(n_samples), True)]

        while stack:
            node, idx, always_right = stack.pop()

            if node["type"] == "leaf":
                preds[idx] = 0.0 if always_right else node["prob_class_1"]
                continue

            # Extract precomputed indices
            feature_indices = node["feature_indices"]

            # Compute OR condition efficiently for all rows in this subset
            # X[idx][:, feature_indices] is a sparse submatrix
            mask = X[idx][:, feature_indices].getnnz(axis=1) > 0

            # Send samples left or right
            if mask.any():
                stack.append((node["left"], idx[mask], always_right & False))
            if (~mask).any():
                stack.append((node["right"], idx[~mask], always_right & True))

        return preds

    def predict(self, X):
        """
        Predict binary classes for input samples based on probability threshold.

        Returns
        -------
        np.ndarray
            Binary class predictions (0 or 1).
        """
        if self._tree is None:
            raise ValueError("Tree not trained. Call fit() first.")
        probs = self.predict_proba(X)
        preds = (probs >= self._optimal_threshold - 1e-8).astype(int)
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
                weighted_class_0 = total_counts[0] * self.class_weight.get(0, 1)
                weighted_class_1 = total_counts[1] * self.class_weight.get(1, 1)
                total_weight = weighted_class_0 + weighted_class_1
                probs = {
                    0: weighted_class_0 / total_weight,
                    1: weighted_class_1 / total_weight,
                }
                node.clear()
                node["type"] = "leaf"
                node["class"] = left_class
                node["counts"] = total_counts
                node["prob_class_1"] = probs[1]
                node["pruned"] = True
                return True, left_class

            return False, None

        prune(self._tree)

    def _find_optimal_threshold(
        self,
        X,
        y_true,
        metric="precision",
        constraint=None,
        term_expansions=None,
    ):
        """
        Find the probability threshold maximizing the chosen metric.
        Optionally apply a hard constraint on another metric.
        """
        if self._tree is None:
            raise ValueError("Tree not trained. Call fit() first.")

        probs = self.predict_proba(X)
        best_score = -float("inf")
        final_constraint_score = -float("inf")
        best_threshold = None

        def pubmed_precision(y_true, y_pred, term_expansions):
            """Compute precision using PubMed-derived FP count."""
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            # Query PubMed for these predicted positives
            pubmed_query, query_size = self.pubmed_query(term_expansions)
            if not pubmed_query:
                return 0.0
            count_retrieved = int(search_pubmed(pubmed_query)["Count"])
            if count_retrieved == 0:
                return 0.0
            return tp / count_retrieved

        def get_metric(y_true, y_pred, name, term_expansions):
            if name == "pubmed_precision":
                return pubmed_precision(y_true, y_pred, term_expansions)
            if name == "pubmed_count":
                pubmed_query, query_size = self.pubmed_query(term_expansions)
                if not pubmed_query:
                    return -float("inf")
                return -1 * int(search_pubmed(pubmed_query)["Count"])
            elif name.startswith("pubmed_f"):
                prec = pubmed_precision(y_true, y_pred, term_expansions)
                match = re.match(r"pubmed_f(\d+(\.\d+)?)", name)
                beta = float(match.group(1))
                recall = recall_score(y_true, y_pred, zero_division=0)
                if prec + recall == 0:
                    return 0.0
                return (1 + beta**2) * (prec * recall) / (beta**2 * prec + recall)
            elif name == "precision":
                return precision_score(y_true, y_pred, zero_division=0)
            elif name == "recall":
                return recall_score(y_true, y_pred, zero_division=0)
            elif name.startswith("f"):
                match = re.match(r"f(\d+(\.\d+)?)", name)
                beta = float(match.group(1))
                return fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {name}")

        for t in sorted(set(self._possible_thresholds) - {0.0}):
            self._optimal_threshold = t  # just temporary for calculations
            y_pred = (probs >= t - 1e-8).astype(int)
            main_score = get_metric(y_true, y_pred, metric, term_expansions)

            if constraint:
                constraint_score = get_metric(
                    y_true, y_pred, constraint["metric"], term_expansions
                )

                if constraint_score < constraint["value"]:
                    # Does NOT satisfy constraint
                    if final_constraint_score < constraint["value"]:
                        # no solution that satisifies constraint found yet
                        if final_constraint_score < constraint_score:
                            # current solution is closer to satisifying constraint
                            best_score = main_score
                            best_threshold = t
                            final_constraint_score = constraint_score
                    continue

            # If we reach here → the constraint is satisfied OR we have no constraint
            if main_score > best_score:
                best_score = main_score
                best_threshold = t
                if constraint:
                    final_constraint_score = constraint_score

        self._optimal_threshold = best_threshold
        self._optimal_score = best_score
        self._optimal_metric = metric
        if final_constraint_score == -float("inf"):
            final_constraint_score = None
        return self._optimal_threshold, self._optimal_score, final_constraint_score

    def to_json(self) -> str:
        """
        Export the tree to a JSON string, removing non-serializable attributes.
        """
        # Create a shallow copy of the object's __dict__
        tree_copy = self.__dict__.copy()

        # Remove attributes that are not serializable (like tqdm progress bars)
        for attr in ["_feature_names", "_possible_thresholds", "_pbar"]:
            tree_copy.pop(attr, None)

        # Encode the cleaned dictionary using jsonpickle
        return jsonpickle.encode(tree_copy, unpicklable=True)

    def _recompute_possible_thresholds(self):
        """Rebuilds the set of possible thresholds from all leaf nodes."""
        thresholds = set()

        def recurse(node):
            thresholds.add(node["prob_class_1"])
            if node["type"] == "leaf":
                return
            recurse(node["left"])
            recurse(node["right"])

        recurse(self._tree)
        return thresholds

    @classmethod
    def from_json(cls, json_input: str):
        """
        Load a GreedyORDecisionTree from a JSON string exported by to_json().
        """
        # Decode JSON into a dict of attributes
        data_dict = jsonpickle.decode(json_input)

        # Create an empty instance
        tree = cls.__new__(cls)

        # Restore all attributes
        for k, v in data_dict.items():
            setattr(tree, k, v)

        if (
            not hasattr(tree, "_possible_thresholds")
            or tree._possible_thresholds is None
        ):
            tree._possible_thresholds = tree._recompute_possible_thresholds()

        return tree

    def _node_class(self, node):
        # Helper to determine the class of a node
        return int(node["prob_class_1"] >= self._optimal_threshold - 1e-8)

    def _all_same_class(self, node):
        # Helper to check if all descendants have the same class
        if node["type"] == "leaf":
            return self._node_class(node)
        left_class = self._all_same_class(node["left"])
        right_class = self._all_same_class(node["right"])
        if left_class == right_class:
            return left_class
        return None  # Mixed classes

    def pretty_print(self, feature_names=None, verbose=False, prune=False):
        """
        Prints _tree.

        Parameters
        ----------
        feature_names : list[str] or None
            If provided, use these names in pretty-print instead of internal names.
        prune : bool
            If True, merge nodes where all descendants have the same class.

        Returns
        -------
        str
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

        lines = []

        def recurse(node, indent=""):
            if prune:
                same_class = self._all_same_class(node)
                if same_class is not None:
                    # Merge node into a leaf
                    leaf_text = f"{indent}class: {same_class}"
                    if verbose:
                        leaf_text += f" ({node['prob_class_1']:.10f}>={self._optimal_threshold:.10f}), {node['counts']})"
                    lines.append(leaf_text)
                    return

            if node["type"] == "leaf":
                leaf_text = f"{indent}class: {self._node_class(node)}"
                if verbose:
                    leaf_text += f" ({node['prob_class_1']:.10f}>={self._optimal_threshold:.10f}), {node['counts']})"
                lines.append(leaf_text)
            else:
                features = " OR ".join(get_name(f) for f in node["features"])
                lines.append(f"{indent}if ({features}):")
                recurse(node["left"], indent + "    ")
                lines.append(f"{indent}else:")
                recurse(node["right"], indent + "    ")

        recurse(self._tree)
        return "\n".join(lines)

    def __repr__(self):
        params = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):  # skip hidden
                abrevation = "".join(w[0] for w in k.split("_"))
                params.append(f"{abrevation}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def get_tree_paths(self):
        """
        Extract all root-to-leaf rules from a GreedyORDecisionTree.

        Returns
        -------
        Set(Rule)
        """
        if self._tree is None:
            raise ValueError("Tree not trained. Call fit() first.")

        rules = set()

        def recurse(node, literals):
            same_class = self._all_same_class(node)
            if same_class is not None:
                if same_class == 1:
                    rules.add(frozenset(literals))
                return

            if node["type"] == "leaf":
                if node["prob_class_1"] >= self._optimal_threshold - 1e-8:
                    rules.add(frozenset(literals))
                return

            feature_indices = frozenset(node["feature_indices"])

            # LEFT = feature present (OR semantics)
            recurse(node["left"], literals | {(feature_indices, True)})

            # RIGHT = feature absent
            recurse(node["right"], literals | {(feature_indices, False)})

        recurse(self._tree, set())
        
        # Filter out rules that have only negative terms
        rules = {r for r in rules if any(t[-1] for t in r)}

        return rules

    def pubmed_query(self, feature_names, term_expansions: dict = None):
        all_rules = self.get_tree_paths()
        return rules_to_pubmed_query(
            rules=all_rules,
            feature_names=feature_names,
            term_expansions=term_expansions,
        )

    def get_feature_names(self):
        if self._tree is None:
            raise ValueError("Tree not trained. Call fit() first.")

        collected = set()

        def recurse(node):
            if node["type"] == "node":
                # Add all features at this node
                for f in node["features"]:
                    collected.add(f)

                # Recurse down both children
                recurse(node["left"])
                recurse(node["right"])

        recurse(self._tree)
        return collected

    def _compute_max_features(self, n_features):
        mf = self.max_features  # note: you have a typo, see below

        if mf is None:
            value = n_features
        elif isinstance(mf, int):
            value = max(1, min(mf, n_features))
        elif isinstance(mf, float):
            value = max(1, int(mf * n_features))
        elif mf == "sqrt":
            value = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            value = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {mf}")

        if self.randomize_max_feature:
            value = int(
                biased_random(
                    low=value,
                    high=n_features,
                    exponent=self.randomize_max_feature,  # higher for stronger bias toward low numbers
                    rng=self.random_state,
                )
            )
        return value


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
        texts.append([var for var, val in mapping.items() if val])
        labels.append(label)

    if word_pool_size > len(variables):
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
            if n_extra > len(word_pool):
                extra_words = word_pool
            else:
                extra_words = random.sample(word_pool, n_extra)
            new_text = text + extra_words
            new_texts.append(new_text)
        texts = new_texts

    # 5. Apply label noise (error)
    n_flip = int(len(labels) * error)
    flip_indices = np.random.choice(len(labels), n_flip, replace=False)
    for i in flip_indices:
        labels[i] = 1 - labels[i]

    return texts, labels


# if __name__ == "__main__":
def main():
    def f(d):
        return (
            not (d["cats"] or d["dogs"] or d["mice[mh]"])
            and (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"])
        )

    variables = ["cats", "dogs", "mice[mh]", "house", "wohnung", "bowl", "box"]
    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        error=0.1,
        completeness=0.9,
        seed=42,
        doc_count=500_000,
        word_pool_size=50_000,
        average_doc_length=60,
    )

    # --- Calculate actual statistics from the result ---
    actual_doc_count = len(texts)
    all_words = [word for text in texts for word in text]
    unique_words = set(all_words)
    actual_word_pool_size = len(unique_words)
    avg_doc_length = sum(len(text) for text in texts) / len(texts)

    # --- Print for verification ---
    print("Actual doc_count:", actual_doc_count)
    print("Actual word_pool_size:", actual_word_pool_size)
    print("Actual average_doc_length:", round(avg_doc_length, 2))

    # vectorizer = CountVectorizer(binary=True)
    # X = vectorizer.fit_transform(texts)
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        binary=True,
        # stop_words="english",
        # min_df=1,
        # max_df=0.6
    )
    X = vectorizer.fit_transform(texts)

    tree = GreedyORDecisionTree(
        max_depth=4,
        min_impurity_decrease_range=[0.01, 0.03],
        top_k_or_candidates=500,
        verbose=True,
        min_samples_split=1,
        class_weight="balanced",
    )
    start_time = time.time()
    tree.fit(X, np.array(labels), feature_names=vectorizer.get_feature_names_out())
    end_time = time.time()

    print()
    print(tree.pretty_print(verbose=True, prune=True))
    preds = tree.predict(X)
    true_preds = np.array(
        [f({var: int(var in text) for var in variables}) for text in texts]
    )
    precision = precision_score(true_preds, preds)
    recall = recall_score(true_preds, preds)
    print()
    print(f"Precision: {precision_score(np.array(labels), preds):.4f}")
    print(f"Recall:    {recall_score(np.array(labels), preds):.4f}")
    print(f"Precision (compared to formula): {precision:.4f}")
    print(f"Recall (compared to formula):    {recall:.4f}")
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

    tree_json = tree.to_json()
    print("tree_json", tree_json)
    tree_loaded = GreedyORDecisionTree.from_json(tree_json)
    print("tree_loaded", tree_loaded)
    print(tree_loaded.pretty_print(verbose=True, prune=True))
    print("result tree_loaded", tree_loaded.predict(X_test))
    print("pubmed", tree_loaded.pubmed_query({"dogs": ["dogs", "dog"]}))
    print(f"Fit time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp.add_function(greedy_or_expand)  # GreedyORDecisionTree._grow)

    lp.run("main()")
    lp.print_stats()
