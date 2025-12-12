import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, fbeta_score, fbeta_score
from scipy.sparse import csr_matrix
import random
import itertools
import math
import re
import time
import json
import inspect
import jsonpickle
from tqdm import tqdm
from copy import deepcopy

from app.pubmed.retrieval import search_pubmed

def fast_gini_both(mask, y_true, min_samples_split, class_weight):
    """Compute Gini impurity for mask and its complement in one pass."""
    n_total = len(y_true)

    # Left side
    n_left = mask.sum()
    if n_left < min_samples_split or n_left > n_total - min_samples_split:
        # return high impuriy to avoid useless splits
        return 1.0, False

    n_class_1_left = mask @ y_true
    # np.sum(y_true[mask], dtype=np.float64)  # number of positives on left
    n_class_0_left = n_left - n_class_1_left
    weighted_class_1_left = n_class_1_left * class_weight[1]
    weighted_class_0_left = n_class_0_left * class_weight[0]
    p_left = weighted_class_1_left / (weighted_class_1_left + weighted_class_0_left)
    left_imp = 2 * p_left * (1 - p_left)

    # Right side
    n_right = n_total - n_left
    n_class_1_right = y_true.sum() - n_class_1_left
    n_class_0_right = n_right - n_class_1_right
    weighted_class_1_right = n_class_1_right * class_weight[1]
    weighted_class_0_right = n_class_0_right * class_weight[0]
    p_right = weighted_class_1_right / (weighted_class_1_right + weighted_class_0_right)
    right_imp = 2 * p_right * (1 - p_right)


    weighted_left_total = weighted_class_1_left + weighted_class_0_left
    weighted_right_total = weighted_class_1_right + weighted_class_0_right
    weighted_total = weighted_left_total + weighted_right_total
    return (weighted_left_total * left_imp + weighted_right_total * right_imp) / weighted_total, True


def best_split(X, y, features, min_samples_split, class_weight):
    """Find best split feature (sparse version, optimized with fast_gini)."""
    if not features:
        return None, None, 0.0, [], []

    best_feature = None
    best_impurity = 1.0

    # Compute current impurity once
    n_class_1 = np.sum(y)
    n_class_0 = X.shape[0] - n_class_1
    weighted_class_0 = n_class_0 * class_weight[0]
    weighted_class_1 = n_class_1 * class_weight[1]
    p_total = weighted_class_1 / (weighted_class_0 + weighted_class_1)
    initial_impurity = 2 * p_total * (1 - p_total)

    improvements = []
    invalid_features = []
    for f in features:
        # Sparse column slice
        col = X[:, f]
        mask = col.getnnz(axis=1) > 0  # rows where feature is present
        weighted, valid_split = fast_gini_both(mask, y, min_samples_split, class_weight)
        if not valid_split:
            invalid_features.append(f)
            continue
        improvements.append((f, initial_impurity - weighted))

        if weighted < best_impurity:
            best_impurity = weighted
            best_feature = f

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
    class_weight=None,
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
            weighted, valid_split = fast_gini_both(
                mask, y, min_samples_split, class_weight=class_weight
            )
            if valid_split:
                weighted_impurities.append(weighted)

        if weighted_impurities:  # found suitable or features
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
        min_impurity_decrease_range=[0.01, 0.03],
        top_k_or_candidates=500,
        class_weight=None,
        verbose=False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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


    def fit(self, X, y, feature_names=None):
        self._n_samples = X.shape[0]
        self._grow_counter = 0
        self._feature_names = (
            feature_names.tolist()
            if feature_names is not None
            else [f"f{i}" for i in range(X.shape[1])]
        )
        n_class_1 = y.sum()
        n_class_0 = self._n_samples - n_class_1
        if n_class_0 == 0 or n_class_1 == 0:
            print("All sample are of the same class")
            return
        if self.class_weight == "balanced":
            self.class_weight  = {
                0: self._n_samples / (2 * n_class_0),
                1: self._n_samples / (2 * n_class_1),
            }
        elif self.class_weight is None:
            self.class_weight  = {0: 1, 1: 1}

        # Initialize tqdm progress bar
        if self._verbose:
            self._pbar = tqdm(
                total=2 ** (self.max_depth) - 1, desc="Growing Tree", ncols=80
            )
        else:
            self._pbar = None

        # Convert X to CSC once for fast column access
        self._tree = self._grow(X.tocsc(), y, depth=0, features=list(range(X.shape[1])))

        # self._find_optimal_threshold(
        #     X,
        #     y,
        #     metric="f2",
        #     constraint="recall",
        #     constraint_value=0.7,
        # )

    def _calc_node_stats(self, y):
        n_class_1 = int(np.sum(y))
        n_class_0 = int((len(y) - n_class_1))
        weighted_class_1 = n_class_1 * self.class_weight[1]
        weighted_class_0 = n_class_0 * self.class_weight[0]
        counts = {0: n_class_0, 1: n_class_1}
        total_weight = weighted_class_0 + weighted_class_1
        prob_class_1 = weighted_class_1 / total_weight
        return counts, prob_class_1

    def _create_leaf(self, y):
        node = {}
        node["type"] = "leaf"
        counts, prob_class_1 = self._calc_node_stats(y)
        node["counts"] = counts
        node["prob_class_1"] = prob_class_1
        self._possible_thresholds.add(prob_class_1)
        return node

    def _grow(self, X, y, depth, features):
        node = {}
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._create_leaf(y)

        if self._verbose and hasattr(self, "_pbar") and self._pbar is not None:
            self._grow_counter += 1
            self._pbar.update(1)
            self._pbar.set_postfix(depth=depth)

        (
            best_feature,
            best_impurity,
            initial_impurity,
            best_sorted_features,
            invalid_features,
        ) = best_split(
            X, y, features, self.min_samples_split, class_weight=self.class_weight
        )
        if best_feature is None or initial_impurity - best_impurity <= 0:
            return self._create_leaf(y)

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
            class_weight=self.class_weight,
        )
        # Update for child nodes
        new_features = [
            f for f in features if f not in (or_features + invalid_features)
        ]
        combined_mask = X[:, or_features].getnnz(axis=1) > 0

        node["type"] = "node"
        counts, prob_class_1 = self._calc_node_stats(y)
        node["counts"] = counts
        node["prob_class_1"] = prob_class_1
        node["features"] = [self._feature_names[f] for f in or_features]
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
                    1: weighted_class_1 / total_weight
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
            # print("count_retrieved", count_retrieved)
            # print("query size", len(pubmed_query))
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
                # print("prec", prec)
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

        print(self._possible_thresholds)
        for t in sorted(set(self._possible_thresholds) - {0.0}):
            self._optimal_threshold = t # just temporary for calculations
            y_pred = (probs >= t - 1e-8).astype(int)
            main_score = get_metric(y_true, y_pred, metric, term_expansions)
            
            # print(self.pretty_print(verbose=True))
            # print("t", t)
            # print("mainscore", main_score)
            # print("y_pred.sum()", y_pred.sum())
            # print()
            if constraint:
                constraint_score = get_metric(y_true, y_pred, constraint["metric"], term_expansions)
                
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
        # print("==============================optimal===============================", self._optimal_threshold)
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

        if not hasattr(tree, "_possible_thresholds") or tree._possible_thresholds is None:
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
            if not k.startswith('_'):  # skip hidden
                abrevation = ''.join(w[0] for w in k.split('_'))
                params.append(f"{abrevation}={v!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"

    def _expand_term(self, term_expansions, feature):
        if feature.endswith("[mh]"):
            # mesh terms
            # TODO maybe only add :noexp on negative meshterms
            return feature.replace("[mh]", "[mh:noexp]")
        # Helper: get PubMed-safe name for a feature, expand terms
        terms = term_expansions.get(feature, [feature])
        # TODO remove [tiab] title abstract. but needed since NOT diagnsis matches mesh term diagnsos otherwise
        terms = [f + "[tiab]" for f in terms]
        return " OR ".join(terms)

    def pubmed_query(self, term_expansions: dict = None):
        """
        Converts a GreedyORDecisionTree into a PubMed DNF boolean query.
        - Left child = NOT feature (omitted if NOT at root of clause)
        - Right child = feature present
        - Skips paths that lead only to negative class
        - Expands each term using term_expansions dict
        """
        if self._tree is None:
            raise ValueError("Tree not trained. Call fit() first.")

        # Traverse recursively
        added_ORs = 0
        synonym_ORs = 0
        def recurse(node, path_terms=None):
            nonlocal added_ORs, synonym_ORs
            if path_terms is None:
                path_terms = []

            # Early stop if subtree pure
            same_class = self._all_same_class(node)
            if same_class is not None:
                if same_class == 1:  # positive leaf
                    return [path_terms]  # return current path
                else:
                    return []  # skip paths leading to negative class only

            if node["type"] == "leaf":
                if self._node_class(node):
                    return [path_terms]
                else:
                    return []

            clauses = []

            addition = " OR ".join(self._expand_term(term_expansions, f) for f in node["features"])
            added_ORs += len(node["features"]) - 1
            synonym_ORs += addition.count("OR") - added_ORs
            # Right child = feature present
            left_addition = addition
            if "OR" in left_addition:
                left_addition = "(" + left_addition + ")"
            left_terms = path_terms + [left_addition]
            clauses.extend(recurse(node["left"], left_terms))

            # Left child = NOT of feature
            right_addition = addition
            if "OR" in right_addition:
                right_addition = "NOT (" + right_addition + ")"
            else:
                right_addition = "NOT " + right_addition
            right_terms = path_terms + [right_addition]
            clauses.extend(recurse(node["right"], right_terms))

            return clauses

        # Collect all DNF clauses
        dnf_clauses = recurse(self._tree)

        # Combine with OR for PubMed query
        query_clauses = []
        path_lens = []
        for clause in dnf_clauses:
            # Split into positive and negative terms
            pos_terms = [t for t in clause if not t.startswith("NOT ")]
            not_terms = [t for t in clause if t.startswith("NOT ")]

            # Skip clauses that are all NOT
            if len(pos_terms) == 0:
                continue

            # Reorder: positive terms first, NOT terms last
            query_clauses.append(f"({' AND '.join(pos_terms)} {' '.join(not_terms)})")
            path_lens.append(len(pos_terms) + len(not_terms))
        # Combine all clauses with OR
        query = " OR ".join(query_clauses)
        avg_path_len = sum(path_lens) / len(path_lens) if path_lens else 0
        query_size = {
            "paths": len(query_clauses), 
            "avg_path_len": avg_path_len, 
            "ANDs": query.count('AND'),
            "NOTs": query.count('NOT'),
            "added_ORs": added_ORs,
            "synonym_ORs": synonym_ORs,
            "ORs": query.count('OR'),
        }
        return query, query_size
    
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
            not (d["cats"] or d["dogs"] or d["mice[mh]"]) and
            (d["house"] or d["wohnung"])
            and (d["bowl"] or d["box"])  
        )

    variables = ["cats", "dogs", "mice[mh]", "house", "wohnung", "bowl", "box"]
    texts, labels = generate_texts_from_boolean(
        func=f,
        variables=variables,
        error=0.1,
        completeness=0.9,
        seed=42,
        doc_count=10_000,
        word_pool_size=400,
        average_doc_length=30,
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

from line_profiler import LineProfiler

if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(GreedyORDecisionTree.fit)

    lp.run("main()")
    lp.print_stats()


# TODO balanced min sample split (weighted)
