import numpy as np
from typing import List, Tuple
from joblib import Parallel, delayed
from scipy.sparse import issparse
from numbers import Integral, Real
from collections import defaultdict
from app.config.config import DEBUG
from app.tree_learning.disjunctive_dt import (
    GreedyORDecisionTree,
    compute_class_weight,
    compute_sample_weight,
)
from app.tree_learning.query_generation import (
    rules_to_pubmed_query,
    extract_and_vectorize_rules,
    select_rules_via_ga,
    query_size,
    query_cost,
)
from app.helper.helper import biased_random


class RandomForest:
    """
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split (should be named min_samples_leaf) : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If random then _biased_random_max_features is callled (sqrt(n) <= random <= n_features)
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `"sqrt"`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : TODO int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : TODO (other formual than mine -> scales with N_t??) float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max(round(n_samples * max_samples), 1)` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

        .. versionadded:: 0.22

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : Ensemble of extremely randomized
        tree classifiers.
    sklearn.ensemble.HistGradientBoostingClassifier : A Histogram-based Gradient
        Boosting Classification Tree, very fast for big datasets (n_samples >=
        10_000).

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.
    """


class RandomForest:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        randomize_max_feature=None,
        min_impurity_decrease_range_start=0.01,
        min_impurity_decrease_range_end=0.01,
        randomize_min_impurity_decrease_range=None,
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        verbose=True,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        top_k_or_candidates=None,
        prefer_pos_splits=None,
        max_or_features=100,
        top_k=None,
        rank_weight=None,
    ):
        """
        Initialize a RandomForest instance.

        Parameters match the class docstring and are stored as attributes.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.randomize_max_feature = randomize_max_feature
        self.min_impurity_decrease_range_start = min_impurity_decrease_range_start
        self.min_impurity_decrease_range_end = min_impurity_decrease_range_end
        self.randomize_min_impurity_decrease_range = (
            randomize_min_impurity_decrease_range
        )
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.top_k_or_candidates = top_k_or_candidates
        self.max_or_features = max_or_features
        
        # Will be populated after fit
        self.estimators_ = []
        self._n_samples_bootstrap = None
        self.n_outputs_ = None
        self.prefer_pos_splits = prefer_pos_splits
        self.top_k = top_k
        self.rank_weight = rank_weight
        

    def fit(self, X, y, sample_weight=None, feature_names=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        # X, y = validate_data(
        #     self,
        #     X,
        #     y,
        #     multi_output=True,
        #     accept_sparse="csc",
        #     dtype=DTYPE,
        #     ensure_all_finite=False,
        # )

        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        self.class_weight = compute_class_weight(self.class_weight, X, y)
        expanded_class_weight = compute_sample_weight(self.class_weight, np.copy(y))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        # random_state = np.random.RandomState(self.random_state) # Turn seed into a np.random.RandomState instance

        self.estimators_ = []
        trees = []
        for i in range(self.n_estimators):
            tree_max_features = self.max_features
            tree_randomize_max_feature = self.randomize_max_feature
            tree_min_impurity_decrease_range_start = self.min_impurity_decrease_range_start
            tree_min_impurity_decrease_range_end = self.min_impurity_decrease_range_end
            if i == 0:
                if (
                    self.randomize_max_feature
                ):  # first random tree always gets all features
                    tree_max_features = None
                    tree_randomize_max_feature = False
            else:
                if self.randomize_min_impurity_decrease_range:
                    tree_min_impurity_decrease_range_start = biased_random(
                            low=self.min_impurity_decrease_range_start,
                            high=1.0,
                            exponent=self.randomize_min_impurity_decrease_range,
                        )
                    tree_min_impurity_decrease_range_end =  biased_random(
                        low=self.min_impurity_decrease_range_end,
                        high=1.0,
                        exponent=self.randomize_min_impurity_decrease_range,
                    )
            tree_config = {
                "max_depth": self.max_depth,
                "min_impurity_decrease_range_start": tree_min_impurity_decrease_range_start,
                "min_impurity_decrease_range_end": tree_min_impurity_decrease_range_end,
                "top_k_or_candidates": self.top_k_or_candidates,
                "verbose": self.verbose
                and not self.n_jobs,  # has to be set to false because of tqdm breaking Parallel
                "min_samples_split": self.min_samples_split,
                "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
                "class_weight": self.class_weight,
                "max_features": tree_max_features,
                "randomize_max_feature": tree_randomize_max_feature,
                "random_state": self.random_state,
                "prefer_pos_splits": self.prefer_pos_splits,
                "max_or_features": self.max_or_features
            }
            trees.append(GreedyORDecisionTree(**tree_config))

        trees = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="processes",
        )(
            delayed(_parallel_build_trees)(
                t,
                self.bootstrap,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                verbose=self.verbose,
                n_samples_bootstrap=n_samples_bootstrap,
                feature_names=feature_names,
            )
            for i, t in enumerate(trees)
        )

        # Collect newly grown trees
        trees = [t for t in trees if t is not None]
        self.estimators_.extend(trees)
        
        return self

    def get_tree_paths(self):
        """
        Extract rules from all trees.

        Returns
        -------
        rules : List[Rule]
        rule_tree_map : dict: Rule -> int
            tree index for each rule
        """
        # all_rules = []
        rule_tree_map = defaultdict(set)

        for t_idx, tree in enumerate(self.estimators_):
            tree_rules = tree.get_tree_paths()
            if DEBUG:
                print(tree.pretty_print(verbose=True))
                print("tree_rules", tree_rules)
                print("class_weight", tree.class_weight)
            # all_rules.extend(tree_rules)
            for r in tree_rules:
                rule_tree_map[r].add(t_idx)
            # rule_tree_map.extend([t_idx] * len(tree_rules))
        assert rule_tree_map
        return rule_tree_map

    def pubmed_query(
        self,
        feature_names,
        pruning_thresholds: dict,
        term_expansions: dict = None,
        X=None,
        labels=None,
        min_tree_occ=0.05,
        min_rule_occ=0.05,
        cost_factor=0.002,
        min_rule_precision=0.01,
        cover_beta=2.0,
        pruning_beta:float = 0.1,
        mh_noexp = False, 
        tiab = False,
    ):
        if X is None or labels is None:
            all_rules, rule_tree_map = self.get_tree_paths()
            return rules_to_pubmed_query(
                rules=all_rules,
                feature_names=feature_names,
                term_expansions=term_expansions,
            )
        else:
            vec_result = extract_and_vectorize_rules(
                forest=self,
                X=X,
                y=np.asarray(labels),
                min_tree_occ=min_tree_occ,
                min_rule_occ=min_rule_occ,
                min_rule_precision=min_rule_precision,
                verbose=self.verbose,
                feature_names=feature_names,
                pruning_thresholds=pruning_thresholds,
                pruning_beta=pruning_beta,
            )
            rules = vec_result["rules"]
            print("NUMBER OF RULES", len(rules))
            cover_score = None
            if len(rules) > 1:
                # kept_variables = vec_result["kept_variables"]
                coverage = vec_result["coverage"]
                rule_costs = np.array([query_cost(query_size([r])) for r in rules])
                selection_result = select_rules_via_ga(
                    coverage=coverage,
                    y=np.array(labels),
                    rule_costs=rule_costs,
                    cost_factor=cost_factor,
                    initial_solutions=vec_result["initial_solutions_binary"],
                    beta=cover_beta,
                )
                rules = [rules[i] for i in selection_result["selected_rule_indices"]]
                cover_score = selection_result["objective"]
            pubmed_query = rules_to_pubmed_query(
                rules=rules,
                feature_names=feature_names,
                term_expansions=term_expansions,
                tiab=tiab,
                mh_noexp=mh_noexp,
            )

            # REMOVE
            if DEBUG:
                print("INITIAL SOLUTIONS")
                for tree, rs in vec_result["initial_solutions"].items():
                    print(
                        tree,
                        rules_to_pubmed_query(
                            rules=rs,
                            feature_names=feature_names,
                            term_expansions=term_expansions,
                        )[0].replace("[tiab]", ""),
                    )

            return pubmed_query, rules, cover_score

    def _find_optimal_threshold(self, **args):
        for tree in self.estimators_:
            tree._find_optimal_threshold(**args)


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
            the interval `(0.0, 1.0]`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.

    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, Integral):
        if max_samples > n_samples:
            msg = "`max_samples` must be <= n_samples={} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, Real):
        return max(round(n_samples * max_samples), 1)


def _parallel_build_trees(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    n_samples_bootstrap=None,
    feature_names=None,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = tree.random_state.randint(
            0, n_samples, n_samples_bootstrap, dtype=np.int32
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts
    else:
        curr_sample_weight = sample_weight

    # --- Early check: must have at least one sample of each class with weight > 0 ---
    has_pos = np.any((y == 1) & (curr_sample_weight > 0))
    has_neg = np.any((y == 0) & (curr_sample_weight > 0))

    if not (has_pos and has_neg):
        if verbose:
            print(
                f"Skipping tree {tree_idx + 1}: only one class present after weighting"
            )
        return None  # Early exit

    tree.fit(
        X,
        y,
        sample_weight=curr_sample_weight,
        feature_names=feature_names,
    )

    return tree
