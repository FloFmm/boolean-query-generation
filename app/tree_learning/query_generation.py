from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Literal, Dict, Optional, FrozenSet, Set, Union
import numpy as np
from app.helper.helper import f_beta
from app.config.config import DEBUG  # TODO remove

# Rule = List[Tuple[List[int], List[str], bool]]
Rule = FrozenSet[Tuple[FrozenSet[int], bool]]


def rule_size(rule):
    return sum(len(t) for t in rule)


def compute_variable_frequencies(rule_tree_map: dict[Rule, set[int]], n_trees):
    """
    Parameters
    ----------
    rule_tree_map : dict[Rule, set[int]]

    Returns
    -------
    tree_freq : dict[int, float]
    rule_freq : dict[int, float]
    """
    trees_per_var = defaultdict(set)
    rules_per_var = defaultdict(int)

    for rule, trees in rule_tree_map.items():
        vars_in_rule = set()
        for feature_indices, is_pos in rule:
            vars_in_rule.update(feature_indices)

        for v in vars_in_rule:
            trees_per_var[v].update(trees)
            rules_per_var[v] += 1

    tree_freq = {v: len(tset) / n_trees for v, tset in trees_per_var.items()}
    rule_freq = {v: cnt / len(rule_tree_map) for v, cnt in rules_per_var.items()}
    return tree_freq, rule_freq


def prune_rare_features(
    rule_tree_map: dict[Rule, set[int]],
    tree_freq,
    rule_freq,
    min_tree_occ=0.05,
    min_rule_occ=0.05,
    feature_names=None,
):
    """
    Removes variables not meeting frequency thresholds.
    """
    kept_vars = {
        v
        for v in tree_freq
        if tree_freq[v] >= min_tree_occ and rule_freq.get(v, 0) >= min_rule_occ
    }
    pruned_rule_tree_map: dict[Rule, set[int]] = defaultdict(set)

    for rule, tree_indices in rule_tree_map.items():
        new_terms = []
        for feat_inds, is_pos in rule:
            kept_feat_ind = feat_inds & kept_vars
            if DEBUG:
                for f in feat_inds - kept_vars:
                    print(feature_names[f])
            if not kept_feat_ind:
                break

            new_terms.append((frozenset(kept_feat_ind), is_pos))

        if new_terms:
            new_rule = frozenset(new_terms)
            if any(
                t[-1] for t in new_rule
            ):  # Filter out rules that have only negative terms
                pruned_rule_tree_map[new_rule].update(tree_indices)

    return pruned_rule_tree_map, kept_vars


def compute_rule_coverage(X, rules: Set[Rule], verbose: bool = False):
    """
    Parameters
    ----------
    X : csr_matrix (binary)
    rules : Set(Rule)

    Returns
    -------
    coverage : np.ndarray, shape (n_rules, n_samples)
    """
    n_samples = X.shape[0]
    coverage = np.zeros((len(rules), n_samples), dtype=np.uint8)

    rule_iter = rules
    if verbose:
        rule_iter = tqdm(
            rules,
            desc="Computing rule coverage",
            total=len(rules),
        )
    for i, rule in enumerate(rule_iter):
        mask = np.ones(n_samples, dtype=bool)

        for feature_indices, is_pos in rule:
            cols = list(feature_indices)
            if is_pos:
                # OR present
                mask &= X[:, cols].getnnz(axis=1) > 0
            else:
                # OR absent
                mask &= X[:, cols].getnnz(axis=1) == 0

            if not mask.any():
                break

        coverage[i, mask] = 1
    if verbose:
        rule_iter.close()
    return coverage


def extract_and_vectorize_rules(
    forest,
    X,
    y,
    pruning_thresholds: dict,
    min_tree_occ=0.05,
    min_rule_occ=0.02,
    min_rule_precision=0.01,
    verbose: bool = False,
    feature_names=None,
    pruning_beta: float = 0.1,
) -> List[Rule]:
    """
    Full pipeline for AND-of-OR rules.
    """
    rule_tree_map = forest.get_tree_paths()

    tree_freq, rule_freq = compute_variable_frequencies(
        rule_tree_map,
        n_trees=len(forest.estimators_),
    )
    assert rule_tree_map
    rule_tree_map, kept_vars = prune_rare_features(
        rule_tree_map,
        tree_freq,
        rule_freq,
        min_tree_occ,
        min_rule_occ,
        feature_names=feature_names,
    )
    # pruned_rules = deduplicate_rules(pruned_rules) # deduplicate twice for speed?
    new_rule_tree_map: dict[Rule, set[int]] = defaultdict(set)
    histories: set[tuple[Rule]] = set()
    rule_stats: dict[Rule, dict] = {}  # changed in place by prune_rule_greedy
    initial_solutions: dict[int, set[Rule]] = defaultdict(set)

    assert rule_tree_map
    rule_tree_map_iter = rule_tree_map.items()
    if verbose:
        rule_tree_map_iter = tqdm(
            rule_tree_map_iter,  # reuse the same iterator
            desc="Pruning rule greedily",
            total=len(rule_tree_map),
        )

    X = (
        X.tocsc()
    )  # makes compute_rule_coverage much faster (needed for greeddy pruning)
    for rule, tree_indices in rule_tree_map_iter:
        if not rule:
            continue
        # history = [rule]
        # rule_stats[rule] = {"precision" : 0.1}
        history = prune_rule_greedy(
            X,
            y,
            rule,
            histories=histories,
            rule_stats=rule_stats,
            feature_names=feature_names,
            pruning_thresholds=pruning_thresholds,
            beta=pruning_beta,
        )
        # history can be empty
        if not history:
            continue
        # assert history, f"rule: {rules_to_pubmed_query([rule], feature_names=feature_names)}\n {forest.estimators_[next(iter(tree_indices))].pretty_print(feature_names=feature_names, verbose=True)}"

        try:
            x = set(history)
        except:
            print("rule")
            print(rule)
            print()
            print("history")
            print(history)
            print()
            print("histories")
            print(histories)
            # somtimes happens (TODO fix this bug) (probably fixed)
            assert False
        history = tuple(
            r for r in history if rule_stats[r]["precision"] >= min_rule_precision
        )
        if not history:
            continue

        biggest_rule = history[0]
        for tree_inx in tree_indices:
            initial_solutions[tree_inx].add(biggest_rule)

        histories.add(tuple(history))
        for r in history:
            new_rule_tree_map[r].update(tree_indices)

    assert new_rule_tree_map

    if verbose:
        rule_tree_map_iter.close()
    rule_tree_map = new_rule_tree_map
    pruned_rules = list(rule_tree_map.keys())
    assert len(pruned_rules) > 0
    rule_to_idx = {r: i for i, r in enumerate(pruned_rules)}
    initial_solutions_binary = [
        np.isin(
            np.arange(len(pruned_rules)),
            [rule_to_idx[r] for r in initial_solutions[tree_inx]],
        ).astype(int)
        for tree_inx in sorted(initial_solutions)
    ]
    initial_solutions_list = [
        [rule_to_idx[r] for r in initial_solutions[tree_inx]]
        for tree_inx in sorted(initial_solutions)
    ]

    # pruned_rules = deduplicate_rules(pruned_rules) # deduplicate twice for speed?
    coverage = compute_rule_coverage(X, pruned_rules, verbose=forest.verbose)

    return {
        "rules": pruned_rules,
        "kept_variables": kept_vars,
        "coverage": coverage,
        "initial_solutions_binary": initial_solutions_binary,
        "initial_solutions_list": initial_solutions_list,
        "initial_solutions": initial_solutions,
    }


def expand_term(
    term_expansions, feature, is_positive: bool, mh_noexp: bool, tiab: bool
):
    if feature.endswith("[mh]"):  # mesh terms
        if mh_noexp:
            return feature.replace("[mh]", "[mh:noexp]")
        else:
            return feature
    if term_expansions is None:
        terms = [feature]
    else:
        terms = term_expansions.get(feature, [feature])
    terms = [f'"{w}"' if " " in w else w for w in terms]
    if tiab or not is_positive:  # negative terms always get [tiab]
        terms = [f + "[tiab]" for f in terms]
    return " SYNONYM_OR ".join(terms)


def literal_to_pubmed(
    features, is_positive, term_expansions, tiab: bool = False, mh_noexp: bool = False
):
    clause = " ADDED_OR ".join(
        expand_term(
            term_expansions, f, is_positive=is_positive, tiab=tiab, mh_noexp=mh_noexp
        )
        for f in features
    )

    if "OR" in clause:
        clause = f"({clause})"

    if not is_positive:
        clause = f"NOT {clause}"

    return clause


def rules_to_pubmed_query(
    rules: List[Rule],
    feature_names,
    term_expansions: dict = None,
    tiab: bool = False,
    mh_noexp: bool = False,
):
    """
    Converts extracted AND-of-OR rules into a PubMed DNF boolean query.
    """
    query_clauses = []
    path_lens = []
    for rule in rules:
        if not any(is_pos for _, is_pos in rule):
            # Skip clauses that are all NOT
            continue
        pos_literals = [
            literal_to_pubmed(
                [feature_names[i] for i in features_indices],
                is_pos,
                term_expansions,
                mh_noexp=mh_noexp,
                tiab=tiab,
            )
            for features_indices, is_pos in rule
            if is_pos
        ]
        neg_literals = [
            literal_to_pubmed(
                [feature_names[i] for i in features_indices],
                is_pos,
                term_expansions,
                mh_noexp=mh_noexp,
                tiab=tiab,
            )
            for features_indices, is_pos in rule
            if not is_pos
        ]

        clause = " AND ".join(pos_literals)
        if neg_literals:
            clause += " " + " ".join(neg_literals)

        if len(rule) > 1:
            clause = f"({clause})"

        query_clauses.append(clause)
        path_lens.append(len(rule))

    query = " OR ".join(query_clauses)

    query = query.replace("ADDED_OR", "OR").replace("SYNONYM_OR", "OR")
    assert query

    return query, query_size(rules)


def query_size(rules):
    query_size = {
        "paths": len(rules),
        "avg_path_len": sum([len(rule) for rule in rules]) / len(rules)
        if len(rules)
        else 0,
        "ANDs": sum([len([term for term in rule if term[-1]]) for rule in rules]) - 1,
        "NOTs": sum([len([term for term in rule if not term[-1]]) for rule in rules]),
        "added_ORs": sum([sum([len(term[0]) for term in rule]) for rule in rules]),
        "synonym_ORs": -1,
        "ORs": len(rules),
    }
    return query_size


def query_cost(query_size, weights=None):
    """
    Compute a cost for a query based on its complexity.

    Args:
        query_size: dict with keys ["paths", "avg_path_len", "ANDs", "NOTs", "added_ORs", "synonym_ORs", "ORs"]
        weights: optional dict to weight different components

    Returns:
        float: total cost
    """
    if weights is None:
        weights = {
            "paths": 2.0,
            # "avg_path_len": 0.0,
            "ANDs": 1.0,
            "NOTs": 1.5,
            "added_ORs": 0.5,
            # "synonym_ORs": 0.0,
            # "ORs": 0.0,
        }

    cost = 0.0
    for key, weight in weights.items():
        cost += weight * query_size.get(key, 0)

    return cost


def coverage_of_rule(X, rule):
    """Return boolean mask of samples covered by a single rule using compute_rule_coverage."""
    cov = compute_rule_coverage(X, {rule})  # shape (1, n_samples)
    return cov[0].astype(bool)


def metrics_from_mask(mask: np.ndarray, y: np.ndarray):
    """Return (precision, recall, tp, returned, pos_count)"""
    returned = int(mask.sum())
    pos_mask = y == 1
    tp = int((mask & pos_mask).sum())
    pos_count = int(pos_mask.sum())
    precision = tp / returned if returned > 0 else 0.0
    recall = tp / pos_count if pos_count > 0 else 0.0
    return precision, recall, tp, returned, pos_count


# def deep_copy_rule(rule: Rule):
#     # rule is list of tuples (feature_indices, feature_names, is_pos)
#     new_rule = []
#     for feat_inds, is_pos in rule:
#         # ensure we copy lists, sets etc.
#         new_rule.append(
#             (
#                 list(feat_inds),
#                 list(feat_names) if feat_names is not None else None,
#                 bool(is_pos),
#             )
#         )
#     return new_rule


def generate_one_step_rule_variations(
    rule: Rule, mode: Literal["or", "and"]
) -> Set[Rule]:
    """
    Generate all rules obtainable by a single pruning step.

    Parameters
    ----------
    rule : Rule
    mode : "or" | "and"
        - "or"  : remove ONE feature from a disjunction
        - "and" : remove ONE entire term

    Returns
    -------
    pruned_rules : Set[Rule]
        All 1-step pruned variants of the rule
    removal_in_negated_term : list[bool]
    """
    pruned_rules: list[Rule] = []
    removal_in_negated_term: list[bool] = []
    removed_features: list[set[int]] = []
    if mode == "or":
        for term in rule:
            feat_inds, is_pos = term
            if len(feat_inds) <= 1:
                continue
            for f in feat_inds:
                removed_features.append({f})
                new_inds = feat_inds - {f}
                new_term = (new_inds, is_pos)
                new_rule = (rule - {term}) | {new_term}
                pruned_rules.append(new_rule)
                removal_in_negated_term.append(not term[-1])
    elif mode == "and":
        if len(rule) > 1:
            for term in rule:
                pruned_rules.append(rule - {term})
                removed_features.append(set(term[0]))
                removal_in_negated_term.append(not term[-1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Filter out rules that have only negative terms
    pruned_rules = [r for r in pruned_rules if any(t[-1] for t in r)]

    return pruned_rules, removal_in_negated_term, removed_features


def select_best_metric(
    metrics: List[Dict],
    threshold: dict,
) -> Optional[int]:
    """
    Select index of the best metric from a list of metric dicts.

    Parameters
    ----------
    metrics : list of dict
        Each dict must contain keys 'f' and the acceptance_metric
    acceptance_metric : str
        Name of the metric to use for acceptance and removal thresholds
    acceptance_threshold : float
        Minimum value of acceptance_metric to be considered
    removal_threshold : float
        If a metric exceeds this, it is preferred even if f is lower

    Returns
    -------
    index : int or None
        Index of the selected metric in the list, or None if none pass acceptance
    """

    best_idx = None
    best_f = -float("inf")

    remove_old = False
    # First, filter only metrics passing acceptance_threshold
    for i, m in enumerate(metrics):
        removal_mode = m["mode"]
        in_neg_term = m["removal_in_negated_term"]
        acceptance_metric = threshold[removal_mode][in_neg_term]["acceptance_metric"]
        value = m[acceptance_metric]
        acceptance_threshold = threshold[removal_mode][in_neg_term][
            "acceptance_threshold"
        ]
        removal_threshold = threshold[removal_mode][in_neg_term]["removal_threshold"]

        if value < acceptance_threshold:
            continue

        if value < removal_threshold:
            if remove_old:
                continue

        if value >= removal_threshold and not remove_old:
            # first remove_old
            best_f = m["f"]
            best_idx = i
            remove_old = True
            continue

        # if value >= removal_threshold:
        #     print("in_neg_term", in_neg_term)
        #     print("acceptance_metric", acceptance_metric)
        #     print("value", value)
        #     print("acceptance_threshold", acceptance_threshold)
        #     print("removal_threshold", removal_threshold)

        if m["f"] > best_f:
            best_f = m["f"]
            best_idx = i
    return best_idx, remove_old


def prune_rule_greedy(
    X,
    y,
    rule: Rule,
    histories: set[list[Rule]],
    rule_stats: dict[Rule, dict],  # changed in place
    pruning_thresholds: dict,
    beta: float = 0.1,
    feature_names=None,
):
    """
    Greedy prune a single rule in two stages:
      1) remove OR features (inside disjunctions) - only disjunctions with >1 features
      2) remove AND terms (entire tuple/term)
    Selection criterion: maximize F_beta (default beta=0.1).
    Acceptance / remove-old thresholds follow your spec:
      - OR-feature removal: accept if tp_gain > -0.1 ; remove old if tp_gain > -0.01
      - AND-term removal: accept if precision_gain > -0.1 ; remove old if precision_gain > -0.01

    Returns:
      history: list of remembered rules (each rule is a deep copy)
    """
    # Prepare
    current_rule: Rule = rule
    history: List[Rule] = [current_rule]

    # baseline metrics
    mask = coverage_of_rule(X, current_rule)
    p_old, r_old, tp_old, ret_old, pos_count = metrics_from_mask(mask, y)
    if tp_old == 0:
        return []

    f_old = f_beta(p_old, r_old, beta)
    best_metric = {
        "f": f_old,
        "precision": p_old,
        "recall": r_old,
        "tp": tp_old,
        "returned": ret_old,
    }
    # old_rule = None
    if DEBUG:
        print("START")
    while True:
        if DEBUG:
            print()
            print(
                rules_to_pubmed_query(
                    [current_rule],
                    feature_names=feature_names,
                    tiab=True,
                    mh_noexp=True,
                )[0]
            )
        if current_rule not in rule_stats:
            rule_stats[current_rule] = best_metric
        else:
            for h in histories:
                if len(current_rule) > 1 and current_rule in h:
                    # smaller_rules = {
                    #     r
                    #     for r in h
                    #     if rule_size(r) <= rule_size(current_rule)
                    #     if r != current_rule
                    # }
                    # sorted_rules = sorted(smaller_rules, key=rule_size, reverse=True)
                    rule_idx = h.index(current_rule) + 1
                    history += h[rule_idx:]
                    return history

        # best_candidate = None
        # best_metrics = None
        # best_f = -np.inf
        metrics = []
        # iterate over terms
        and_cand_rules, and_removal_in_negated_term, and_removed_features = (
            generate_one_step_rule_variations(
                rule=current_rule,
                mode="and",
            )
        )
        or_cand_rules, or_removal_in_negated_term, or_removed_features = (
            generate_one_step_rule_variations(
                rule=current_rule,
                mode="or",
            )
        )
        candidate_rules = or_cand_rules + and_cand_rules
        removal_in_negated_term = (
            or_removal_in_negated_term + and_removal_in_negated_term
        )
        removed_features = or_removed_features + and_removed_features

        if not candidate_rules:
            break

        for i, (cand_rule, removal_in_neg, rem_f) in enumerate(
            zip(candidate_rules, removal_in_negated_term, removed_features)
        ):
            mode = "or" if i < len(or_cand_rules) else "and"

            # compute metrics
            mask_c = coverage_of_rule(X, cand_rule)
            p_new, r_new, tp_new, ret_new, _ = metrics_from_mask(mask_c, y)
            f_new = f_beta(p_new, r_new, beta)
            tp_gain = (tp_new - tp_old) / tp_old
            precision_gain = (p_new - p_old) / p_old

            metrics.append(
                {
                    "f": f_new,
                    "precision": p_new,
                    "recall": r_new,
                    "tp": tp_new,
                    "returned": ret_new,
                    "tp_gain": tp_gain,
                    "precision_gain": precision_gain,
                    "removal_in_negated_term": removal_in_neg,
                    "mode": mode,
                    "removed_features": rem_f,
                }
            )

        best_candidate_index, remove_old = select_best_metric(
            metrics,
            pruning_thresholds,
        )
        if best_candidate_index is None:
            break
        best_rule = candidate_rules[best_candidate_index]

        best_metric = metrics[best_candidate_index]

        # accept the new rule
        current_rule = best_rule

        # update old metrics
        p_old, r_old = best_metric["precision"], best_metric["recall"]

        if DEBUG:
            print("====MODE====", best_metric["mode"])
            print(
                "====REMOVED FEATURES====",
                [feature_names[f] for f in best_metric["removed_features"]],
            )
            print(best_metric)
        if remove_old:
            if DEBUG:
                print("======remove old==========")
            # remove the previous rule from memory (i.e., drop the version before the last)
            # We keep only the most recent one in history when removal condition met.
            if len(history) >= 1:
                # drop the second-to-last (the previous state) because "new rule is so good"
                history.pop(-1)

        assert current_rule is not None
        assert len(current_rule) > 0
        assert not any([not term[0] for term in current_rule])

        history.append(current_rule)
        # history_meta.append(best_metric)
        tp_old = best_metric["tp"]
        p_old = best_metric["precision"]
        r_old = best_metric["recall"]
        ret_old = best_metric["returned"]
        f_old = f_beta(p_old, r_old, beta)
    return history
