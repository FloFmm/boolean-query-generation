from collections import defaultdict
from typing import List, Tuple, Literal, Dict, Optional, FrozenSet, Set
import numpy as np
import copy
import scipy.sparse as sp
from deap import base, creator, tools, algorithms
from app.helper.helper import f_beta

# Rule = List[Tuple[List[int], List[str], bool]]
Rule = FrozenSet[Tuple[FrozenSet[int], bool]]


def compute_variable_frequencies(rules: Set[Rule], rule_tree_map, n_trees):
    """
    Parameters
    ----------
    rules : Set[Rule]

    Returns
    -------
    tree_freq : dict[int, float]
    rule_freq : dict[int, float]
    """
    trees_per_var = defaultdict(set)
    rules_per_var = defaultdict(int)

    for r_idx, rule in enumerate(rules):
        vars_in_rule = set()
        for feature_indices, is_pos in rule:
            vars_in_rule.update(feature_indices)

        for v in vars_in_rule:
            trees_per_var[v].add(rule_tree_map[r_idx])
            rules_per_var[v] += 1

    tree_freq = {v: len(tset) / n_trees for v, tset in trees_per_var.items()}
    rule_freq = {v: cnt / len(rules) for v, cnt in rules_per_var.items()}

    return tree_freq, rule_freq


def prune_rules(
    rules: Set[Rule],
    tree_freq,
    rule_freq,
    min_tree_occ=0.05,
    min_rule_occ=0.05,
):
    """
    Removes variables not meeting frequency thresholds.
    """
    kept_vars = {
        v
        for v in tree_freq
        if tree_freq[v] >= min_tree_occ and rule_freq.get(v, 0) >= min_rule_occ
    }
    pruned_rules: set[Rule] = set()

    for rule in rules:
        new_terms = []
        for feat_inds, is_pos in rule:
            kept_feat_ind = feat_inds & kept_vars

            if not kept_feat_ind:
                break

            new_terms.append((frozenset(kept_feat_ind), is_pos))

        if new_terms:
            pruned_rules.add(frozenset(new_terms))

    # Filter out rules that have only negative terms
    pruned_rules = {r for r in pruned_rules if any(t[-1] for t in r)}

    return pruned_rules, kept_vars


# def deduplicate_rules(rules):
#     """
#     Removes duplicate rules. Order of triples in a rule and order of features/indices does not matter.

#     Args:
#         rules: list of rules, each rule is a list of triples
#                (feature_indices: list[int], features: list[str], is_pos: bool)

#     Returns:
#         List of unique rules.
#     """
#     seen = set()
#     unique_rules = []

#     for rule in rules:
#         canonical_rule = []

#         for feature_indices, features, is_pos in rule:
#             # Sort indices and features to make order irrelevant
#             canonical_triple = (frozenset(feature_indices), frozenset(features), is_pos)
#             canonical_rule.append(canonical_triple)

#         # The order of triples in the rule doesn't matter
#         canonical_rule_set = frozenset(canonical_rule)

#         if canonical_rule_set not in seen:
#             seen.add(canonical_rule_set)
#             # Recover original rule structure
#             unique_rules.append(rule)

#     return unique_rules


def compute_rule_coverage(X, rules: Set[Rule]):
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
    for i, rule in enumerate(rules):
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

    return coverage


def extract_and_vectorize_rules(
    forest,
    X,
    y,
    min_tree_occ=0.05,
    min_rule_occ=0.02,
):
    """
    Full pipeline for AND-of-OR rules.
    """
    rules, rule_tree_map = forest.get_tree_paths()
    tree_freq, rule_freq = compute_variable_frequencies(
        rules,
        rule_tree_map,
        n_trees=len(forest.estimators_),
    )

    pruned_rules, kept_vars = prune_rules(
        rules,
        tree_freq,
        rule_freq,
        min_tree_occ,
        min_rule_occ,
    )
    # pruned_rules = deduplicate_rules(pruned_rules) # deduplicate twice for speed?
    new_pruned_rules: set[Rule] = set()
    for rule in pruned_rules:
        # print(rules_to_pubmed_query(
        #         rules=[rule])[0].replace("[tiab]", ""))
        current_rule, history, history_meta = prune_rule_greedy(X, y, rule)
        # print()
        # print("after")
        # for i, r in enumerate(history):
        #     print(i, rules_to_pubmed_query(
        #             rules=[r])[0].replace("[tiab]", ""))
        new_pruned_rules.update(history)
    pruned_rules = list(new_pruned_rules)
    # pruned_rules = deduplicate_rules(pruned_rules) # deduplicate twice for speed?
    coverage = compute_rule_coverage(X, pruned_rules)

    return {
        "rules": pruned_rules,
        "kept_variables": kept_vars,
        "coverage": coverage,
    }


def expand_term(term_expansions, feature):
    if feature.endswith("[mh]"):
        # mesh terms
        # TODO maybe only add :noexp on negative meshterms
        return feature.replace("[mh]", "[mh:noexp]")
    # Helper: get PubMed-safe name for a feature, expand terms
    if term_expansions is None:
        terms = [feature]
    else:
        terms = term_expansions.get(feature, [feature])
    # TODO remove [tiab] title abstract. but needed since NOT diagnsis matches mesh term diagnsos otherwise
    terms = [f + "[tiab]" for f in terms]
    return " SYNONYM_OR ".join(terms)


def literal_to_pubmed(features, is_positive, term_expansions):
    clause = " ADDED_OR ".join(expand_term(term_expansions, f) for f in features)

    if "OR" in clause:
        clause = f"({clause})"

    if not is_positive:
        clause = f"NOT {clause}"

    return clause


def rules_to_pubmed_query(
    rules: List[Rule],
    feature_names,
    term_expansions: dict = None,
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
                [feature_names[i] for i in features_indices], is_pos, term_expansions
            )
            for features_indices, is_pos in rule
            if is_pos
        ]
        neg_literals = [
            literal_to_pubmed(
                [feature_names[i] for i in features_indices], is_pos, term_expansions
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

    query_size = {
        "paths": len(rules),
        "avg_path_len": sum(path_lens) / len(rules) if len(rules) else 0,
        "ANDs": query.count("AND"),
        "NOTs": query.count("NOT"),
        "added_ORs": query.count("ADDED_OR"),
        "synonym_ORs": query.count("SYNONYM_OR"),
        "ORs": query.count("OR"),
    }
    query = query.replace("ADDED_OR", "OR").replace("SYNONYM_OR", "OR")
    assert query

    return query, query_size


def query_size(rules):
    print(rules)
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
            "added_ORs": 0.25,
            # "synonym_ORs": 0.0,
            # "ORs": 0.0,
        }

    cost = 0.0
    for key, weight in weights.items():
        cost += weight * query_size.get(key, 0)

    return cost


def select_rules_via_ga(
    coverage: np.ndarray | sp.csr_matrix,
    y: np.ndarray,
    rule_costs: np.ndarray | None = None,
    initial_solutions=None,
    pop_size=100,
    ngen=50,
    cxpb=0.5,
    mutpb=0.2,
    seed=42,
    cost_factor=0.002,
):
    """
    Select a subset of rules using a Genetic Algorithm.

    Parameters
    ----------
    coverage : (n_rules, n_samples) binary matrix
        Rule coverage matrix.
    y : (n_samples,) binary labels
        1 = positive, 0 = negative.
    rule_costs : (n_rules,) array or None
        Cost per rule (default = 1.0 per rule).
    initial_solutions : list[np.ndarray] or None
        Optional warm-start solutions (binary vectors).
    """

    rng = np.random.default_rng(seed)
    n_rules, n_samples = coverage.shape

    if rule_costs is None:
        rule_costs = np.ones(n_rules)

    # ensure sparse
    if not sp.issparse(coverage):
        coverage = sp.csr_matrix(coverage)

    # ----------------------------
    # DEAP setup
    # ----------------------------
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", rng.integers, 0, 2)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_rules,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------------------------
    # Objective
    # ----------------------------
    def evaluate(individual):
        mask = np.asarray(individual, dtype=bool)
        if not mask.any():
            return (-np.inf,)

        # OR over selected rules → predicted positives
        # coverage: (n_rules, n_samples)
        covered = coverage[mask].sum(axis=0).A1 > 0

        # confusion terms
        y_pos = y == 1
        y_neg = ~y_pos

        TP = np.sum(covered & y_pos)
        FP = np.sum(covered & y_neg)
        FN = np.sum(~covered & y_pos)

        # F3 score
        beta2 = 9.0
        denom = (1 + beta2) * TP + beta2 * FN + FP
        if denom == 0:
            f3 = 0.0
        else:
            f3 = (1 + beta2) * TP / denom

        # penalize complexity
        cost = np.sum(rule_costs[mask])

        # maximize
        return (f3 - cost_factor * cost,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # for printing stats during run
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(1)  # store best individual

    # ----------------------------
    # Population
    # ----------------------------
    population = []

    if initial_solutions is not None:
        for sol in initial_solutions:
            population.append(creator.Individual(list(sol)))

    while len(population) < pop_size:
        population.append(toolbox.individual())

    # ----------------------------
    # Run GA
    # ----------------------------
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # ----------------------------
    # Best solution
    # ----------------------------
    best = tools.selBest(population, k=1)[0]
    best_mask = np.array(best, dtype=bool)
    best_obj = evaluate(best)[0]

    return {
        "selected_rule_indices": np.where(best_mask)[0],
        "mask": best_mask,
        "objective": best_obj,
        "n_selected": best_mask.sum(),
    }


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
    """
    pruned_rules: Set[Rule] = set()

    if mode == "or":
        for term in rule:
            feat_inds, is_pos = term
            if len(feat_inds) <= 1:
                continue
            for f in feat_inds:
                new_inds = feat_inds - {f}
                new_term = (new_inds, is_pos)
                new_rule = (rule - {term}) | {new_term}
                pruned_rules.add(new_rule)
    elif mode == "and":
        if len(rule) > 1:
            for term in rule:
                pruned_rules.add(rule - {term})
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Filter out rules that have only negative terms
    pruned_rules = {r for r in pruned_rules if any(t[-1] for t in r)}

    return pruned_rules


def select_best_metric(
    metrics: List[Dict],
    acceptance_metric: str,
    acceptance_threshold: float,
    removal_threshold: float,
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

    # First, filter only metrics passing acceptance_threshold
    accepted_indices = [
        i for i, m in enumerate(metrics) if m[acceptance_metric] > acceptance_threshold
    ]

    if not accepted_indices:
        return None, False

    # Separate metrics above removal_threshold
    preferred_indices = [
        i for i in accepted_indices if metrics[i][acceptance_metric] > removal_threshold
    ]

    # Decide which indices to consider
    if preferred_indices:
        remove_old = True
        candidate_indices = preferred_indices
    else:
        remove_old = False
        candidate_indices = accepted_indices

    # Pick the one with max f among candidates
    for i in candidate_indices:
        if metrics[i]["f"] > best_f:
            best_f = metrics[i]["f"]
            best_idx = i

    return best_idx, remove_old


def prune_rule_greedy(X, y, rule: Rule, beta: float = 0.1):
    """
    Greedy prune a single rule in two stages:
      1) remove OR features (inside disjunctions) - only disjunctions with >1 features
      2) remove AND terms (entire tuple/term)
    Selection criterion: maximize F_beta (default beta=0.1).
    Acceptance / remove-old thresholds follow your spec:
      - OR-feature removal: accept if tp_gain > -0.1 ; remove old if tp_gain > -0.01
      - AND-term removal: accept if precision_gain > -0.1 ; remove old if precision_gain > -0.01

    Returns:
      best_rule: pruned rule (in same format)
      history: list of remembered rules (each rule is a deep copy)
      history_meta: list of dicts with metrics for each remembered rule
    """
    # Prepare
    current_rule: Rule = rule
    history: List[Rule] = [current_rule]
    history_meta = []

    # baseline metrics
    mask = coverage_of_rule(X, current_rule)
    p_old, r_old, tp_old, ret_old, pos_count = metrics_from_mask(mask, y)
    if tp_old == 0:
        return None, [], []
    f_old = f_beta(p_old, r_old, beta)
    history_meta.append(
        {
            "f": f_old,
            "precision": p_old,
            "recall": r_old,
            "tp": tp_old,
            "returned": ret_old,
        }
    )

    for mode in ["and", "or"]:
        while True:
            # best_candidate = None
            # best_metrics = None
            # best_f = -np.inf
            metrics = []
            # iterate over terms
            candidate_rules = list(
                generate_one_step_rule_variations(
                    rule=current_rule,
                    mode=mode,
                )
            )
            if not candidate_rules:
                break
            # print()
            # print("current", current_rule)
            # print("candidate_rules", candidate_rules)
            # print()
            for cand_rule in candidate_rules:
                # compute metrics
                mask_c = coverage_of_rule(X, cand_rule)
                p_new, r_new, tp_new, ret_new, _ = metrics_from_mask(mask_c, y)
                f_new = f_beta(p_new, r_new, beta)

                tp_gain = (tp_new - tp_old) / tp_old
                precision_gain = p_new - p_old

                metrics.append(
                    {
                        "f": f_new,
                        "precision": p_new,
                        "recall": r_new,
                        "tp": tp_new,
                        "returned": ret_new,
                        "tp_gain": tp_gain,
                        "precision_gain": precision_gain,
                    }
                )
            if mode == "or":
                acceptance_metric = "tp_gain"
                acceptance_threshold = -0.1
                removal_threshold = -0.01
            else:
                acceptance_metric = "precision_gain"
                acceptance_threshold = -0.1
                removal_threshold = -0.01
            best_candidate_index, remove_old = select_best_metric(
                metrics,
                acceptance_metric=acceptance_metric,
                acceptance_threshold=acceptance_threshold,
                removal_threshold=removal_threshold,
            )
            if best_candidate_index is None:
                break
            best_rule = candidate_rules[best_candidate_index]
            best_metric = metrics[best_candidate_index]

            # accept the new rule
            current_rule = best_rule

            # update old metrics
            p_old, r_old = best_metric["precision"], best_metric["recall"]

            if remove_old:
                # remove the previous rule from memory (i.e., drop the version before the last)
                # We keep only the most recent one in history when removal condition met.
                if len(history) >= 1:
                    # drop the second-to-last (the previous state) because "new rule is so good"
                    history.pop(-1)
                    history_meta.pop(-1)

            assert current_rule is not None
            assert current_rule != []
            assert not any([not term[0] for term in current_rule])

            history.append(current_rule)
            history_meta.append(best_metric)
            # print("history", history)
            # print("best_metric", best_metric)
    return current_rule, history, history_meta


""""
TODO
-1) ==DONE== add complexity to ga
0) note all the below into nodes for later writing
1) pass intial solutions (trees of forest) to ga
    - for that we have to remember which rules we deleted by duplicate
2) extract rules from tree better (find optimal threshold makes it worse)
    - simply threshold is not good
    - cost complexity pruning? which rules and their ancestors to include?
    - include more rules -> hence remove last view features of each rule and include those aswell?
    - algo:
        - caclulate very low threshold for each tree -> to get many 1 rules
            - e.g. simply take standard threshold of weighted trees: num_class_1 > class_1/(class_1+class_2) (using weight)
        - for each rule:
            - let:
                - precision_gain = (precision - precision_old) 
                    - range: [-1, 1]
                - tp_gain = (tp_new - tp_old) / tp_old
            - greedily remove (to remove useless stuff):
                - AND term that maximizes: F0.1-score
                    - can only increase #covered by removing AND term
                    - accept new rule if: precision_gain > -0.1
                    - remove old rule if: precision_gain > -0.01
                - OR term that maximizes: F0.1-score
                    - can only decrease #covered by removing OR term
                    - accept new rule if: tp_gain > -0.1
                    - remove old rule if: tp_gain > -0.01 
            - greedily remove (for better rules):
                - AND term that maximizes: f0.1-score (precision 10 times more improtant than recall)
                    - f_imp = 0.01*(#covered - #covered_old)/#covered_old + (precision - precision_old)
                    - condition: (precision_old - precision) < 0.2
                    - can only increase #covered by removing
                    - can increase and decrease precision
                    - if we cover twice as much than its ok if we lose 1% precision
                    - GPT: f = α · log(covered / covered_old) + (precision − precision_old)
                    - remove old rule if: (precision_old - precision) < 0.01
                - OR term that maximizes: 0.1*(#covered - #covered_old)/#covered_old + (precision - precision_old)
                    - can only decrease #covered by removing OR term
                    - can increase and decrease precision
                    - if we gain 10% preicsion then its ok if we cover half as much
                    - remove old rule if: (precision_old - precision) > 0 and (#covered - #covered_old)/#covered_old < 0.05
3) different target? select subset rules which mimic the forest output (not the training data)???
4) ==DONE== varying parameters across trees of forest. vary the following: 
    - min_impurity_decrease_range because we otherwise cannot learn the function (A and B) OR (C AND D)
        because always A OR C is selected first. (migh this be solvable using competing AND and OR splits?)
    - max_features because with small max_features (=sqrt) we cannot learn A and B and C and D since it is unlikely that those are all avaiable at the given nodes
5) ==DONE== first tree always gets all features currently. is that good? somewhat cheating?
6) compare rf results to best tree (did we find better solutions htat best tree among the forest)
7) does old prune algo work when it removes the last word of a term? hence empty list?
8) todo: only keep two best pruned rules per rule (instead of delete old)
9) always check if pruned rule is already generated then we can stop there (not prune it further)
10) pruned rule sometimes empty
"""
