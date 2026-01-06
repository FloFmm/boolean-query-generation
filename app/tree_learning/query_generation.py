from collections import defaultdict
from typing import List, Tuple
import numpy as np
import numpy as np
import scipy.sparse as sp
from deap import base, creator, tools, algorithms

def compute_variable_frequencies(rules, rule_tree_map, n_trees):
    """
    Parameters
    ----------
    rules : list[list[tuple[list[int], bool]]]

    Returns
    -------
    tree_freq : dict[int, float]
    rule_freq : dict[int, float]
    """
    trees_per_var = defaultdict(set)
    rules_per_var = defaultdict(int)

    for r_idx, rule in enumerate(rules):
        vars_in_rule = set()
        for feature_indices, features, is_pos in rule:
            vars_in_rule.update(features)

        for v in vars_in_rule:
            trees_per_var[v].add(rule_tree_map[r_idx])
            rules_per_var[v] += 1

    tree_freq = {v: len(tset) / n_trees for v, tset in trees_per_var.items()}
    rule_freq = {v: cnt / len(rules) for v, cnt in rules_per_var.items()}

    return tree_freq, rule_freq

def prune_rules(
    rules,
    tree_freq,
    rule_freq,
    min_tree_occ=0.05,
    min_rule_occ=0.05,
):
    """
    Removes variables not meeting frequency thresholds.
    """
    kept_vars = {
        v for v in tree_freq
        if tree_freq[v] >= min_tree_occ and rule_freq.get(v, 0) >= min_rule_occ
    }
    pruned_rules = []

    for rule in rules:
        new_rule = []

        for feature_indices, features, is_pos in rule:
            new_features = []
            new_feature_indices = []
            for i, feature_name in enumerate(features):
                if feature_name in kept_vars:
                    new_features.append(feature_name)
                    new_feature_indices.append(feature_indices[i])
            if not new_features:
                break  # AND-clause fails → drop rule
            new_rule.append((new_feature_indices, new_features, is_pos))

        if new_rule:
            pruned_rules.append(new_rule)

    return pruned_rules, kept_vars

def deduplicate_rules(rules):
    """
    Removes duplicate rules. Order of triples in a rule and order of features/indices does not matter.
    
    Args:
        rules: list of rules, each rule is a list of triples
               (feature_indices: list[int], features: list[str], is_pos: bool)
    
    Returns:
        List of unique rules.
    """
    seen = set()
    unique_rules = []

    for rule in rules:
        canonical_rule = []

        for feature_indices, features, is_pos in rule:
            # Sort indices and features to make order irrelevant
            canonical_triple = (
                frozenset(feature_indices),
                frozenset(features),
                is_pos
            )
            canonical_rule.append(canonical_triple)

        # The order of triples in the rule doesn't matter
        canonical_rule_set = frozenset(canonical_rule)

        if canonical_rule_set not in seen:
            seen.add(canonical_rule_set)
            # Recover original rule structure
            unique_rules.append(rule)

    return unique_rules

def compute_rule_coverage(X, rules):
    """
    Parameters
    ----------
    X : csr_matrix (binary)
    rules : list[list[tuple[list[int], bool]]]

    Returns
    -------
    coverage : np.ndarray, shape (n_rules, n_samples)
    """
    n_samples = X.shape[0]
    coverage = np.zeros((len(rules), n_samples), dtype=np.uint8)
    for i, rule in enumerate(rules):
        mask = np.ones(n_samples, dtype=bool)

        for feature_indices, features, is_pos in rule:
            cols = list(feature_indices)
            if is_pos:
                # OR present
                mask &= (X[:, cols].getnnz(axis=1) > 0)
            else:
                # OR absent
                mask &= (X[:, cols].getnnz(axis=1) == 0)

            if not mask.any():
                break

        coverage[i, mask] = 1

    return coverage

def extract_and_vectorize_rules(
    forest,
    X,
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
    pruned_rules = deduplicate_rules(pruned_rules)
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
    clause = " ADDED_OR ".join(
        expand_term(term_expansions, f) for f in features
    )

    if "OR" in clause:
        clause = f"({clause})"

    if not is_positive:
        clause = f"NOT {clause}"

    return clause 
    
def rules_to_pubmed_query(rules: List[List[Tuple[List[str], List[int], bool]]], term_expansions: dict = None):
    """
    Converts extracted AND-of-OR rules into a PubMed DNF boolean query.
    """
    query_clauses = []
    path_lens = []

    for rule in rules:
        if not any(is_pos for _, _, is_pos in rule):
            # Skip clauses that are all NOT
            continue
        pos_literals = [
            literal_to_pubmed(features, is_pos, term_expansions)
            for features_indices, features, is_pos in rule if is_pos
        ]
        neg_literals = [
            literal_to_pubmed(features, is_pos, term_expansions)
            for features_indices, features, is_pos in rule if not is_pos
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
    return query.replace("ADDED_OR", "OR").replace("SYNONYM_OR", "OR"), query_size

def query_size(rules):
    query_size = {
        "paths": len(rules),
        "avg_path_len": sum([len(rule) for rule in rules]) / len(rules) if len(rules) else 0,
        "ANDs": sum([len(rule) for rule in rules if rule[-1]]) - 1,
        "NOTs": sum([len(rule) for rule in rules if not rule[-1]]),
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
        y_pos = (y == 1)
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
3) different target? select subset rules which mimic the forest output (not the training data)???
4) ==DONE== varying parameters across trees of forest. vary the following: 
    - min_impurity_decrease_range because we otherwise cannot learn the function (A and B) OR (C AND D)
        because always A OR C is selected first. (migh this be solvable using competing AND and OR splits?)
    - max_features because with small max_features (=sqrt) we cannot learn A and B and C and D since it is unlikely that those are all avaiable at the given nodes
5) ==DONE== first tree always gets all features currently. is that good? somewhat cheating?
6) compare rf results to best tree (did we find better solutions htat best tree among the forest)
"""
