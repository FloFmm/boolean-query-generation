from typing import Union
import numpy as np
import scipy.sparse as sp
from deap import base, creator, tools, algorithms
from app.config.config import DEBUG

def select_rules_via_ga(
    coverage: Union[np.ndarray, sp.csr_matrix],
    y: np.ndarray,
    rule_costs: np.ndarray = None,
    initial_solutions=None,
    pop_size=100,
    ngen=50,
    cxpb=0.5,
    mutpb=0.2,
    seed=42,
    cost_factor=0.002,
    beta=3, # -> f3 score is maximized
    max_rules=10,
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
    beta2 = beta*beta
    rng = np.random.default_rng(seed)
    n_rules, n_samples = coverage.shape

    if rule_costs is None:
        rule_costs = np.ones(n_rules)

    # ensure sparse
    if not sp.issparse(coverage): #TODO (currently ga algo transforms matrix himself into sparse). can we do it before that?
        coverage = sp.csr_matrix(coverage)

    # ----------------------------
    # DEAP setup
    # ----------------------------
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", rng.integers, 0, 2)
    # toolbox.register(
    #     "individual",
    #     tools.initRepeat,
    #     creator.Individual,
    #     toolbox.attr_bool,
    #     n=n_rules,
    # )
    def sparse_individual():
        ind = np.zeros(n_rules, dtype=int)
        k = rng.integers(1, min(max_rules, n_rules) + 1)
        idx = rng.choice(n_rules, size=k, replace=False)
        ind[idx] = 1
        return creator.Individual(ind.tolist())
    toolbox.register("individual", sparse_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------------------------
    # Objective
    # ----------------------------
    def evaluate(individual):
        mask = np.asarray(individual, dtype=bool)
        if not mask.any() or mask.sum() > max_rules:
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
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # toolbox.register("mate", tools.cxTwoPoint)
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
        verbose=DEBUG,
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
