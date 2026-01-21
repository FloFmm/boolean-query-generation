from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Iterable, List, Optional, Tuple, Union
from app.helper.helper import f_beta, precision_score, recall_score

@dataclass
class IncState:
    selected: set
    cover_count: np.ndarray
    TP: int
    FP: int
    FN: int
    cost: float
    score: float


def score_from_counts(
    TP: int, FP: int, FN: int, cost: float, beta: float, cost_factor: float
) -> float:
    p = precision_score(TP=TP, FP=FP)
    r = recall_score(TP=TP, FN=FN)
    return f_beta(precision=p, recall=r, beta=beta) - cost_factor * cost


def init_state(
    init_solution: Iterable[int],
    coverage: csr_matrix,
    y_pos: np.ndarray,
    rule_costs: np.ndarray,
    beta: float,
    cost_factor: float,
) -> IncState:
    n_samples = coverage.shape[1]
    cover_count = np.zeros(n_samples, dtype=np.int32)

    for r in init_solution:
        cover_count[coverage[r].indices] += 1

    covered = cover_count > 0
    TP = int(np.sum(covered & y_pos))
    FP = int(np.sum(covered & ~y_pos))
    FN = int(np.sum(~covered & y_pos))
    cost = float(
        np.sum(rule_costs[list(init_solution)]) if len(list(init_solution)) > 0 else 0.0
    )

    score = score_from_counts(TP, FP, FN, cost, beta, cost_factor)

    return IncState(set(init_solution), cover_count, TP, FP, FN, cost, score)


def delta_from_masks(
    lost_idx,
    gained_idx,
    cover_count,
    y_pos,
    y_neg,
):
    dTP = 0
    dFP = 0

    if lost_idx is not None:
        lost = cover_count[lost_idx] == 1
        dTP -= np.sum(lost & y_pos[lost_idx])
        dFP -= np.sum(lost & y_neg[lost_idx])

    if gained_idx is not None:
        gained = cover_count[gained_idx] == 0
        dTP += np.sum(gained & y_pos[gained_idx])
        dFP += np.sum(gained & y_neg[gained_idx])

    dTP = int(dTP)
    dFP = int(dFP)
    dFN = -dTP
    return dTP, dFP, dFN


def score_after_move(
    state, r_in, r_out, coverage, y_pos, y_neg, rule_costs
):
    idx_in = None
    idx_out = None
    new_cost = state.cost
    if r_in is not None:  # ADD
        idx_in = coverage[r_in].indices
        new_cost += float(rule_costs[r_in])
    if r_out is not None:  # REMOVE
        idx_out = coverage[r_out].indices
        new_cost -= float(rule_costs[r_out])

    if r_in is not None and r_out is not None:  # SWAP
        overlap = np.intersect1d(idx_out, idx_in, assume_unique=True)
        if overlap.size:
            idx_out = np.setdiff1d(idx_out, overlap, assume_unique=True)
            idx_in = np.setdiff1d(idx_in, overlap, assume_unique=True)
    dTP, dFP, dFN = delta_from_masks(idx_out, idx_in, state.cover_count, y_pos, y_neg)
    return state.TP + dTP, state.FP + dFP, state.FN + dFN, new_cost


def apply_delta(
    state: IncState, dTP: int, dFP: int, dFN: int, dcost: float, new_score: float
) -> None:
    state.TP += dTP
    state.FP += dFP
    state.FN += dFN
    state.cost += dcost
    state.score = new_score


def select_rules_greedy(
    coverage: Union[np.ndarray, csr_matrix],
    y: np.ndarray,
    rule_costs: np.ndarray,
    beta: float,
    cost_factor: float,
    initial_solutions: Optional[List[Iterable[int]]] = None,
    max_iter: int = 100,
    max_rules: int = 10,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Fully incremental greedy local search with add / remove / swap moves.
    Returns dict
    """
    if not isinstance(coverage, csr_matrix):
        coverage = csr_matrix(coverage)

    y_pos = y == 1
    y_neg = ~y_pos

    n_rules = coverage.shape[0]
    initial_solutions = initial_solutions or [[]]

    best_global: Optional[List[int]] = None
    best_score = -np.inf
    memo = {}
    for init_idx, init in enumerate(initial_solutions):
        state = init_state(init, coverage, y_pos, rule_costs, beta, cost_factor)
        if verbose:
            print(
                f"\n[Init {init_idx}] start score = {state.score:.6f}, rules = {sorted(state.selected)}"
            )
        rules = frozenset(state.selected)
        if rules in memo and memo[rules][-1]:
            print("pruned")
            continue
        else:
            memo[rules] = ((state.TP, state.FP, state.FN, state.cost), state.score, True)

        for i in range(max_iter):
            best_move = None
            best_move_score = state.score

            cand_moves = []
            if len(state.selected) < max_rules:  # ADD
                cand_moves += [
                    (r_in, None) for r_in in range(n_rules) if r_in not in state.selected
                ]
            if len(state.selected) > 1:  # REMOVE
                cand_moves += [(None, r_out) for r_out in list(state.selected)]
            cand_moves += [ # SWAP
                (r_in, r_out)
                for r_out in list(state.selected)
                for r_in in range(n_rules)
                if r_in not in state.selected
            ]
            for r_in, r_out in cand_moves:
                rules = state.selected.copy()
                if r_in is not None:
                    rules.add(r_in)
                if r_out is not None:
                    rules.remove(r_out)
                rules = frozenset(rules)
                if rules in memo:
                    # if verbose:
                    #     print("score_loaded")
                    (TP, FP, FN, cost), score, _ = memo[rules]
                else:
                    TP, FP, FN, cost = score_after_move(
                        state=state,
                        r_in=r_in,
                        r_out=r_out,
                        coverage=coverage,
                        y_pos=y_pos,
                        y_neg=y_neg,
                        rule_costs=rule_costs,
                    )
                    score = score_from_counts(
                        TP,
                        FP,
                        FN,
                        cost,
                        beta,
                        cost_factor,
                    )
                    memo[rules] = ((TP, FP, FN, cost), score, False)
                if score > best_move_score:
                    # store as (type, r, d, score)
                    move_type = "REMOVE"
                    if r_in is not None:
                        if r_out is not None:
                            move_type = "SWAP"
                        else:
                            move_type = "ADD"
                    
                    best_move = (move_type, r_in, r_out, (TP, FP, FN, cost), score)
                    best_move_score = score

            if best_move is None:
                break
            move_type, r_in, r_out, (TP, FP, FN, cost), score = best_move
            state.TP = TP
            state.FP = FP
            state.FN = FN
            state.cost = cost
            state.score = score
            
            if r_in is not None:
                idx = coverage[r_in].indices
                state.cover_count[idx] += 1
                state.selected.add(r_in)
            if r_out is not None:
                idx = coverage[r_out].indices
                state.cover_count[idx] -= 1
                state.selected.remove(r_out)
            if verbose:
                print(f"new score {state.score:.6f} | iter {i}: {move_type} | r_in={r_in}, r_out={r_out} | {state.selected}")

            d, score, explored = memo[frozenset(state.selected)]
            if explored:
                if verbose:
                    print("pruned")
                break
            else:
                memo[frozenset(state.selected)] = (d, score, True)

        if state.score > best_score:
            best_score = state.score
            best_global = sorted(list(state.selected))

    return {
        "selected_rule_indices": best_global,
        "objective": best_score,
        "n_selected": len(best_global),
    }


# ---------------------- small helper to recompute score from a full solution (for validation) ----------------------
def recompute_score_full(
    solution: Iterable[int],
    coverage: csr_matrix,
    y: np.ndarray,
    rule_costs: np.ndarray,
    beta: float,
    cost_factor: float,
) -> float:
    """Recompute the score by forming the OR coverage from scratch (useful for validation)."""
    if not isinstance(coverage, csr_matrix):
        coverage = csr_matrix(coverage)
    n_samples = coverage.shape[1]
    mask = np.zeros(coverage.shape[0], dtype=bool)
    for r in solution:
        mask[r] = True
    if mask.sum() == 0:
        covered = np.zeros(n_samples, dtype=bool)
    else:
        covered = coverage[mask].sum(axis=0).A1 > 0
    y_pos = y == 1
    TP = int(np.sum(covered & y_pos))
    FP = int(np.sum(covered & ~y_pos))
    FN = int(np.sum(~covered & y_pos))
    cost = float(np.sum(rule_costs[list(solution)]) if len(list(solution)) > 0 else 0.0)
    return score_from_counts(TP, FP, FN, cost, beta, cost_factor)


# --------------------------------- Example __main__ test ---------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # Problem size for demo
    n_samples = 500_000
    n_rules = 250

    # Create a random sparse coverage: each rule covers ~p fraction of samples
    p = 0.01
    rows = []
    cols = []
    for r in range(n_rules):
        covered_idx = rng.choice(n_samples, size=int(p * n_samples), replace=False)
        rows.extend([r] * covered_idx.size)
        cols.extend(covered_idx.tolist())

    cov = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_rules, n_samples)
    )

    # True labels (imbalance)
    y = (rng.random(n_samples) < 0.2).astype(int)

    # rule costs small positive
    rule_costs = rng.uniform(0.01, 0.2, size=n_rules)

    # params
    beta = 3.0
    cost_factor = 0.01

    # some initial solutions: empty, random singletons, and some random pairs
    initial_solutions = [[]]
    for _ in range(50):
        initial_solutions.append(
            [int(x) for x in rng.choice(n_rules, size=6, replace=False)]
        )

    result = select_rules_greedy(
        cov,
        y,
        rule_costs,
        beta,
        cost_factor,
        initial_solutions,
        max_iter=200,
        max_rules=10,
        verbose=True,
    )
    best_sol, best_score = result["selected_rule_indices"], result["objective"]

    print("Best solution (rules):", best_sol)
    print("Reported best score:", best_score)

    # Validate by recomputing full score
    full_score = recompute_score_full(best_sol, cov, y, rule_costs, beta, cost_factor)
    print("Recomputed full coverage score (validation):", full_score)

    # small assertion to ensure scores agree (allow tiny fp rounding)
    assert abs(full_score - best_score) < 1e-8, (
        "Incremental and full recompute disagree!"
    )
    print("Validation passed: incremental and full recompute match.")
