from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from typing import Iterable, List, Optional, Tuple, Union


@dataclass
class IncState:
    selected: set
    cover_count: np.ndarray
    TP: int
    FP: int
    FN: int
    cost: float
    score: float


def score_from_counts(TP: int, FP: int, FN: int, cost: float, beta2: float, cost_factor: float) -> float:
    denom = (1 + beta2) * TP + beta2 * FN + FP
    if denom == 0:
        return -cost_factor * cost
    return (1 + beta2) * TP / denom - cost_factor * cost


def init_state(
    init_solution: Iterable[int],
    coverage: csr_matrix,
    y_pos: np.ndarray,
    rule_costs: np.ndarray,
    beta2: float,
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
    cost = float(np.sum(rule_costs[list(init_solution)]) if len(list(init_solution)) > 0 else 0.0)

    score = score_from_counts(TP, FP, FN, cost, beta2, cost_factor)

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

def delta_add(state, r, coverage, y_pos, y_neg, rule_costs):
    idx = coverage[r].indices
    dTP, dFP, dFN = delta_from_masks(None, idx, state.cover_count, y_pos, y_neg)
    return dTP, dFP, dFN, float(rule_costs[r])

def delta_remove(state, r, coverage, y_pos, y_neg, rule_costs):
    idx = coverage[r].indices
    dTP, dFP, dFN = delta_from_masks(idx, None, state.cover_count, y_pos, y_neg)
    return dTP, dFP, dFN, -float(rule_costs[r])

def delta_swap(state, r_out, r_in, coverage, y_pos, y_neg, rule_costs):
    idx_out = coverage[r_out].indices
    idx_in = coverage[r_in].indices

    overlap = np.intersect1d(idx_out, idx_in, assume_unique=True)
    if overlap.size:
        idx_out = np.setdiff1d(idx_out, overlap, assume_unique=True)
        idx_in = np.setdiff1d(idx_in, overlap, assume_unique=True)

    dTP, dFP, dFN = delta_from_masks(idx_out, idx_in, state.cover_count, y_pos, y_neg)
    return dTP, dFP, dFN, float(rule_costs[r_in] - rule_costs[r_out])

def apply_delta(state: IncState, dTP: int, dFP: int, dFN: int, dcost: float, new_score: float) -> None:
    state.TP += dTP
    state.FP += dFP
    state.FN += dFN
    state.cost += dcost
    state.score = new_score

def apply_add(state: IncState, r: int, coverage: csr_matrix) -> None:
    idx = coverage[r].indices
    state.cover_count[idx] += 1
    state.selected.add(r)

def apply_remove(state: IncState, r: int, coverage: csr_matrix) -> None:
    idx = coverage[r].indices
    state.cover_count[idx] -= 1
    state.selected.remove(r)

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
    Returns (best_solution_list, best_score).
    """
    if not isinstance(coverage, csr_matrix):
        coverage = csr_matrix(coverage)

    beta2 = beta * beta
    y_pos = (y == 1)
    y_neg = ~y_pos

    n_rules = coverage.shape[0]
    initial_solutions = initial_solutions or [[]]

    best_global: Optional[List[int]] = None
    best_score = -np.inf
    for init_idx, init in enumerate(initial_solutions):
        state = init_state(init, coverage, y_pos, rule_costs, beta2, cost_factor)

        if verbose:
            print(f"\n[Init {init_idx}] start score = {state.score:.6f}, rules = {sorted(state.selected)}")
        
        for i in range(max_iter):
            best_move = None
            best_move_score = state.score

            # ADD
            if len(state.selected) < max_rules:
                for r in range(n_rules):
                    if r in state.selected:
                        continue
                    d = delta_add(state, r, coverage, y_pos, y_neg, rule_costs)
                    score = score_from_counts(state.TP + d[0], state.FP + d[1], state.FN + d[2], state.cost + d[3], beta2, cost_factor)
                    if score > best_move_score:
                        # store as (type, r, d, score)
                        best_move = ("add", r, d, score)
                        best_move_score = score

            # REMOVE (allow removing if at least one selected)
            if len(state.selected) > 0:
                for r in list(state.selected):
                    d = delta_remove(state, r, coverage, y_pos, y_neg, rule_costs)
                    score = score_from_counts(state.TP + d[0], state.FP + d[1], state.FN + d[2], state.cost + d[3], beta2, cost_factor)
                    if score > best_move_score:
                        best_move = ("remove", r, d, score)
                        best_move_score = score

            # SWAP
            # store as (type, r_out, r_in, d, score)
            for r_out in list(state.selected):
                for r_in in range(n_rules):
                    if r_in in state.selected:
                        continue
                    d = delta_swap(state, r_out, r_in, coverage, y_pos, y_neg, rule_costs)
                    score = score_from_counts(state.TP + d[0], state.FP + d[1], state.FN + d[2], state.cost + d[3], beta2, cost_factor)
                    if score > best_move_score:
                        best_move = ("swap", r_out, r_in, d, score)
                        best_move_score = score

            if best_move is None:
                break

            # APPLY BEST MOVE (unpack according to move type)
            if best_move[0] == "add":
                _, r, d, score = best_move
                apply_delta(state, *d, score)
                apply_add(state, r, coverage)
                if verbose:
                    print(f"  iter {i}: ADD   r={r:3d} | new score {state.score:.6f}")

            elif best_move[0] == "remove":
                _, r, d, score = best_move
                apply_delta(state, *d, score)
                apply_remove(state, r, coverage)
                if verbose:
                     print(f"  iter {i}: REMOVE   r={r:3d} | new score {state.score:.6f}")

            else:  # swap
                _, r_out, r_in, d, score = best_move
                apply_delta(state, *d, score)
                # perform removal then add to correctly update cover_count & selected
                apply_remove(state, r_out, coverage)
                apply_add(state, r_in, coverage)
                if verbose:
                     print(f"  iter {i}: SWAP   r={r_out} -> r={r_in} | new score {state.score:.6f}")

        if state.score > best_score:
            best_score = state.score
            best_global = sorted(list(state.selected))

    return {
        "selected_rule_indices": best_global,
        "objective": best_score,
        "n_selected": len(best_global),
    }


# ---------------------- small helper to recompute score from a full solution (for validation) ----------------------
def recompute_score_full(solution: Iterable[int], coverage: csr_matrix, y: np.ndarray, rule_costs: np.ndarray, beta2: float, cost_factor: float) -> float:
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
        covered = (coverage[mask].sum(axis=0).A1 > 0)
    y_pos = (y == 1)
    TP = int(np.sum(covered & y_pos))
    FP = int(np.sum(covered & ~y_pos))
    FN = int(np.sum(~covered & y_pos))
    cost = float(np.sum(rule_costs[list(solution)]) if len(list(solution)) > 0 else 0.0)
    return score_from_counts(TP, FP, FN, cost, beta2, cost_factor)


# --------------------------------- Example __main__ test ---------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # Problem size for demo
    n_samples = 50_000
    n_rules = 250

    # Create a random sparse coverage: each rule covers ~p fraction of samples
    p = 0.08
    rows = []
    cols = []
    for r in range(n_rules):
        covered_idx = rng.choice(n_samples, size=int(p * n_samples), replace=False)
        rows.extend([r] * covered_idx.size)
        cols.extend(covered_idx.tolist())

    cov = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_rules, n_samples))

    # True labels (imbalance)
    y = (rng.random(n_samples) < 0.2).astype(int)

    # rule costs small positive
    rule_costs = rng.uniform(0.01, 0.2, size=n_rules)

    # params
    beta2 = 9.0  # F3-like when beta^2 = 9
    cost_factor = 0.01

    # some initial solutions: empty, random singletons, and some random pairs
    initial_solutions = [[]]
    for _ in range(50):
        initial_solutions.append(
            [int(x) for x in rng.choice(n_rules, size=6, replace=False)]
        )
        
    best_sol, best_score = select_rules_greedy(cov, y, rule_costs, beta2, cost_factor, initial_solutions, max_iter=200, max_rules=10, verbose=True)

    print("Best solution (rules):", best_sol)
    print("Reported best score:", best_score)

    # Validate by recomputing full score
    full_score = recompute_score_full(best_sol, cov, y, rule_costs, beta2, cost_factor)
    print("Recomputed full coverage score (validation):", full_score)

    # small assertion to ensure scores agree (allow tiny fp rounding)
    assert abs(full_score - best_score) < 1e-8, "Incremental and full recompute disagree!"
    print("Validation passed: incremental and full recompute match.")
    
    #TODO mabye remeber all already computed rule combiantions, 
    # i can reuse their results and if my best solution is one that was already found then we can stop entirely
