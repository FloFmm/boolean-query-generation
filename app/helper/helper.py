import numpy as np

def biased_random(low, high, exponent=2, rng=None):
    """
    Generates a random integer between low and high, biased toward low.

    Args:
        n: total number of features
        exponent: >0, higher = stronger bias toward low
    
    Returns:
        float: biased random number between low and high
    """
    if rng is None:
        rng = np.random
    
    # Uniform sample between 0 and 1
    u = rng.rand()

    # Apply exponential decay: bias toward 0
    biased = u ** exponent

    # Map to range [low, high]
    result = low + (high - low) * biased

    return result


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (precision * recall) / (b2 * precision + recall)

def precision_score(TP: int, FP: int) -> float:
    """Compute precision = TP / (TP + FP)"""
    denom = TP + FP
    if denom == 0:
        return 0.0
    return TP / denom


def recall_score(TP: int, FN: int) -> float:
    """Compute recall = TP / (TP + FN)"""
    denom = TP + FN
    if denom == 0:
        return 0.0
    return TP / denom