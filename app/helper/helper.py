import numpy as np

def biased_random(low, high, exponent=2, rng=None):
    """
    Generates a random integer between low and high, biased toward low.

    Args:
        n: total number of features
        exponent: >0, higher = stronger bias toward low
    
    Returns:
        int: biased random number between low and high
    """
    if rng is None:
        rng = np.random
    
    # Uniform sample between 0 and 1
    u = rng.rand()

    # Apply exponential decay: bias toward 0
    biased = u ** exponent

    # Map to range [low, high]
    result = int(low + (high - low) * biased)

    return result