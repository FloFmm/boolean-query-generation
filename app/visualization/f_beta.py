import numpy as np
import matplotlib.pyplot as plt
from app.config.config import apply_matplotlib_style, COLORS

apply_matplotlib_style()

def f_beta(precision, recall, beta):
    """Compute F-beta score."""
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


# Example set of (precision, recall) pairs
pr_pairs = [
    (0.001, 0.95),
    (0.01, 0.9),
    (0.02, 0.8),
    (0.06, 0.6),
    (0.10, 0.4),
]

# X values: beta from 1 to 50
x = np.arange(1, 51)

plt.figure()

for i, (p, r) in enumerate(pr_pairs):
    y = [f_beta(p, r, beta) for beta in x]
    plt.plot(x, y, label=f"P={p}, R={r}", color=COLORS["category"][i % len(COLORS["category"])])

plt.xlabel("beta (β)")
plt.ylabel("F-beta score")
plt.title("F-beta Score for Different Precision–Recall Pairs")
plt.legend()
plt.grid(True)
plt.savefig("../master-thesis-writing/writing/thesis/images/graphs/f_beta/f_beta.png")
