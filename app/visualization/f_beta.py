import numpy as np
import matplotlib.pyplot as plt
from app.config.config import apply_matplotlib_style, COLORS
import app.config.config as config
apply_matplotlib_style()

def f_beta(precision, recall, beta):
    """Compute F-beta score."""
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


# Example set of (precision, recall) pairs
pr_pairs = [
    # (0.001, 0.95),
    # (0.01, 0.9),
    # (0.02, 0.8),
    # (0.06, 0.6),
    # (0.10, 0.4),
    # (0.022, 0.934), #Manual
    (0.024, 0.816, "Semantic"),
    (0.051, 0.757, "Fine-Tuned LLM"),
    (0.075, 0.504, "Chat-GPT"),
]

# X values: beta from 1 to 50
x = np.arange(1, 51)


plt.figure(figsize=(
    config.FIGURE_CONFIG["full_width"],
    config.FIGURE_CONFIG["full_width"] * config.FIGURE_CONFIG["aspect_ratio"] * 0.5
))

for i, (p, r, name) in enumerate(pr_pairs):
    y = [f_beta(p, r, beta) for beta in x]
    plt.plot(
        x, y,
        label=f"{name} (P={p}, R={r})",
        color=COLORS["category"][i % len(COLORS["category"])]
    )

plt.xlabel("β")
plt.ylabel("F-β score")
# plt.title("F-β Score for Different Precision–Recall Pairs")
plt.legend()
plt.grid(True)
plt.savefig("../master-thesis-writing/writing/thesis/images/graphs/f_beta/f_beta.png")
