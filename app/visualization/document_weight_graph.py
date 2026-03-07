from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from app.config.config import apply_matplotlib_style, COLORS
apply_matplotlib_style()
import matplotlib as mpl
# mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math mode (serif)

def plot_custom_rank_graph(w_rank1=2, k=1, n=3, inf=8, save_path=None):
    # Define points
    points = [
        (0, w_rank1),
        (k, 1),
        (k, 0),
        (k * (n + 1), 0),
        (k * (n + 1), 1),
        (k * (n + 2), w_rank1),
        (inf, w_rank1)
    ]
    x, y = zip(*points)

    fig, ax = plt.subplots(figsize=(7, 1.75))
    ax.plot(x, y, color=COLORS['primary'], linewidth=2)

    # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Define range_y before axis drawing
    range_y = w_rank1 + 0.3
    # Draw real axes with arrows
    ax.annotate('', xy=(inf+0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2, color='black'))  # x axis
    # Draw y axis as a visible line with arrow
    ax.annotate('', xy=(0, range_y), xytext=(0, min(0, min(y))-0.1), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    # Add y axis label next to axis

    # Set y ticks and labels (math mode, subscript)
    ax.set_yticks([0, 1, w_rank1])
    ax.set_yticklabels(['0', '1', r'$w_{rank1}$'])

    # Remove x ticks and labels
    ax.set_xticks([])
    # Set x label closer to axis
    ax.set_xlabel('Rank')
    ax.set_ylabel('Weight')

    # Draw real lines for ranges at same height, moved up by 0.15
    range_y = w_rank1 + 0.3 + 0.15
    # pseudo relevant: 0 to k (with arrow)
    ax.annotate('', xy=(k, range_y), xytext=(0, range_y), arrowprops=dict(arrowstyle='<->', lw=2, color=COLORS['positive']))
    ax.text((0+k)/2, range_y+0.1, 'pseudo relevant', ha='center', va='bottom', color=COLORS['positive'])
    # Add length label below green arrow
    ax.text((0+k)/2, range_y-0.08, r'$k$', ha='center', va='top', color=COLORS['positive'])
    # don't cares: k to (n+1)k (with arrow)
    ax.annotate('', xy=(k*(n+1), range_y), xytext=(k, range_y), arrowprops=dict(arrowstyle='<->', lw=2, color=COLORS['neutral']))
    ax.text((k+k*(n+1))/2, range_y+0.1, "don't cares", ha='center', va='bottom', color=COLORS['neutral'])
    # Add length label below orange arrow
    ax.text((k+k*(n+1))/2, range_y-0.08, r"$f_{\text{don't-cares}} \cdot k$", ha='center', va='top', color=COLORS['neutral'])
    # pseudo non-relevant: (n+1)k to inf (with arrow)
    ax.annotate('', xy=(inf, range_y), xytext=(k*(n+1), range_y), arrowprops=dict(arrowstyle='<->', lw=2, color=COLORS['negative']))
    ax.text((k*(n+1)+inf)/2, range_y+0.1, 'pseudo non-relevant', ha='center', va='bottom', color=COLORS['negative'])

    # Draw horizontal lines for y labels
    for y_val in [0, 1, w_rank1]:
        ax.axhline(y=y_val, color='gray', linestyle='--', linewidth=0.5)

    # Set limits (reduce whitespace)
    ax.set_xlim(-0.5, inf+1)
    # Tighten y-limits to reduce whitespace
    ax.set_ylim(min(0, min(y))-0.1, range_y+0.2)

    # ...existing code...

    # Optional: save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()




if __name__ == "__main__":
    save_folder = Path("../master-thesis-writing/writing/thesis/images/graphs")
    save_folder.mkdir(parents=True, exist_ok=True)
    save_file = save_folder / "document_weight_graph.png"
    plot_custom_rank_graph(w_rank1=2.5, k=3, n=3, inf=20, save_path=save_file)