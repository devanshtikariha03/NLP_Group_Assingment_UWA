from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("report")
OUT.mkdir(exist_ok=True)


def box(ax, xy, w, h, text, color, fontsize=9):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, style="-"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color="#37474f", linestyle=style)
    )


fig, ax = plt.subplots(figsize=(7.0, 2.55))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

box(ax, (0.05, 2.95), 1.05, 0.55, "Goal $G$", "#e3f2fd", 8)
box(ax, (0.05, 1.73), 1.05, 0.55, "Solution $S_1$", "#fff3e0", 7.5)
box(ax, (0.05, 0.51), 1.05, 0.55, "Solution $S_2$", "#fff3e0", 7.5)
box(ax, (1.40, 1.50), 1.22, 1.05, "Shared\nembedding +\nBiGRU", "#ede7f6", 8)
arrow(ax, (1.10, 3.22), (1.40, 2.35))
arrow(ax, (1.10, 2.00), (1.40, 2.00))
arrow(ax, (1.10, 0.78), (1.40, 1.68))

box(ax, (2.95, 2.93), 1.20, 0.58, "Mean goal\nsummary $g$", "#e8f5e9", 8)
box(ax, (2.95, 1.62), 1.20, 0.76, "Goal-aware\nattention $a_1$", "#fffde7", 8)
box(ax, (2.95, 0.39), 1.20, 0.76, "Goal-aware\nattention $a_2$", "#fffde7", 8)
arrow(ax, (2.62, 2.25), (2.95, 3.10))
arrow(ax, (2.62, 2.05), (2.95, 2.00))
arrow(ax, (2.62, 1.78), (2.95, 0.77))
arrow(ax, (3.50, 2.93), (3.50, 2.38), style="--")
arrow(ax, (3.68, 2.93), (3.68, 1.15), style="--")

box(ax, (4.45, 1.62), 1.45, 0.76, "$r_1=[g; a_1; g\\odot a_1;$\n$|g-a_1|]$", "#e0f7fa", fontsize=7.5)
box(ax, (4.45, 0.39), 1.45, 0.76, "$r_2=[g; a_2; g\\odot a_2;$\n$|g-a_2|]$", "#e0f7fa", fontsize=7.5)
arrow(ax, (4.15, 2.00), (4.45, 2.00))
arrow(ax, (4.15, 0.77), (4.45, 0.77))
arrow(ax, (3.95, 3.00), (4.75, 2.38), style="--")
arrow(ax, (4.00, 2.96), (4.75, 1.15), style="--")

box(ax, (6.25, 1.10), 1.28, 1.18, "Pair contrast\n$r_i$\n$r_i-r_j$\n$r_i\\odot r_j$", "#fce4ec", fontsize=7.5)
arrow(ax, (5.90, 2.00), (6.25, 1.88))
arrow(ax, (5.90, 0.77), (6.25, 1.43))

box(ax, (7.90, 1.25), 0.90, 0.90, "Shared\nMLP score", "#f3e5f5", 7.5)
arrow(ax, (7.53, 1.69), (7.90, 1.69))
box(ax, (9.20, 1.37), 0.72, 0.67, "Softmax\n$p(y)$", "#e8f5e9", 7.5)
arrow(ax, (8.80, 1.69), (9.20, 1.69))

ax.text(7.45, 0.20, "Swap $S_1,S_2$ $\\Rightarrow$ swap logits", ha="center", fontsize=7.5, color="#455a64")
fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Wrote report/architecture.pdf and report/architecture.png")
