"""Figures for the grocery case: repeated product types in a top-10 list, and what recency weighting buys."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import simulate as S

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#E5E7EB", "axes.axisbelow": True})
BLUE, ORANGE, GREEN, GREY = "#006699", "#E8871E", "#2E8B57", "#8A8A8A"
summary, rows = S.run()

# 1. How many lists repeat a product type, and how many slots go to a second variant
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
names = [("item", "Ranks SKUs directly", ORANGE), ("two_stage", "Type, then variant", BLUE)]
dup_hist = {}
for key, _, _ in names:
    c = np.bincount([r["duplicate_slots"] for r in rows[key]], minlength=4)[:4]
    dup_hist[key] = 100 * c / len(rows[key])
x = np.arange(4)
for i, (key, label, color) in enumerate(names):
    axes[0].bar(x + (i - 0.5) * 0.38, dup_hist[key], 0.38, color=color, label=label)
axes[0].set_xticks(x, ["0", "1", "2", "3+"]); axes[0].set_xlabel("Slots spent on a second variant of a type already shown")
axes[0].set_ylabel("% of customers"); axes[0].set_title("Repeated product types in the top 10"); axes[0].legend(fontsize=8)
vals = [summary[k]["type_hits"] for k, _, _ in names] + [summary["item"]["sku_hits"], summary["two_stage"]["sku_hits"]]
labels = ["types bought", "types bought", "exact SKUs bought", "exact SKUs bought"]
pos = [0, 1, 2.6, 3.6]
cols = [ORANGE, BLUE, ORANGE, BLUE]
axes[1].bar(pos, vals, color=cols, width=0.8)
for p, v in zip(pos, vals):
    axes[1].text(p, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
axes[1].set_xticks([0.5, 3.1], ["Product types bought\nfrom the list", "Exact SKUs bought\nfrom the list"])
axes[1].set_ylabel("Per held-out order"); axes[1].set_title("What the next order took from the list"); axes[1].set_ylim(0, 4.6)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_repeated_types.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Recency weighting: helps customers who changed habits, costs a little on stable ones
fig, ax = plt.subplots(figsize=(8, 3.4))
groups = [("type_hits_stable", "Stable habits (60%)"), ("type_hits_changed_habits", "Changed habits halfway (40%)")]
models = [("item", "Ranks SKUs directly", ORANGE), ("two_stage_no_recency", "Two-stage, all orders equal", GREY),
          ("two_stage", "Two-stage, recent orders weigh more", BLUE)]
for j, (key, label, color) in enumerate(models):
    v = [summary[key][g] for g, _ in groups]
    xs = np.arange(2) + (j - 1) * 0.26
    ax.bar(xs, v, 0.26, color=color, label=label)
    for xx, vv in zip(xs, v):
        ax.text(xx, vv + 0.05, f"{vv:.2f}", ha="center", fontsize=8)
ax.set_xticks([0, 1], [g for _, g in groups]); ax.set_ylabel("Product types bought from the list\nper held-out order")
ax.set_ylim(0, 5); ax.set_title("Weighting recent orders, the idea behind DIN")
ax.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_recency.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok", {k: dup_hist[k].round(1).tolist() for k in dup_hist})
