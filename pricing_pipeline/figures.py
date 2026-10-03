"""Figures for the pricing case: the five steps, and the latest-price shortcut against the as-of join."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
NAVY, BLUE, TEAL, GREY, ORANGE, RED, GREEN = "#003366", "#006699", "#4A90A4", "#8A8A8A", "#E8871E", "#C0392B", "#2E8B57"
res, versions, lines = P.run()

# 1. The five steps
fig, ax = plt.subplots(figsize=(10, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
steps = [("1 Load", "net revenue,\nSKU mapping", BLUE), ("2 Consolidate", "monthly volume\nand price", BLUE),
         ("3 Compliance", "as-of join to the\nprice in force", BLUE), ("4 Discounts", "top 5 customers\nbelow suggestion", BLUE),
         ("5 Market", "reading for\nthe report", ORANGE)]
for i, (title, sub, color) in enumerate(steps):
    x = 2 + i * 19.6
    ax.add_patch(FancyBboxPatch((x, 9), 16, 13, boxstyle="round,pad=0.3,rounding_size=1.2", fc=color, ec="none"))
    ax.text(x + 8, 18.6, title, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    ax.text(x + 8, 13.3, sub, ha="center", va="center", color="white", fontsize=8)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + 16.4, 15.5), (x + 19.4, 15.5), arrowstyle="-|>", mutation_scale=10, color=GREY))
ax.text(10, 5.5, "pauses for a person if a SKU\nis missing from the mapping", ha="center", fontsize=7.5, color=RED)
ax.text(41, 3.2, "deterministic: SQL and Python, tested like any code", ha="center", fontsize=8.5, color=BLUE)
ax.text(89.3, 3.2, "language model", ha="center", fontsize=8.5, color=ORANGE)
ax.text(50, 27.5, "Five steps, one model call", ha="center", fontweight="bold", fontsize=11)
fig.savefig(os.path.join(out, "fig1_steps.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Price versions for one product, and what each join compares an invoice line against
sku = "ETH-HYD-M3"
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
starts = [d for d, _ in versions[sku]] + [P.DAYS]
for (d0, p), d1 in zip(versions[sku], starts[1:]):
    ax.hlines(p, d0, d1, color=BLUE, lw=2.2)
latest = versions[sku][-1][1]
ax.axhline(latest, color=ORANGE, ls="--", lw=1.2)
ax.text(2, latest + 15, "latest suggested price", color=ORANGE, fontsize=8)
pts = [l for l in lines if l["sku"] == sku]
ax.scatter([l["day"] for l in pts], [l["unit_price"] for l in pts], s=6, color=GREY, alpha=0.6, label="invoice lines")
ax.set_xlabel("Day"); ax.set_ylabel("Price per m3"); ax.set_title("Suggested price in force (blue) and invoices")
ax.legend(loc="lower right", fontsize=8, frameon=False)
labels = ["As-of join", "Latest price"]
tp = [res["asof"]["flagged"] - res["asof"]["false_alarms"], res["latest"]["flagged"] - res["latest"]["false_alarms"]]
fa = [res["asof"]["false_alarms"], res["latest"]["false_alarms"]]
miss = [res["asof"]["missed"], res["latest"]["missed"]]
y = np.arange(2)
ax2.barh(y, tp, color=GREEN, label="real discount, flagged")
ax2.barh(y, fa, left=tp, color=RED, label="false alarm")
ax2.barh(y, miss, left=np.array(tp) + np.array(fa), color=GREY, label="real discount, missed")
for i in range(2):
    ax2.text(tp[i] + fa[i] + miss[i] + 10, i, f"{tp[i] + fa[i]} flagged", va="center", fontsize=8)
ax2.set_yticks(y, labels); ax2.invert_yaxis(); ax2.set_xlim(0, 1150)
ax2.set_xlabel(f"Invoice lines (of {res['lines']})"); ax2.set_title("Lines sold >2% below suggestion")
ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_asof.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
