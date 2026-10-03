"""Figures for the migration case: the loop, and mismatches per field at each turn."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
import payroll as P

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
NAVY, BLUE, TEAL, GREY, ORANGE, GREEN = "#003366", "#006699", "#4A90A4", "#8A8A8A", "#E8871E", "#2E8B57"

# 1. The loop around one spec
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
steps = [("Scan", "inventory the\nlegacy code"), ("Transpile", "generate code\nfrom the spec"), ("Validate", "legacy vs new,\nfield by field"),
         ("QA", "coverage and\nregression gates"), ("Homologate", "business\nsign-off")]
for i, (t, sub) in enumerate(steps):
    x = 1 + i * 20
    ax.add_patch(FancyBboxPatch((x, 13), 16, 11, boxstyle="round,pad=0.3,rounding_size=1", fc=BLUE if i else GREY, ec="none"))
    ax.text(x + 8, 21, t, ha="center", color="white", fontweight="bold", fontsize=9.5)
    ax.text(x + 8, 16.2, sub, ha="center", va="center", color="white", fontsize=8)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + 16.4, 18.5), (x + 19.6, 18.5), arrowstyle="-|>", mutation_scale=9, color=GREY))
ax.add_patch(FancyBboxPatch((21, 1), 56, 6.5, boxstyle="round,pad=0.3,rounding_size=1", fc=ORANGE, ec="none"))
ax.text(49, 4.2, "One spec per language pair: rules learned from validated fixes", ha="center", va="center", color="white", fontsize=9)
for x in (29, 49, 69, 89):
    ax.add_patch(FancyArrowPatch((x, 12.6), (min(x, 76), 8), arrowstyle="<|-|>", mutation_scale=8, color=ORANGE, lw=1))
ax.text(50, 31.5, "Generate, execute, diagnose, fix: every step reads the same spec", ha="center", fontweight="bold", fontsize=11)
fig.savefig(os.path.join(out, "fig1_loop.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Mismatches by field and turn
res = P.run()
fields = ["overtime", "gross", "inss", "net"]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
x = np.arange(len(res["turns"]))
for j, (f, c) in enumerate(zip(fields, [NAVY, BLUE, TEAL, ORANGE])):
    vals = [t["mismatches_by_field"][f] for t in res["turns"]]
    ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, color=c, label=f)
ax.set_xticks(x, [t["turn"].replace(": ", ":\n") for t in res["turns"]])
ax.set_ylabel(f"Employees with a different value\n(of {res['employees']:,})")
ax.set_title("Mismatches per field after each turn of the loop")
ax.legend(title="field (in statement order)", fontsize=8, title_fontsize=8, frameon=False, ncol=4, loc="upper right")
ax.set_ylim(0, 12500); ax.grid(axis="y", color="#E5E7EB"); ax.set_axisbelow(True)
ax.text(3, 400, "0", ha="center", color=GREEN, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_mismatches.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
