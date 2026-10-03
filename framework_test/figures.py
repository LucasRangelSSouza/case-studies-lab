"""Figures for the framework comparison: the test architecture, and seconds per answer with pass or fail."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else here
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
NAVY, BLUE, TEAL, LIGHT, GREY = "#003366", "#006699", "#4A90A4", "#E8F0F5", "#8A8A8A"
GREEN, ORANGE, RED = "#2E8B57", "#E8871E", "#C0392B"

# 1. The minimal architecture built in each framework
fig, ax = plt.subplots(figsize=(8.5, 3.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=9.5)
def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=GREY, lw=1.3))
box(2, 15, 16, 10, "User\n(chat)", GREY)
box(30, 15, 20, 10, "Orchestrator\nchooses an agent", NAVY)
box(64, 25, 32, 10, "SQL agent\nreads the HR table", BLUE)
box(64, 4, 32, 10, "General agent\nsmall talk, user facts", TEAL)
arrow((18.8, 20), (29.2, 20)); arrow((50.8, 22), (63.2, 29)); arrow((50.8, 18), (63.2, 10))
ax.text(50, 38.5, "Same design, same model, built three times", ha="center", fontweight="bold", fontsize=11)
fig.savefig(os.path.join(out, "fig1_architecture.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Seconds per answer, coloured by outcome
res = json.load(open(os.path.join(here, "project_results.json")))["frameworks"]
qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
colors = {"pass": GREEN, "partial": ORANGE, "fail": RED}
fig, ax = plt.subplots(figsize=(9, 3.8))
width = 0.26
for j, (fw, rows) in enumerate(res.items()):
    for i, q in enumerate(qs):
        sec, outcome, _ = rows[q]
        x = i + (j - 1) * width
        ax.bar(x, sec, width * 0.92, color=colors[outcome])
        ax.text(x, sec + 2, fw[0] if fw != "AutoGen" else "AG", ha="center", fontsize=7.5, color="#333")
ax.set_xticks(range(5), ["Q1\nuser facts", "Q2\nname?", "Q3\njob?", "Q4\nlongest tenure", "Q5\nand the shortest?"])
ax.set_ylabel("Seconds to answer"); ax.set_ylim(0, 132)
ax.set_title("Five questions, three frameworks (C = CrewAI, A = Agno, AG = AutoGen)")
for k, c in colors.items():
    ax.bar(0, 0, color=c, label=k)
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.grid(axis="y", color="#E5E7EB"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_answers.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
