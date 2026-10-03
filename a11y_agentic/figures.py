"""Figures for the accessibility case: the three moments, and tickets opened under each dedupe strategy."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
import scan_dedupe as S

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
NAVY, BLUE, TEAL, GREY, ORANGE, RED, GREEN = "#003366", "#006699", "#4A90A4", "#8A8A8A", "#E8871E", "#C0392B", "#2E8B57"

# 1. Three moments, three triggers
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")
rows = [("Discover", "daily schedule", "Playwright + axe-core scan the live store", "one ticket per app,\ndeduplicated", BLUE),
        ("Prevent", "pull request opened", "review with an accessibility checklist on the diff", "comments; a new gap\ndoesn't merge unnoticed", TEAL),
        ("Fix", "a person moves the ticket", "coding agent finds the repo, fixes, checks visually", "pull request for\nhuman review", ORANGE)]
for i, (name, trigger, engine, output, color) in enumerate(rows):
    y = 27 - i * 11.5
    ax.add_patch(FancyBboxPatch((1, y), 13, 8, boxstyle="round,pad=0.3,rounding_size=1", fc=color, ec="none"))
    ax.text(7.5, y + 4, name, ha="center", va="center", color="white", fontweight="bold")
    ax.text(17, y + 4, trigger, va="center", fontsize=9, color="#333")
    ax.text(40, y + 4, engine, va="center", fontsize=9, color="#333")
    ax.text(82, y + 4, output, va="center", fontsize=8.5, color="#333")
for x, t in ((17, "Trigger"), (40, "Engine"), (82, "Output")):
    ax.text(x, 37, t, fontweight="bold", fontsize=9, color=GREY)
ax.text(50, -1.5, "Merge is always human. The scan never starts the agent; only a person moving a ticket does.",
        ha="center", fontsize=8.5, color=RED)
fig.savefig(os.path.join(out, "fig1_three_moments.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Cumulative tickets over three weeks
d = S.simulate()
days = [r["day"] + 1 for r in d]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax.plot(days, [r["per_finding_total"] for r in d], color=RED, label="one per finding per scan")
ax.plot(days, [r["per_scan_total"] for r in d], color=ORANGE, label="deduplicated within a scan")
ax.plot(days, [r["fingerprint_total"] for r in d], color=GREEN, label="fingerprint, one per app")
ax.set_yscale("log"); ax.set_xlabel("Day"); ax.set_xticks(range(1, 22, 4)); ax2.set_xticks(range(1, 22, 4)); ax.set_ylabel("Tickets opened (cumulative, log)")
ax.set_title("Tickets from a daily scan"); ax.legend(fontsize=8, frameon=False); ax.grid(color="#E5E7EB")
for r, c in ((d[-1]["per_finding_total"], RED), (d[-1]["fingerprint_total"], GREEN)):
    ax.text(21.3, r, f"{r:,}", va="center", fontsize=8, color=c)
ax2.plot(days, [r["open_violations"] for r in d], color=BLUE, label="violations in the store")
ax2.plot(days, [r["fingerprint_open"] for r in d], color=GREEN, label="open tickets (fingerprint)")
for rel in (4, 11, 18):
    ax2.axvline(rel, color=GREY, ls=":", lw=1)
ax2.text(4.2, 112, "releases", color=GREY, fontsize=8)
ax2.set_xlabel("Day"); ax2.set_ylabel("Count"); ax2.set_title("Violations and the tickets that track them")
ax2.legend(fontsize=8, frameon=False, loc="upper right"); ax2.grid(color="#E5E7EB")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_tickets.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
