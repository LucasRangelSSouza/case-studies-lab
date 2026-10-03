"""Figures for the backlog forecast: this month's revenue distribution against the target, and backlog size needed."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import forecast as F

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#E5E7EB", "axes.axisbelow": True})
BLUE, ORANGE, RED, GREY, GREEN = "#006699", "#E8871E", "#C0392B", "#8A8A8A", "#2E8B57"
res, dist = F.run()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
bins = np.linspace(dist.min(), max(dist.max(), F.TARGET * 1.05), 45)
ax.hist(dist[dist < F.TARGET], bins=bins, color=BLUE, label="misses the target")
ax.hist(dist[dist >= F.TARGET], bins=bins, color=GREEN, label="hits the target")
ax.axvline(F.TARGET, color=RED, lw=1.5); ax.text(F.TARGET + 5, ax.get_ylim()[1] * 0.9, "target", color=RED, fontsize=8)
ax.set_xlabel("New monthly revenue activated in the next 30 days (k USD)"); ax.set_ylabel("Simulations (of 4,000)")
ax.set_title(f"Current backlog: {res['p_hit_target']:.0%} chance of the target")
ax.legend(fontsize=8, frameon=False, loc="upper left")

n = [x["wip_orders"] for x in res["wip_needed"]]
p = [100 * x["p_hit"] for x in res["wip_needed"]]
ax2.plot(n, p, color=BLUE, marker="o", ms=3.5)
ax2.axvline(res["wip_orders"], color=GREY, ls="--", lw=1); ax2.text(res["wip_orders"] - 8, 50, "today", color=GREY, fontsize=8, ha="right")
ax2.axhline(80, color=ORANGE, ls=":", lw=1); ax2.text(725, 82, "80%", color=ORANGE, fontsize=8)
ax2.set_xlabel("Orders in progress (same product mix and ages)"); ax2.set_ylabel("Chance of hitting the target (%)")
ax2.set_title("How large the backlog would need to be")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_forecast.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Expected revenue this month by product: where the month comes from
items = sorted(res["expected_by_product"].items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
for i, (_, v) in enumerate(items):
    ax.text(v + 3, i, f"{v:.0f}", va="center", fontsize=8)
ax.set_xlabel("Expected revenue activated in 30 days (k USD)"); ax.set_title("Where this month's revenue comes from")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_by_product.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
