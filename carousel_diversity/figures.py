"""Figures for the carousel case: what the visible slots contain, and conversion against variety as the cap moves."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
import simulate as S

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
data = S.sessions(60_000); train_s, test_s = data[:45_000], data[45_000:]
item_ctx, cat_ctx = S.train(train_s)

# 1. composition of the 10 visible slots, averaged over test sessions
cats = list(S.CATEGORIES)
def composition(rec):
    c = Counter()
    for ctx, _ in test_s:
        for i in rec(ctx)[:10]:
            c[dict(S.PRODUCTS)[i]] += 1
    return np.array([c[k] / len(test_s) for k in cats])
conv = composition(lambda ctx: S.conversion_model(item_ctx, ctx))
two = composition(lambda ctx: S.two_stage_model(item_ctx, cat_ctx, ctx))
fig, ax = plt.subplots(figsize=(9.5, 3.2))
colors = plt.cm.tab10(np.arange(len(cats)))
for row, (name, comp) in enumerate((("Trained for conversion", conv), ("Two-stage, at most 3 per category", two))):
    left = 0
    for k, v, c in zip(cats, comp, colors):
        ax.barh(row, v, left=left, color=c, edgecolor="white", label=k if row == 0 else None)
        if v > 0.6:
            ax.text(left + v / 2, row, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)
        left += v
ax.set_yticks([0, 1], ["Trained for conversion", "Two-stage, max 3 per category"]); ax.invert_yaxis()
ax.set_xlabel("Average number of visible slots (of 10) per category"); ax.set_xlim(0, 10)
ax.set_title("What the customer sees in the first 10 slots")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=8, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_slots.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. cap sweep: conversion and variety
caps = [1, 2, 3, 4, 5, 6, 8, 10]
convs, varieties, others = [], [], []
for cap in caps:
    rec = lambda ctx, cap=cap: S.two_stage_model(item_ctx, cat_ctx, ctx, cap=cap)
    r = S.evaluate(rec, test_s)
    convs.append(r["conversion_at_10"] * 100); varieties.append(r["categories_at_10"]); others.append(S.non_popcorn_hits(rec, test_s))
fig, ax1 = plt.subplots(figsize=(9, 3.8))
ax1.plot(caps, convs, marker="o", color="#006699", label="conversion@10 (%)")
ax1.set_xlabel("Maximum items per category in the 10 visible slots"); ax1.set_ylabel("Sessions that bought a shown item (%)", color="#006699")
ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(caps, varieties, marker="s", color="#E8871E", label="categories shown")
ax2.set_ylabel("Distinct categories in the 10 slots", color="#E8871E")
ax1.axvline(3, color="grey", ls="--", lw=1); ax1.text(3.1, min(convs) + 1, "cap = 3", fontsize=9, color="grey")
ax1.set_title("Every slot taken from popcorn buys conversion and costs variety")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_cap_tradeoff.png"), dpi=200, facecolor="white"); plt.close(fig)
print({"caps": caps, "conversion": convs, "categories": varieties, "non_popcorn_hits": others})
