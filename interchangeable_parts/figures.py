"""Figure for the interchangeable parts case: precision and recall of the top 5 for each retriever."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import search as S

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#E5E7EB", "axes.axisbelow": True})
BLUE, ORANGE, GREEN = "#006699", "#E8871E", "#2E8B57"
res, *_ = S.run()
names = [("text", "Text similarity\nonly"), ("attributes", "Structured fields\nonly"),
         ("hybrid", "Extract + filter +\nrank, always 5"), ("hybrid_cutoff", "Extract + filter +\nrank, within tolerance")]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9.5, 3.8))
for j, (metric, label, c) in enumerate((("precision_at_5", "precision (suggestions that fit)", BLUE),
                                        ("recall_at_5", "recall (substitutes found)", ORANGE),
                                        ("top1_correct", "first suggestion fits", GREEN))):
    vals = [100 * res[k][metric] for k, _ in names]
    ax.bar(x + (j - 1) * 0.26, vals, 0.26, color=c, label=label)
    for xx, v in zip(x + (j - 1) * 0.26, vals):
        ax.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(x, [n for _, n in names]); ax.set_ylabel("% of queries"); ax.set_ylim(0, 112)
ax.set_title(f"Top-5 substitutes for {res['text']['queries']:,} part numbers (synthetic catalogue)")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_retrievers.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
