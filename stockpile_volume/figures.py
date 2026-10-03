"""Figures for the stockpile case, written to the folder given as the first argument."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})

res, pts, piles, labels = P.run("hdbscan", False)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
sub = pts[P.RNG.choice(len(pts), 12000, replace=False)]
axes[0].scatter(sub[:, 0], sub[:, 1], c=sub[:, 2], s=1, cmap="viridis")
axes[0].set_title("Input: synthetic point cloud (colour = height)")
pal = plt.cm.tab10
for lab in sorted(set(labels)):
    m = labels == lab
    axes[1].scatter(piles[m, 0], piles[m, 1], s=1, color="#BDBDBD" if lab == -1 else pal(lab % 10))
axes[1].set_title("After ground removal and HDBSCAN")
for ax in axes:
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_clusters.png"), dpi=200, facecolor="white"); plt.close(fig)

import json
results = json.load(open(os.path.join(os.path.dirname(__file__), "results.json")))
names = {("hdbscan", False): "HDBSCAN + curved ground", ("dbscan", False): "DBSCAN (fixed radius) + curved ground", ("hdbscan", True): "HDBSCAN + flat ground"}
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(4); w = 0.26
colors = ["#2E8B57", "#C0392B", "#E8871E"]
for i, r in enumerate(results):
    errs = [p["error_pct"] for p in r["piles"]]
    bars = ax.bar(x + (i - 1) * w, np.clip(errs, -100, 100), w, color=colors[i], label=names[(r["method"], r["plane_only"])])
    for b, e in zip(bars, errs):
        if abs(e) > 100:
            ax.text(b.get_x() + b.get_width() / 2, 92, f"+{e:.0f}%", ha="center", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, [f"Pile {i + 1}\n{p['volume_m3']:.0f} m³" for i, p in enumerate(results[0]["piles"])])
ax.set_ylabel("Volume error (%)"); ax.set_ylim(-105, 105)
ax.set_title("Volume error per pile against the known volume")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_volume_error.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# cross-section at y = 40: true ground, plane fit, quadratic fit
xs = np.linspace(0, 100, 300); ys = np.full_like(xs, 30.0)
section = np.column_stack([xs, ys, np.zeros_like(xs)])
pts = P.make_scene()
quad = P.fit_ground_ransac(pts)
g = pts[np.abs(pts[:, 2] - quad(pts)) < 0.15]
coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(g)), g[:, 0], g[:, 1]]), g[:, 2], rcond=None)
plane = np.column_stack([np.ones(len(xs)), xs, ys]) @ coef
surface = P.ground(xs, ys) + sum(P.pile_height(xs, ys, p) for p in P.PILES)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.fill_between(xs, P.ground(xs, ys), surface, color="#D9C9A3", label="pile")
ax.plot(xs, P.ground(xs, ys), color="black", lw=1.5, label="true ground")
ax.plot(xs, quad(section), color="#2E8B57", ls="--", label="quadratic fit")
ax.plot(xs, plane, color="#E8871E", ls=":", lw=2, label="flat-plane fit")
ax.set_xlabel("x (m), section at y = 30 m"); ax.set_ylabel("height (m)")
ax.set_title("A flat ground model turns the yard's curvature into volume")
ax.legend(loc="upper left", ncol=4, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_ground_fit.png"), dpi=200, facecolor="white"); plt.close(fig)
print("figures written to", out)
