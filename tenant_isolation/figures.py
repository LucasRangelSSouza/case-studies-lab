"""Figures for the multi-tenant case: the two architectures, and the access matrix with and without the wildcard."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
import isolation as I

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
NAVY, BLUE, TEAL, GREY, ORANGE, RED, GREEN, LIGHT = "#003366", "#006699", "#4A90A4", "#8A8A8A", "#E8871E", "#C0392B", "#2E8B57", "#E8F0F5"


def box(ax, x, y, w, h, text, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=GREY, lw=1.1))


fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2))
for ax in (a1, a2):
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
a1.text(0, 28.5, "Generation 1: a function per tenant behind a router function", fontweight="bold", fontsize=10)
box(a1, 0, 11, 13, 8, "API Gateway\nAPI key,\nusage plan", NAVY)
box(a1, 19, 11, 16, 8, "Router function\nreads tenant ID\nfrom the token", BLUE)
for i, t in enumerate(["Tenant A function", "Tenant B function", "Tenant C function"]):
    box(a1, 43, 21 - i * 9, 20, 6.5, t, TEAL)
    arrow(a1, (35.5, 15), (42.5, 24.2 - i * 9))
box(a1, 70, 11, 28, 8, "Own image, own config,\nagents per domain", GREY)
arrow(a1, (13.5, 15), (18.5, 15)); arrow(a1, (63.5, 15), (69.5, 15))
a2.text(0, 28.5, "Generation 2: a service per tenant in containers, routed by path", fontweight="bold", fontsize=10)
box(a2, 0, 11, 12, 8, "Edge\n(planned: CDN,\nWAF)", GREY)
box(a2, 16, 11, 13, 8, "HTTP API\n+ VPC link", NAVY)
box(a2, 33, 11, 14, 8, "Internal load\nbalancer\n/{tenant}/*", BLUE)
for i, t in enumerate(["Tenant A service", "Tenant B service", "Tenant C service"]):
    box(a2, 52, 21 - i * 9, 18, 6.5, t, TEAL)
    arrow(a2, (47.5, 15), (51.5, 24.2 - i * 9))
box(a2, 75, 18, 23, 8, "Per tenant: image repo,\nbucket, secrets, key, logs", GREEN)
box(a2, 75, 5, 23, 8, "Shared: cache, network,\nendpoints (no tenant data)", GREY)
arrow(a2, (12.5, 15), (15.5, 15)); arrow(a2, (29.5, 15), (32.5, 15)); arrow(a2, (70.5, 15), (74.5, 21))
fig.savefig(os.path.join(out, "fig1_generations.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, (name, fn, title) in zip(axes, (("wildcard", I.shared_policies, 'Secrets with Resource "*"'),
                                        ("scoped", I.scoped_policies, "Every statement scoped to the tenant"))):
    m = np.array(I.matrix(fn()))
    ax.imshow(m, cmap="Reds", vmin=0, vmax=4)
    for i in range(len(I.TENANTS)):
        for j in range(len(I.TENANTS)):
            color = "white" if m[i, j] >= 3 else "#333"
            ax.text(j, i, f"{m[i, j]}/4", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(4), [t.replace("tenant-", "") .upper() for t in I.TENANTS])
    ax.set_yticks(range(4), [t.replace("tenant-", "").upper() for t in I.TENANTS])
    ax.set_xlabel("Resources owned by tenant"); ax.set_ylabel("Role of tenant")
    cross = int(m.sum() - np.trace(m))
    ax.set_title(f"{title}\n{cross} cross-tenant grants", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_access_matrix.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
