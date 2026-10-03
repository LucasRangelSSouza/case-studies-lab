"""Figures for the monitoring case: mean against P90 latency, and the zero-traffic rule with and without a quiet window."""
import json
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
                     "axes.grid": True, "grid.color": "#E5E7EB"})
BLUE, ORANGE, RED, GREY, GREEN = "#006699", "#E8871E", "#C0392B", "#8A8A8A", "#2E8B57"
rows = S.simulate()
rules = {r["name"]: r for r in json.load(open(os.path.join(os.path.dirname(__file__), "rules.json")))}


def rolling(metric, window):
    out = []
    for m in range(len(rows)):
        v = S.window_values(rows, metric, m, window)
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


# 1. Day 1, 12:00 to 18:00: the slow tail moves P90 and barely moves the mean
hours = np.arange(len(rows)) / 60
mean15, p90 = rolling("mean_ms", 15), rolling("p90_ms", 15)
sel = (hours >= 12) & (hours < 18)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(15, 17, color=ORANGE, alpha=0.12, label="12% of calls 4-8x slower")
ax.plot(hours[sel], p90[sel], color=RED, label="P90 latency (15 min)")
ax.plot(hours[sel], mean15[sel], color=BLUE, label="Mean latency (15 min)")
ax.axhline(300, color=RED, ls="--", lw=1); ax.text(12.05, 308, "P90 alert at 300 ms", color=RED, fontsize=8)
ax.axhline(200, color=BLUE, ls="--", lw=1); ax.text(12.05, 186, "mean alert at 200 ms", color=BLUE, fontsize=8, va="top")
ax.set_xlabel("Hour of day 1"); ax.set_ylabel("Latency (ms)"); ax.set_ylim(0, 650)
ax.set_title("A slow tail that the mean never reports")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_mean_vs_p90.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)

# 2. Two days of traffic and when each zero-traffic rule notifies
rpm = np.array([r["rpm"] for r in rows])
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
ax.plot(hours, rpm, color=BLUE, lw=0.6)
ax.axvspan(19 + 10 / 60, 20 + 40 / 60, color=RED, alpha=0.15)
ax.text(20, 200, "app stops calling\n(no errors)", color=RED, ha="center", fontsize=8)
ax.axvspan(24 + 21, 24 + 21 + 20 / 60, color=ORANGE, alpha=0.3)
ax.text(45.3, 330, "retry loop", color=ORANGE, fontsize=8)
for d in (0, 24):
    ax.axvspan(d, d + 10, color=GREY, alpha=0.08)
ax.text(0.4, 60, "overnight:\nnear zero is normal", color=GREY, ha="left", fontsize=8)
ax.set_ylabel("Requests per minute"); ax.set_title("Two days of traffic and the zero-traffic alerts")
names = ["RPM == 0, no quiet window", "RPM == 0, quiet 00:00-10:00"]
for y, name in enumerate(names):
    spans = S.evaluate(rows, rules[name])
    for a, b in spans:
        ax2.barh(y, (b - a + 1) / 60, left=a / 60, height=0.5, color=RED)
    ax2.text(48.3, y, f"{len(spans)} notification" + ("s" if len(spans) != 1 else ""), va="center", fontsize=8)
ax2.set_yticks([0, 1], ["no quiet window", "quiet 00:00-10:00"]); ax2.invert_yaxis(); ax2.set_ylim(1.6, -0.6)
ax2.set_xlabel("Hours since start"); ax2.set_xlim(0, 53); ax2.set_xticks(range(0, 49, 6))
ax2.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_zero_traffic.png"), dpi=200, facecolor="white", bbox_inches="tight"); plt.close(fig)
print("ok")
