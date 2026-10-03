"""Will the service backlog hit this month's revenue target? A forecast from the orders in progress.

Synthetic data with the shape of the real problem: a telecom operator delivers service orders (fibre links, IP
services, cross-connects) whose monthly revenue counts towards a target only once the service is activated. Two tables:

  completed   past orders: product, country, revenue, days from start to activation
  wip         orders in progress today: product, country, revenue, days since start

For each order in progress, the chance it activates in the next `horizon` days is read from the completed orders of the
same product that were still open at the same age (an empirical conditional distribution, no model training).
A Monte Carlo over those chances gives the distribution of revenue delivered this month, the probability of hitting
the target, and how large the backlog would have to be to hit it. A priority score (country and product weight of
evidence for high revenue, plus urgency) orders the backlog.

  python backlog_forecast/forecast.py
"""
from __future__ import annotations

import json
import os

import numpy as np

RNG = np.random.default_rng(8)
HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = {"Cross-Connect": (54, 1.6), "Cloud Connect": (59, 2.2), "Broadband": (65, 0.9), "Ethernet": (81, 3.1),
            "IP/MPLS": (88, 3.6), "Dark Fiber": (121, 9.5), "Wavelength": (142, 12.0)}  # mean days, revenue k USD/month
COUNTRIES = {"BR": 1.0, "CO": 1.1, "AR": 1.15, "MX": 0.95, "CL": 1.2, "PE": 1.25}  # delivery time multiplier
TARGET = 1000.0  # thousands USD of new monthly revenue activated this month


def order(product, country):
    mean, rev = PRODUCTS[product]
    days = RNG.gamma(3.0, mean * COUNTRIES[country] / 3.0)
    revenue = rev * RNG.lognormal(0, 0.6)
    return days, revenue


def build(n_completed=6000, n_wip=900):
    prods, ctrs = list(PRODUCTS), list(COUNTRIES)
    pw = np.array([0.22, 0.12, 0.2, 0.18, 0.15, 0.06, 0.07])
    completed = []
    for _ in range(n_completed):
        p, c = RNG.choice(prods, p=pw), RNG.choice(ctrs)
        d, r = order(p, c)
        completed.append({"product": p, "country": c, "days": d, "revenue": r})
    wip = []
    for _ in range(n_wip):
        p, c = RNG.choice(prods, p=pw), RNG.choice(ctrs)
        d, r = order(p, c)
        age = RNG.uniform(0, d)  # a snapshot: each open order is somewhere along its own duration
        wip.append({"product": p, "country": c, "age": age, "revenue": r})
    return completed, wip


def p_activate(completed, product, country, age, horizon):
    """Share of past orders of this product (and country, if enough) still open at `age` that activated by age+horizon."""
    def share(rows):
        open_at = [r["days"] for r in rows if r["days"] > age]
        return (sum(d <= age + horizon for d in open_at) / len(open_at), len(open_at)) if open_at else (0.0, 0)
    same = [r for r in completed if r["product"] == product and r["country"] == country]
    p, n = share(same)
    if n < 30:  # too few: fall back to the product across countries
        p, n = share([r for r in completed if r["product"] == product])
    return p


def simulate(probs, revenue, runs=4000):
    draws = RNG.random((runs, len(probs))) < probs
    return draws.astype(float) @ revenue


def woe(completed, field, threshold):
    high = np.array([r["revenue"] > threshold for r in completed])
    tot_h, tot_l = high.sum(), (~high).sum()
    out = {}
    for v in {r[field] for r in completed}:
        m = np.array([r[field] == v for r in completed])
        out[v] = float(np.log(((high & m).sum() + 0.5) / tot_h) - np.log(((~high & m).sum() + 0.5) / tot_l))
    return out


def priority(wip, completed, threshold=5.0):
    """30% country, 20% product (weight of evidence for high revenue), 50% urgency (less time left, higher)."""
    wc, wp = woe(completed, "country", threshold), woe(completed, "product", threshold)
    mean_days = {p: np.mean([r["days"] for r in completed if r["product"] == p]) for p in PRODUCTS}
    left = np.array([max(mean_days[o["product"]] - o["age"], 0) for o in wip])
    urgency = 1 - left / left.max()
    norm = lambda d, k: (d[k] - min(d.values())) / (max(d.values()) - min(d.values()))
    return np.array([0.3 * norm(wc, o["country"]) + 0.2 * norm(wp, o["product"]) + 0.5 * u for o, u in zip(wip, urgency)])


def run(horizon=30):
    completed, wip = build()
    probs = np.array([p_activate(completed, o["product"], o["country"], o["age"], horizon) for o in wip])
    rev = np.array([o["revenue"] for o in wip])
    dist = simulate(probs, rev)
    scale = np.linspace(0.8, 1.6, 17)
    needed = [{"wip_orders": int(len(wip) * s), "expected": float(dist.mean() * s),
               "p_hit": float((simulate(probs, rev) * s >= TARGET).mean())} for s in scale]
    score = priority(wip, completed)
    order_idx = np.argsort(-score)
    by_product = {p: float(sum(probs[i] * rev[i] for i in range(len(wip)) if wip[i]["product"] == p)) for p in PRODUCTS}
    return {"wip_orders": len(wip), "wip_revenue": round(float(rev.sum()), 1), "expected": round(float(dist.mean()), 1),
            "p10": round(float(np.percentile(dist, 10)), 1), "p90": round(float(np.percentile(dist, 90)), 1),
            "p_hit_target": round(float((dist >= TARGET).mean()), 3), "target": TARGET,
            "expected_by_product": {k: round(v, 1) for k, v in by_product.items()},
            "wip_needed": needed,
            "top10_priority": [{**{k: (round(v, 1) if isinstance(v, float) else v) for k, v in wip[i].items()},
                                "p_activate_30d": round(float(probs[i]), 2), "score": round(float(score[i]), 3)} for i in order_idx[:10]]}, dist


if __name__ == "__main__":
    out, _ = run()
    print(json.dumps({k: v for k, v in out.items() if k != "wip_needed"}, indent=1))
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
