"""A grocery recommender that ranks products directly, against one that picks the product type first and the variant second.

Synthetic grocery shop: product types (olive oil, apples, rice...) in subfamilies, each type sold in 2 to 5 variants
(brand, size, pack) with different margins. Customers have favourite types, a usual variant per type that they buy most
of the time, and some change their habits halfway through their history. The last order of each customer is held out.

  item model        ranks SKUs by the customer's purchase counts, smoothed by global popularity
  two-stage model   stage 1 ranks product types by recency-weighted counts (recent orders weigh more, the idea DIN
                    learns with attention); stage 2 picks one variant per type: the customer's usual one, or for a type
                    they never bought, the highest-margin variant among the popular ones

Production used DeepFM and DIN for these stages; counts stand in for them here so the structure is visible and the
script runs in seconds.

  python grocery_two_stage/simulate.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

RNG = np.random.default_rng(5)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TYPES, N_USERS, N_ORDERS, K = 80, 2000, 14, 10


def catalog():
    types = []
    for t in range(N_TYPES):
        n_var = int(RNG.integers(2, 6))
        popularity = RNG.dirichlet(np.ones(n_var) * 2)
        margin = RNG.uniform(0.08, 0.35, n_var)
        types.append({"type": t, "variants": [f"t{t}v{v}" for v in range(n_var)], "pop": popularity, "margin": margin})
    return types


def customers(types):
    base = RNG.dirichlet(np.ones(N_TYPES) * 0.08, N_USERS)  # a few favourite types each
    later = RNG.dirichlet(np.ones(N_TYPES) * 0.08, N_USERS)
    drift = RNG.random(N_USERS) < 0.4                       # 40% change habits halfway
    usual = [[int(RNG.choice(len(t["variants"]), p=t["pop"])) for t in types] for _ in range(N_USERS)]
    return base, later, drift, usual


def orders(types, base, later, drift, usual):
    history = []
    for u in range(N_USERS):
        user_orders = []
        for o in range(N_ORDERS):
            w = later[u] if drift[u] and o >= N_ORDERS // 2 else base[u]
            w = 0.9 * w + 0.1 / N_TYPES
            picked = RNG.choice(N_TYPES, size=int(RNG.integers(4, 9)), replace=False, p=w / w.sum())
            basket = set()
            for t in picked:
                tv = types[t]
                v = usual[u][t] if RNG.random() < 0.7 else int(RNG.choice(len(tv["variants"]), p=tv["pop"]))
                basket.add(tv["variants"][v])
            user_orders.append(basket)
        history.append(user_orders)
    return history


def type_of(sku):
    return int(sku[1:sku.index("v")])


def item_model(train, global_items):
    counts = Counter(i for basket in train for i in basket)
    score = {i: counts.get(i, 0) + 0.01 * global_items[i] / max(global_items.values()) for i in global_items}
    return sorted(score, key=lambda i: -score[i])[:K]


def two_stage_model(train, types, global_types, half_life=3.0):
    n = len(train)
    tscore = defaultdict(float)
    variant_counts = defaultdict(Counter)
    for age, basket in enumerate(reversed(train)):  # age 0 = most recent order
        weight = 0.5 ** (age / half_life)
        for sku in basket:
            tscore[type_of(sku)] += weight
            variant_counts[type_of(sku)][sku] += 1
    top_g = max(global_types.values())
    ranked = sorted(range(N_TYPES), key=lambda t: -(tscore[t] + 0.01 * global_types[t] / top_g))[:K]
    out = []
    for t in ranked:
        if variant_counts[t]:
            out.append(variant_counts[t].most_common(1)[0][0])
        else:  # business rule: among variants at least 80% as popular as the best, take the highest margin
            tv = types[t]
            ok = [v for v in range(len(tv["variants"])) if tv["pop"][v] >= 0.8 * tv["pop"].max()]
            out.append(tv["variants"][max(ok, key=lambda v: tv["margin"][v])])
    return out


def metrics(recs, test, margin_of):
    sku_hits = len(set(recs) & test)
    rec_types = [type_of(s) for s in recs]
    test_types = {type_of(s) for s in test}
    dup = len(rec_types) - len(set(rec_types))
    return {"sku_hits": sku_hits, "type_hits": len(set(rec_types) & test_types), "distinct_types": len(set(rec_types)),
            "lists_with_same_type_twice": int(dup > 0), "duplicate_slots": dup,
            "margin": float(np.mean([margin_of[s] for s in recs]))}


def run():
    types = catalog()
    base, later, drift, usual = customers(types)
    history = orders(types, base, later, drift, usual)
    margin_of = {v: float(t["margin"][i]) for t in types for i, v in enumerate(t["variants"])}
    global_items = Counter(i for h in history for b in h[:-1] for i in b)
    for t in types:
        for v in t["variants"]:
            global_items.setdefault(v, 0)
    global_types = Counter(type_of(i) for h in history for b in h[:-1] for i in b)
    rows = {"item": [], "two_stage": [], "two_stage_no_recency": []}
    drifted = []
    for u in range(N_USERS):
        train, test = history[u][:-1], history[u][-1]
        rows["item"].append(metrics(item_model(train, global_items), test, margin_of))
        rows["two_stage"].append(metrics(two_stage_model(train, types, global_types), test, margin_of))
        rows["two_stage_no_recency"].append(metrics(two_stage_model(train, types, global_types, half_life=1e9), test, margin_of))
        drifted.append(bool(drift[u]))
    drifted = np.array(drifted)
    summary = {}
    for name, rs in rows.items():
        s = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in rs[0]}
        s["type_hits_changed_habits"] = round(float(np.mean([r["type_hits"] for r, d in zip(rs, drifted) if d])), 3)
        s["type_hits_stable"] = round(float(np.mean([r["type_hits"] for r, d in zip(rs, drifted) if not d])), 3)
        summary[name] = s
    return summary, rows


def export(customer_id, recs, types, tenant="demo-shop", slot="home_for_you", generated_at="2026-10-03T00:00:00Z"):
    """The batch output contract: one JSON per tenant and slot, a ranked list per customer, the reason for each item."""
    margin_of = {v: float(t["margin"][i]) for t in types for i, v in enumerate(t["variants"])}
    return {"tenant": tenant, "slot": slot, "generated_at": generated_at, "model_version": "two_stage-v1",
            "customer_id": customer_id,
            "items": [{"rank": r + 1, "sku": sku, "product_type": f"t{type_of(sku)}", "margin": round(margin_of[sku], 3)}
                      for r, sku in enumerate(recs)]}


if __name__ == "__main__":
    summary, _ = run()
    for k, v in summary.items():
        print(k, v)
    json.dump(summary, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    types = catalog()
    sample = [["t3v1"], ["t3v2"], ["t7v0", "t12v1"]]
    json.dump(export("c-0001", two_stage_model(sample, types, Counter({t: 1 for t in range(N_TYPES)})), types),
              open(os.path.join(HERE, "sample_output.json"), "w"), indent=1)
