"""Why a recommender trained for conversion fills the carousel with one category, and what a two-stage model changes.

Synthetic concession-stand sales: 8 categories, 40 products, sessions that buy 1 to 3 items. One category (popcorn)
is in about half of all purchases. Two recommenders return 20 items and the app shows the first 10:

  conversion   ranks products by how often they are bought in this context (store, weekday, hour)
  two-stage    scores categories, then products inside each category, and shows at most 3 per category in the top 10

Offline metrics on held-out sessions:
  conversion@10   share of sessions that bought at least one shown item
  incidence@10    mean number of shown items the session bought
  categories@10   mean number of distinct categories shown
  share of popcorn in the shown items

  python carousel_diversity/simulate.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

RNG = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = {"popcorn": 8, "soda": 6, "water": 3, "candy": 7, "chocolate": 5, "nachos": 4, "hot dog": 3, "ice cream": 4}
CAT_WEIGHT = {"popcorn": 0.48, "soda": 0.18, "water": 0.06, "candy": 0.08, "chocolate": 0.06, "nachos": 0.06, "hot dog": 0.04, "ice cream": 0.04}
PRODUCTS = [(f"{c} {i + 1}", c) for c, n in CATEGORIES.items() for i in range(n)]
CONTEXTS = [(store, day, hour) for store in range(3) for day in range(7) for hour in (14, 17, 20, 22)]


def sessions(n: int):
    out = []
    for _ in range(n):
        ctx = CONTEXTS[RNG.integers(len(CONTEXTS))]
        k = RNG.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        # context shifts tastes a little: late sessions buy more popcorn, afternoon more ice cream
        w = dict(CAT_WEIGHT)
        if ctx[2] >= 20:
            w["popcorn"] *= 1.2
        if ctx[2] == 14:
            w["ice cream"] *= 2.0
        cats = list(w); p = np.array([w[c] for c in cats]); p /= p.sum()
        basket = set()
        for c in RNG.choice(cats, size=k, p=p, replace=False):
            items = [name for name, cat in PRODUCTS if cat == c]
            popularity = np.linspace(2, 1, len(items)); popularity /= popularity.sum()
            basket.add(items[RNG.choice(len(items), p=popularity)])
        out.append((ctx, basket))
    return out


def train(train_sessions):
    item_ctx = defaultdict(Counter)
    cat_ctx = defaultdict(Counter)
    for ctx, basket in train_sessions:
        for item in sorted(basket):  # sets have no stable order; sorting keeps tie-breaks reproducible
            item_ctx[ctx][item] += 1
            cat_ctx[ctx][dict(PRODUCTS)[item]] += 1
    return item_ctx, cat_ctx


def conversion_model(item_ctx, ctx, n=20):
    return [item for item, _ in item_ctx[ctx].most_common(n)]


def two_stage_model(item_ctx, cat_ctx, ctx, n=20, cap=3, shown=10):
    """Stage 1 scores categories, stage 2 scores items inside each category; the final score is their product,
    P(category | context) * P(item | category, context). The visible slots take items in score order, at most `cap`
    from one category; the rest of the 20 follow in plain score order."""
    total = sum(cat_ctx[ctx].values()) or 1
    score = {}
    for item, count in item_ctx[ctx].items():
        cat = dict(PRODUCTS)[item]
        score[item] = (cat_ctx[ctx][cat] / total) * (count / cat_ctx[ctx][cat])
    ordered = sorted(score, key=lambda i: -score[i])
    visible, used = [], Counter()
    for item in ordered:
        cat = dict(PRODUCTS)[item]
        if used[cat] < cap:
            visible.append(item); used[cat] += 1
        if len(visible) == shown:
            break
    rest = [i for i in ordered if i not in visible]
    return (visible + rest)[:n]


def evaluate(recommend, test_sessions, shown=10):
    conv = inc = cats = pop = 0
    for ctx, basket in test_sessions:
        top = recommend(ctx)[:shown]
        hits = len(basket & set(top))
        conv += hits > 0; inc += hits
        cats += len({dict(PRODUCTS)[i] for i in top})
        pop += sum(dict(PRODUCTS)[i] == "popcorn" for i in top)
    n = len(test_sessions)
    return {"conversion_at_10": round(conv / n, 3), "incidence_at_10": round(inc / n, 3),
            "categories_at_10": round(cats / n, 2), "popcorn_share_at_10": round(pop / (n * shown), 3),
            "non_popcorn_hits_per_session": None}


def non_popcorn_hits(recommend, test_sessions, shown=10):
    return round(sum(len({i for i in basket & set(recommend(ctx)[:shown]) if dict(PRODUCTS)[i] != "popcorn"})
                     for ctx, basket in test_sessions) / len(test_sessions), 3)


if __name__ == "__main__":
    data = sessions(60_000)
    train_s, test_s = data[:45_000], data[45_000:]
    item_ctx, cat_ctx = train(train_s)
    models = {"conversion": lambda ctx: conversion_model(item_ctx, ctx),
              "two_stage": lambda ctx: two_stage_model(item_ctx, cat_ctx, ctx)}
    results = {}
    for name, rec in models.items():
        r = evaluate(rec, test_s)
        r["non_popcorn_hits_per_session"] = non_popcorn_hits(rec, test_s)
        results[name] = r
        print(name, r)
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
