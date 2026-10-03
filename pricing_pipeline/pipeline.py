"""A pricing review in five steps, where only the last one needs a language model.

Synthetic data with the shape of the real problem: invoice lines for sugar and ethanol products, a suggested-price table
that the commercial team revises every few weeks (each revision is a new row with its own start date), and a SKU
mapping. The steps:

  1 load       invoice lines to net revenue; pause for a human if a SKU is not in the mapping
  2 consolidate monthly realised volume and price per product
  3 compliance  each invoice line against the suggested price in force on the invoice date (an as-of join);
                flag lines sold more than 2% below it
  4 discounts  the five customers with the largest total discount below suggestion
  5 market     a short reading of the market for the report; the only step that would call an LLM

The comparison in this script: the as-of join against the shortcut of joining every line to the latest suggested
price. It runs with Python's standard library and numpy.

  python pricing_pipeline/pipeline.py
"""
from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(17)
HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = {"SUG-CRISTAL-50": ("Crystal sugar 50 kg", 140.0), "SUG-CRISTAL-1": ("Crystal sugar 1 kg x 10", 36.0),
            "SUG-VHP-BULK": ("VHP sugar bulk (t)", 2600.0), "ETH-HYD-M3": ("Hydrated ethanol (m3)", 2900.0),
            "ETH-ANH-M3": ("Anhydrous ethanol (m3)", 3200.0)}
MAPPING = set(PRODUCTS)
DAYS = 180


def price_versions():
    """Suggested price per product, revised on random days; prices drift up and down with the market."""
    versions = {}
    for sku, (_, base) in PRODUCTS.items():
        days = sorted({0, *RNG.integers(10, DAYS, 7).tolist()})
        level, rows = base, []
        for d in days:
            level *= 1 + RNG.normal(0.01, 0.045)
            rows.append((d, round(level, 2)))
        versions[sku] = rows
    return versions


def invoices(versions, n=2000):
    lines = []
    customers = [f"C{i:03d}" for i in range(60)]
    for i in range(n):
        sku = RNG.choice(list(PRODUCTS)) if RNG.random() > 0.004 else "SUG-ORGANIC-25"  # a new SKU nobody mapped
        day = int(RNG.integers(0, DAYS))
        if sku in versions:
            starts = [d for d, _ in versions[sku]]
            in_force = versions[sku][bisect_right(starts, day) - 1][1]
        else:
            in_force = 150.0
        cust = customers[int(RNG.integers(0, 60) if RNG.random() > 0.3 else RNG.integers(0, 6))]  # a few big buyers
        discount = RNG.normal(0.003, 0.012) + (0.05 if cust in ("C001", "C004") and RNG.random() < 0.5 else 0)
        price = round(in_force * (1 - discount), 2)
        qty = float(RNG.integers(1, 40))
        lines.append({"line": i, "day": day, "sku": sku, "customer": cust, "qty": qty, "unit_price": price,
                      "gross": round(price * qty, 2), "taxes": round(price * qty * 0.12, 2)})
    return lines


def step_load(lines):
    unknown = sorted({l["sku"] for l in lines if l["sku"] not in MAPPING})
    if unknown:
        return {"paused": True, "reason": "SKUs not in the mapping", "skus": unknown}
    for l in lines:
        l["net"] = l["gross"] - l["taxes"]
    return {"paused": False}


def asof_price(versions, sku, day):
    starts = [d for d, _ in versions[sku]]
    return versions[sku][bisect_right(starts, day) - 1][1]


def step_compliance(lines, versions, mode="asof", threshold=0.02):
    flagged = []
    for l in lines:
        ref = asof_price(versions, l["sku"], l["day"]) if mode == "asof" else versions[l["sku"]][-1][1]
        gap = 1 - l["unit_price"] / ref
        if gap > threshold:
            flagged.append({**l, "reference": ref, "gap": round(gap, 4)})
    return flagged


def step_discounts(flagged, top=5):
    total = defaultdict(float)
    for f in flagged:
        total[f["customer"]] += (f["reference"] - f["unit_price"]) * f["qty"]
    return sorted(total.items(), key=lambda kv: -kv[1])[:top]


def run():
    versions = price_versions()
    lines = invoices(versions)
    first = step_load(lines)  # pauses: one SKU is missing from the mapping
    MAPPING.add("SUG-ORGANIC-25")  # the human adds it, and the run resumes
    lines = [l for l in lines if l["sku"] in PRODUCTS]  # (its price history is not in this synthetic table)
    second = step_load(lines)
    truth = {l["line"] for l in lines if 1 - l["unit_price"] / asof_price(versions, l["sku"], l["day"]) > 0.02}
    out = {"lines": len(lines), "first_run": first, "after_mapping_fix": second, "truly_below_suggestion": len(truth)}
    for mode in ("asof", "latest"):
        flagged = step_compliance(lines, versions, mode)
        ids = {f["line"] for f in flagged}
        out[mode] = {"flagged": len(flagged), "false_alarms": len(ids - truth), "missed": len(truth - ids),
                     "top5": [[c, round(v, 2)] for c, v in step_discounts(flagged)]}
    out["steps_that_need_an_llm"] = ["market"]
    return out, versions, lines


if __name__ == "__main__":
    out, _, _ = run()
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
