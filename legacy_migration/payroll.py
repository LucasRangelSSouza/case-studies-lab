"""The generate, execute, diagnose, fix loop of a legacy migration, on a payroll rule.

The legacy program is a COBOL payroll step. Its arithmetic follows COBOL rules: fields are fixed-point decimals
(PIC 9(7)V99), intermediate results are truncated to the receiving field's scale, and ROUNDED means round half up.
Here `legacy()` reproduces those semantics with Python's decimal module and acts as the oracle: in a real migration the
oracle is the legacy program's own output on the same inputs.

Each candidate below is what a translation looks like after one turn of the loop; each turn fixes the mismatches the
validator reported, and the lesson goes into the spec for that language pair so the next program starts with it.

  python legacy_migration/payroll.py
"""
from __future__ import annotations

import json
import os
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = Decimal("0.01")

# COBOL source of the rule (illustrative):
#   COMPUTE WS-OVERTIME = WS-HOURLY * WS-OT-HOURS * 1.5.
#   COMPUTE WS-GROSS    = WS-BASE + WS-OVERTIME.
#   COMPUTE WS-INSS     = WS-GROSS * WS-INSS-RATE.
#   COMPUTE WS-NET ROUNDED = WS-GROSS - WS-INSS - WS-UNION-FEE.


def legacy(base, hourly, ot_hours, rate, fee):
    t = lambda x: x.quantize(C, rounding=ROUND_DOWN)  # store into PIC 9(7)V99: truncate
    overtime = t(hourly * ot_hours * Decimal("1.5"))
    gross = t(base + overtime)
    inss = t(gross * rate)
    return {"overtime": overtime, "gross": gross, "inss": inss,
            "net": (gross - inss - fee).quantize(C, rounding=ROUND_HALF_UP)}


def v1_float(base, hourly, ot_hours, rate, fee):
    """First translation: floats, two decimals shown with round()."""
    b, h, o, r, f = map(float, (base, hourly, ot_hours, rate, fee))
    overtime = h * o * 1.5
    gross = b + overtime
    inss = gross * r
    d = lambda x: Decimal(str(round(x, 2)))
    return {"overtime": d(overtime), "gross": d(gross), "inss": d(inss), "net": d(gross - inss - f)}


def v2_decimal_default(base, hourly, ot_hours, rate, fee):
    """After turn 1 (spec: money is Decimal, never float): Decimal, quantized with Python's default half-even."""
    q = lambda x: x.quantize(C, rounding=ROUND_HALF_EVEN)
    overtime = q(hourly * ot_hours * Decimal("1.5"))
    gross = q(base + overtime)
    inss = q(gross * rate)
    return {"overtime": overtime, "gross": gross, "inss": inss, "net": q(gross - inss - fee)}


def v3_half_up(base, hourly, ot_hours, rate, fee):
    """After turn 2 (spec: ROUNDED is half up): half up everywhere, including where COBOL truncates."""
    q = lambda x: x.quantize(C, rounding=ROUND_HALF_UP)
    overtime = q(hourly * ot_hours * Decimal("1.5"))
    gross = q(base + overtime)
    inss = q(gross * rate)
    return {"overtime": overtime, "gross": gross, "inss": inss, "net": q(gross - inss - fee)}


def v4_truncate_intermediates(base, hourly, ot_hours, rate, fee):
    """After turn 3 (spec: a COMPUTE without ROUNDED truncates to the field's scale)."""
    t = lambda x: x.quantize(C, rounding=ROUND_DOWN)
    overtime = t(hourly * ot_hours * Decimal("1.5"))
    gross = t(base + overtime)
    inss = t(gross * rate)
    return {"overtime": overtime, "gross": gross, "inss": inss,
            "net": (gross - inss - fee).quantize(C, rounding=ROUND_HALF_UP)}


CANDIDATES = [("1: float", v1_float), ("2: Decimal, default rounding", v2_decimal_default),
              ("3: Decimal, half up", v3_half_up), ("4: truncate intermediates", v4_truncate_intermediates)]
SPEC_LESSONS = ["Money and rates are Decimal, never float.",
                "COBOL ROUNDED rounds half up; Python's default is half-even.",
                "Every COMPUTE stores into a fixed-scale field: truncate intermediates unless the statement says ROUNDED."]


def employees(n=20000, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        base = Decimal(str(round(float(rng.uniform(1500, 15000)), 2)))
        hourly = Decimal(str(round(float(rng.uniform(8, 90)), 2)))
        ot = Decimal(str(round(float(rng.choice([0, 0, 2.5, 4, 7.25, 10, 13.5])), 2)))
        rate = Decimal(str(rng.choice(["0.075", "0.09", "0.12", "0.14"])))
        fee = Decimal(str(round(float(rng.choice([0, 12.5, 23.9, 31.17])), 2)))
        rows.append((base, hourly, ot, rate, fee))
    return rows


def run():
    """The validator compares every field, not only the final one, so a mismatch points at the statement that caused it."""
    rows = employees()
    expected = [legacy(*r) for r in rows]
    out = []
    for name, fn in CANDIDATES:
        got = [fn(*r) for r in rows]
        fields = {f: sum(g[f] != e[f] for g, e in zip(got, expected)) for f in ("overtime", "gross", "inss", "net")}
        net_diff = [g["net"] - e["net"] for g, e in zip(got, expected) if g["net"] != e["net"]]
        out.append({"turn": name, "mismatches_by_field": fields,
                    "max_abs_cents": int(max((abs(d) for d in net_diff), default=0) * 100),
                    "net_total_cents": int(sum(net_diff, Decimal(0)) * 100)})
    return {"employees": len(rows), "turns": out, "spec_lessons": SPEC_LESSONS}


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
