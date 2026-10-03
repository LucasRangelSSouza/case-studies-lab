"""Alert rules for ML endpoints, evaluated against two synthetic days of traffic with three planted incidents.

One endpoint serves a consumer app whose traffic follows opening hours (about 10:00 to midnight) and is near zero
overnight. The simulator writes one row per minute with request count and per-request latencies, then plants:

  slow_tail     from 15:00 to 17:00 on day 1, 12% of calls take 4 to 8 times longer (mean barely moves, P90 does)
  silent_drop   from 19:10 to 20:40 on day 1, the app stops calling the endpoint (no errors, just no traffic)
  burst         from 21:00 to 21:20 on day 2, a client retries in a loop and traffic triples

The rules live in rules.json, the same shape as the alert definitions we kept in code:
  metric, aggregation window, operator, threshold, and an optional quiet window that suppresses notifications.

  python endpoint_monitoring/simulate.py     # prints which rules fired and when, writes results.json
"""
from __future__ import annotations

import json
import os

import numpy as np

RNG = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))
MINUTES = 2 * 24 * 60


def expected_rpm(minute_of_day: int) -> float:
    """Opening hours shape: ramps from 10:00, peaks around 20:00, falls to near zero after midnight."""
    h = minute_of_day / 60
    if h < 1 or h >= 24:
        return 1.0
    if h < 10:
        return 0.3
    return 20 + 140 * np.exp(-((h - 20) ** 2) / 8)


def simulate():
    rows = []
    for m in range(MINUTES):
        day, mod = divmod(m, 1440)
        rpm = RNG.poisson(expected_rpm(mod))
        slow_share = 0.0
        if day == 0 and 15 * 60 <= mod < 17 * 60:
            slow_share = 0.12
        if day == 0 and 19 * 60 + 10 <= mod < 20 * 60 + 40:
            rpm = 0
        if day == 1 and 21 * 60 <= mod < 21 * 60 + 20:
            rpm = RNG.poisson(3 * expected_rpm(mod))
        lat = RNG.lognormal(np.log(110), 0.25, rpm)
        slow = RNG.random(rpm) < slow_share
        lat[slow] *= RNG.uniform(4, 8, slow.sum())
        rows.append({"minute": m, "rpm": int(rpm),
                     "mean_ms": float(lat.mean()) if rpm else None,
                     "p90_ms": float(np.percentile(lat, 90)) if rpm else None})
    return rows


def window_values(rows, metric, end, window):
    vals = [r[metric] for r in rows[max(0, end - window + 1):end + 1]]
    return [v for v in vals if v is not None]


def in_quiet(minute, quiet):
    if not quiet:
        return False
    mod = minute % 1440
    start, stop = quiet
    return start <= mod < stop


def evaluate(rows, rule):
    """Return the list of (start, end) minutes when the rule was firing and its notifications were not suppressed.
    A rule fires when the aggregate of `metric` over the last `window` minutes breaks the threshold."""
    ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b, "==": lambda a, b: a == b}
    firing = []
    for m in range(len(rows)):
        vals = window_values(rows, rule["metric"], m, rule["window_min"])
        if rule["metric"] == "rpm":
            value = float(np.mean(vals)) if vals else 0.0
        else:
            value = float(np.mean(vals)) if vals else None
        hit = value is not None and ops[rule["op"]](value, rule["threshold"])
        if hit and not in_quiet(m, rule.get("quiet_minutes_of_day")):
            firing.append(m)
    spans = []
    for m in firing:
        if spans and m == spans[-1][1] + 1:
            spans[-1][1] = m
        else:
            spans.append([m, m])
    return [tuple(s) for s in spans]


def hhmm(minute):
    day, mod = divmod(minute, 1440)
    return f"day {day + 1} {mod // 60:02d}:{mod % 60:02d}"


if __name__ == "__main__":
    rows = simulate()
    rules = json.load(open(os.path.join(HERE, "rules.json")))
    results = {}
    for rule in rules:
        spans = evaluate(rows, rule)
        results[rule["name"]] = [[hhmm(a), hhmm(b), b - a + 1] for a, b in spans]
        print(rule["name"], results[rule["name"]])
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
