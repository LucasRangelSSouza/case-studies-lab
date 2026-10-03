"""Daily accessibility scans that open one ticket per app, not one per finding per day.

Synthetic store: 30 front-end apps whose components carry CSS handles in the Store Framework pattern
`vendor-app-major-x-handle`. Each day a scan (Playwright + axe-core in the real system) reports violations; scans
are noisy, so a page or a finding can be missed one day and seen again the next. Releases add new violations; fixes
merged by people remove them.

Three ways to turn scan results into tickets:
  per_finding   one ticket per violation per scan (what a cron + webhook does by default)
  per_scan      deduplicated inside a scan, but every run opens its tickets again
  fingerprint   fingerprint = app + axe rule + CSS handle; one open ticket per app; new fingerprints are appended

It also resolves which repository holds the app from each repository's manifest.json, the way the coding agent does.

  python a11y_agentic/scan_dedupe.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

RNG = np.random.default_rng(13)
HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ["image-alt", "color-contrast", "link-name", "button-name", "label", "heading-order", "aria-allowed-attr"]
APPS = [f"store.app-{i:02d}" for i in range(26)] + ["vtex.minicart", "vtex.search-result", "vtex.product-summary", "vtex.menu"]
DAYS = 21


def handle(app, name):
    vendor, app_name = app.split(".")
    return f"{vendor}-{app_name}-0-x-{name}"


def parse_handle(css_handle):
    """`vendor-app-major-x-handle` -> (vendor, app). App names can contain hyphens, so split on the version marker."""
    m = re.match(r"^([a-z0-9]+)-(.+)-\d+-x-.+$", css_handle)
    return (m.group(1), m.group(2)) if m else (None, None)


def new_violations(n):
    out = set()
    while len(out) < n:
        app = APPS[RNG.integers(len(APPS))]
        out.add((app, RULES[RNG.integers(len(RULES))], handle(app, f"el{RNG.integers(40)}")))
    return out


def simulate():
    open_v = new_violations(120)
    tickets = {"per_finding": [], "per_scan": [], "fingerprint": []}
    open_cards = {}  # app -> set of fingerprints
    daily = []
    for day in range(DAYS):
        if day % 7 == 3:  # weekly release adds violations
            open_v |= new_violations(int(RNG.integers(8, 16)))
        if day >= 7:  # people start fixing after the first week: a few cards closed per day
            for app in list(open_cards)[: 2]:
                open_v = {v for v in open_v if v[0] != app}
                open_cards.pop(app)
        seen = {v for v in sorted(open_v) if RNG.random() < 0.8}  # a noisy scan (sorted: sets have no stable order)
        tickets["per_finding"] += [(day, v) for v in seen]
        tickets["per_scan"] += [(day, v) for v in {(a, r) for a, r, _ in seen}]
        created = appended = 0
        for app, rule, css in sorted(seen):
            fp = f"{app}|{rule}|{css}"
            if app not in open_cards:
                open_cards[app] = {fp}
                tickets["fingerprint"].append((day, app))
                created += 1
            elif fp not in open_cards[app]:
                open_cards[app].add(fp)
                appended += 1
        daily.append({"day": day, "open_violations": len(open_v), "seen": len(seen),
                      "per_finding_total": len(tickets["per_finding"]), "per_scan_total": len(tickets["per_scan"]),
                      "fingerprint_total": len(tickets["fingerprint"]), "fingerprint_open": len(open_cards),
                      "created": created, "appended": appended})
    return daily


def resolve_repo(app, repos):
    """The card carries only `suggested_app`; the agent finds the repo whose manifest matches vendor and name."""
    vendor, name = app.split(".")
    for repo, manifest in repos.items():
        if manifest.get("vendor") == vendor and manifest.get("name") == name:
            return repo
    return None  # official apps are not forked: the fix goes to a theme override, or the agent stops


if __name__ == "__main__":
    daily = simulate()
    repos = {f"storefront-{i}": {"vendor": "store", "name": f"app-{i:02d}"} for i in range(26)}
    examples = {h: parse_handle(h) for h in ["store-app-07-0-x-paragraph", "vtex-product-summary-2-x-nameContainer"]}
    out = {"days": DAYS, "final": daily[-1], "daily": daily, "parse_examples": examples,
           "resolve_examples": {a: resolve_repo(a, repos) for a in ["store.app-07", "vtex.minicart"]}}
    print(json.dumps({k: v for k, v in out.items() if k != "daily"}, indent=1))
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
