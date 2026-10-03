"""Given a part number, find the five parts that can replace it: text similarity alone against filters plus similarity.

Synthetic catalogue with the shape of the problem: parts from several plants describe the same kind of component with
different words, abbreviations and languages, and two parts are interchangeable only if they do the same job and their
critical dimensions match within tolerance. Description text is a poor guide to that: a 'filter, oil, M20x1.5' and a
'filter, oil, M22x1.5' read almost the same and don't fit the same engine.

Three retrievers, each returning a top 5 for every part that has at least one true substitute:
  text       TF-IDF cosine similarity over the free-text description (a stand-in for an embedding model)
  attributes exact match on family and thread, nearest in normalised dimensions
  hybrid     hard filters on the critical attributes (family, thread, material class), then text and dimension
             similarity to rank what passes

Ground truth: same family, same thread, same material class, every dimension within 2%.

  python interchangeable_parts/search.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RNG = np.random.default_rng(31)
HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = {"oil filter": ["filter oil", "oil filter elem", "filtro de oleo", "flt oil"],
            "fuel filter": ["filter fuel", "fuel filter elem", "filtro combustivel", "flt fuel"],
            "hydraulic hose": ["hose hyd", "hydraulic hose assy", "mangueira hidraulica", "hyd hose"],
            "bearing": ["bearing ball", "ball bearing", "rolamento", "brg"],
            "bushing": ["bushing", "bush", "bucha", "sleeve bushing"],
            "seal": ["seal oil", "oil seal", "retentor", "lip seal"]}
THREADS = ["M16x1.5", "M18x1.5", "M20x1.5", "M22x1.5", "3/4-16 UNF", "1-12 UN"]
MATERIALS = {"steel": ["steel", "stl", "aco"], "rubber": ["nbr", "rubber", "borracha"], "bronze": ["bronze", "brz"]}
PLANTS = ["P1", "P2", "P3", "P4", "P5"]


def catalogue(n_designs=700):
    parts = []
    for d in range(n_designs):
        fam = list(FAMILIES)[RNG.integers(len(FAMILIES))]
        thread = THREADS[RNG.integers(len(THREADS))]
        mat = list(MATERIALS)[RNG.integers(len(MATERIALS))]
        dims = RNG.uniform(20, 200, 3).round(1)
        for _ in range(int(RNG.integers(1, 5))):  # the same design, catalogued in several plants
            dd = (dims * RNG.normal(1, 0.004, 3)).round(1)  # small measurement noise
            if RNG.random() < 0.15:  # a near-miss variant: same look, one dimension 5-12% off
                dd[RNG.integers(3)] *= RNG.choice([0.9, 0.95, 1.07, 1.12])
            name = FAMILIES[fam][RNG.integers(4)]
            m = re.match(r"M(\d+)x([\d.]+)", thread)
            thread_txt = thread if not m else RNG.choice([thread, f"M{m.group(1)} x {m.group(2).replace('.', ',')}", thread.upper()])
            words = [name, f"od {dd[0]:.0f}", f"len {dd[1]:.0f}", PLANTS[RNG.integers(5)], f"rev {RNG.integers(1, 9)}"]
            if RNG.random() < 0.8:
                words.append(thread_txt)
            if RNG.random() < 0.7:
                words.append(MATERIALS[mat][RNG.integers(len(MATERIALS[mat]))])
            RNG.shuffle(words[1:])
            known = lambda v: v if RNG.random() < 0.6 else None  # the structured fields are often empty
            parts.append({"pn": f"PN{len(parts):05d}", "design": d, "family": fam, "thread": thread, "material": mat,
                          "dims": dd.tolist(), "text": " ".join(words),
                          "fields": {"thread": known(thread), "material": known(mat),
                                     "dims": dd.tolist() if RNG.random() < 0.7 else None}})
    return parts


def truth(parts):
    out = {}
    for i, a in enumerate(parts):
        out[i] = {j for j, b in enumerate(parts) if j != i and a["family"] == b["family"] and a["thread"] == b["thread"]
                  and a["material"] == b["material"]
                  and all(abs(x - y) / x <= 0.02 for x, y in zip(a["dims"], b["dims"]))}
    return out


FAMILY_OF = {alias: fam for fam, aliases in FAMILIES.items() for alias in aliases}
MATERIAL_OF = {alias: mat for mat, aliases in MATERIALS.items() for alias in aliases}


def extract(part):
    """Normalise what the document says: family from its aliases, thread written in any of its forms, material
    aliases, and the dimensions (structured if present, else the rounded values in the text)."""
    t, f = part["text"], part["fields"]
    family = next(fam for alias, fam in sorted(FAMILY_OF.items(), key=lambda kv: -len(kv[0])) if alias in t)
    thread = f["thread"]
    if not thread:
        m = re.search(r"M(\d+)\s*[xX]\s*([\d]+[.,][\d]+)", t)
        if m:
            thread = f"M{m.group(1)}x{m.group(2).replace(',', '.')}"
        else:
            thread = next((th for th in THREADS if not th.startswith("M") and th in t), None)
    material = f["material"] or next((mat for alias, mat in MATERIAL_OF.items() if re.search(rf"{alias}", t)), None)
    dims = f["dims"] or [float(re.search(r"od (\d+)", t).group(1)), float(re.search(r"len (\d+)", t).group(1)), None]
    return {"family": family, "thread": thread, "material": material, "dims": [x for x in dims if x is not None][:2]}


def evaluate(parts, ranker, gold, k=5):
    """precision: share of the returned suggestions (up to k) that are true substitutes.
    recall: share of the true substitutes (up to k) that appear in the suggestions. An empty answer scores 0 on both."""
    prec, rec, top1 = [], [], []
    for i in gold:
        if not gold[i]:
            continue
        top = ranker(i)[:k]
        hits = len(set(top) & gold[i])
        prec.append(hits / len(top) if top else 0.0)
        rec.append(hits / min(k, len(gold[i])))
        top1.append(bool(top) and top[0] in gold[i])
    return {"queries": len(prec), "precision_at_5": round(float(np.mean(prec)), 3),
            "recall_at_5": round(float(np.mean(rec)), 3), "top1_correct": round(float(np.mean(top1)), 3)}


def run():
    parts = catalogue()
    gold = truth(parts)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([p["text"] for p in parts])
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, -1)
    dims = np.array([p["dims"] for p in parts])
    logd = np.log(dims)

    def text(i):
        return list(np.argsort(-sim[i])[:20])

    def attributes(i):
        """Structured fields only: a part whose field is empty can't be matched on it, so it is left out."""
        a = parts[i]["fields"]
        if not (a["thread"] and a["dims"]):
            return []
        cand = [j for j, b in enumerate(parts) if j != i and parts[j]["family"] == parts[i]["family"]
                and b["fields"]["thread"] == a["thread"] and b["fields"]["dims"]]
        return sorted(cand, key=lambda j: np.abs(np.log(parts[j]["fields"]["dims"]) - np.log(a["dims"])).max())

    ex = [extract(p) for p in parts]

    def hybrid(i):
        """Fill empty fields from the text, filter on every critical attribute that is known on both sides,
        then rank by dimension distance and text similarity."""
        a = ex[i]
        cand = []
        for j, b in enumerate(ex):
            if j == i or b["family"] != a["family"]:
                continue
            if a["thread"] and b["thread"] and a["thread"] != b["thread"]:
                continue
            if a["material"] and b["material"] and a["material"] != b["material"]:
                continue
            cand.append(j)
        def key(j):
            unknown = (not a["thread"] or not ex[j]["thread"]) + (not a["material"] or not ex[j]["material"])
            return np.abs(np.log(ex[j]["dims"]) - np.log(a["dims"])).max() * 10 + unknown * 0.3 - sim[i, j]
        return sorted(cand, key=key)

    def hybrid_cutoff(i):
        """Same ranking, but only suggestions whose dimensions are within tolerance: fewer than five when fewer fit."""
        a = ex[i]
        return [j for j in hybrid(i) if np.abs(np.log(ex[j]["dims"]) - np.log(a["dims"])).max() <= 0.03]

    res = {name: evaluate(parts, fn, gold) for name, fn in (("text", text), ("attributes", attributes), ("hybrid", hybrid),
                                                       ("hybrid_cutoff", hybrid_cutoff))}
    res["catalogue_parts"] = len(parts)
    return res, parts, gold, sim


if __name__ == "__main__":
    res, parts, _, _ = run()
    print(json.dumps(res, indent=1))
    print("example:", parts[0]["text"], "|", parts[1]["text"])
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
