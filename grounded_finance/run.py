"""Asks the 24 DRE questions two ways against an OpenAI-compatible model and grades the answers.

  table_in_prompt  the model gets every fact row as CSV (576 rows) plus the dimensions, and answers directly
  query_only       the model writes one SQLite SELECT from the schema; the code runs it (one retry with the error,
                   the job of a "SQL fixer" agent); the model writes the answer from the rows; then a check rejects any
                   number in the answer that is not in the rows, and falls back to printing the rows

An answer is correct when it contains the expected name, or a number within 0.5% of the expected value
(0.3 points for percentages). Lenient on purpose: any number in the answer may match.

  LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python grounded_finance/run.py results/run1.json
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dre

BASE, KEY, MODEL = os.environ["LLM_BASE_URL"].rstrip("/"), os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"]
EXTRA = json.loads(os.environ.get("LLM_EXTRA_HEADERS") or "{}")


def chat(system, user, max_tokens=2000):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", body,
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", **EXTRA})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text, round(time.time() - t0, 1)


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers(text):
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def correct(answer, expected, kind):
    if kind == "name":
        return str(expected).lower() in answer.lower()
    tol = 0.3 if kind == "percent" else abs(expected) * 0.005
    return any(abs(n - expected) <= tol or abs(-n - expected) <= tol for n in numbers(answer))


def table_csv(con):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["month", "unit", "account", "line", "value"])
    for row in con.execute("select f.month, u.unit, a.account, a.line, f.value " + dre.JOIN + " order by 1, 2, 3"):
        w.writerow(row)
    return buf.getvalue()


SYS_TABLE = ("You are a financial analyst. Answer the question using only the income statement rows provided. "
             "Values are in thousands. Give the final number or name in one short sentence.")
SYS_SQL = ("You write one SQLite SELECT query that answers the question. Return only the SQL, no explanation. "
           "Never invent values; every number must come from the tables.\n" + dre.SCHEMA)
SYS_ANSWER = ("You answer a finance question in one short sentence using only the query result given. "
              "Do not compute anything that is not in the result; copy the numbers, rounded to two decimals.")


def query_only(con, question):
    sql_text, t1 = chat(SYS_SQL, question, 1500)
    sql = re.sub(r"^```\w*|```$", "", sql_text.strip(), flags=re.M).strip()
    rows, error, retried = None, None, False
    for attempt in range(2):
        try:
            if not sql.lower().lstrip().startswith(("select", "with")):
                raise ValueError("not a SELECT")
            rows = con.execute(sql).fetchall()[:20]
            break
        except Exception as e:  # one retry, with the error, like a SQL fixer agent
            error = str(e)
            if attempt == 0:
                retried = True
                sql_text, t_fix = chat(SYS_SQL, f"{question}\nYour previous query failed: {error}\nQuery: {sql}", 1500)
                t1 += t_fix
                sql = re.sub(r"^```\w*|```$", "", sql_text.strip(), flags=re.M).strip()
    if rows is None:
        return {"answer": "I could not answer this from the data.", "sql": sql, "error": error, "retried": retried,
                "fallback": False, "seconds": t1}
    answer, t2 = chat(SYS_ANSWER, f"Question: {question}\nQuery result rows: {rows}", 800)
    allowed = [round(float(v), 2) for r in rows for v in r if isinstance(v, (int, float))]
    stray = [n for n in numbers(answer) if not any(abs(n - a) <= max(0.01, abs(a) * 0.001) or abs(-n - a) <= max(0.01, abs(a) * 0.001) for a in allowed)
             and not re.fullmatch(r"20\d\d", str(int(n)) if n == int(n) else "") and n not in (1, 2, 3, 4)]
    fallback = bool(stray)
    if fallback:  # a number the query did not return: do not show the model's sentence
        answer = f"Query result: {rows}"
    return {"answer": answer, "sql": sql, "retried": retried, "fallback": fallback, "stray_numbers": stray,
            "seconds": round(t1 + t2, 1)}


def main(out):
    con = dre.build()
    expected = dre.expected(con)
    table = table_csv(con)
    results = []
    for qid, question, _, kind in dre.QUESTIONS:
        a_text, a_sec = chat(SYS_TABLE, f"Income statement rows (CSV):\n{table}\nQuestion: {question}", 4000)
        b = query_only(con, question)
        r = {"id": qid, "kind": kind, "expected": expected[qid],
             "table_in_prompt": {"answer": a_text[-600:], "correct": correct(a_text, expected[qid], kind), "seconds": a_sec},
             "query_only": {**b, "correct": correct(b["answer"], expected[qid], kind)}}
        results.append(r)
        print(qid, r["table_in_prompt"]["correct"], r["query_only"]["correct"], a_sec, b["seconds"], flush=True)
    summary = {mode: sum(r[mode]["correct"] for r in results) for mode in ("table_in_prompt", "query_only")}
    summary["query_only_fallbacks"] = sum(r["query_only"]["fallback"] for r in results)
    summary["query_only_retries"] = sum(r["query_only"]["retried"] for r in results)
    summary["model"] = MODEL
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=1, default=str)
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
