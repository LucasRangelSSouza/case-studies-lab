"""A synthetic income statement (DRE) as a small star schema in SQLite, and the questions to ask about it.

  fact_dre(month, unit_id, account_id, value)      12 months x 4 business units x 12 accounts
  dim_unit(unit_id, unit)    dim_account(account_id, account, line)    lines: gross_revenue, deductions, cogs, opex, financial

Revenue lines are positive, deductions and costs negative, the way the ledger exports them. Every question has a
reference SQL query written by hand; its result is the expected answer.

  python grounded_finance/dre.py      # writes dre.sqlite and prints the expected answers
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "dre.sqlite")
UNITS = ["North", "South", "Export", "Services"]
ACCOUNTS = [  # (account, line, base monthly value per unit in thousands)
    ("Product sales", "gross_revenue", 900), ("Service revenue", "gross_revenue", 260),
    ("Sales taxes", "deductions", -170), ("Returns", "deductions", -25),
    ("Raw materials", "cogs", -310), ("Direct labour", "cogs", -140), ("Freight", "cogs", -55),
    ("Salaries", "opex", -160), ("Rent", "opex", -40), ("Marketing", "opex", -45), ("IT services", "opex", -30),
    ("Interest", "financial", -22),
]
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def build():
    rng = np.random.default_rng(21)
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.executescript("""
        create table dim_unit (unit_id integer primary key, unit text);
        create table dim_account (account_id integer primary key, account text, line text);
        create table fact_dre (month text, unit_id integer, account_id integer, value real);
    """)
    con.executemany("insert into dim_unit values (?, ?)", list(enumerate(UNITS, 1)))
    con.executemany("insert into dim_account values (?, ?, ?)", [(i, a, l) for i, (a, l, _) in enumerate(ACCOUNTS, 1)])
    scale = {"North": 1.0, "South": 0.7, "Export": 0.55, "Services": 0.35}
    for mi, month in enumerate(MONTHS):
        season = 1 + 0.12 * np.sin(2 * np.pi * (mi - 2) / 12) + 0.015 * mi
        for ui, unit in enumerate(UNITS, 1):
            for ai, (account, line, base) in enumerate(ACCOUNTS, 1):
                v = base * scale[unit] * season * rng.normal(1, 0.06)
                if unit == "Services" and account == "Product sales":
                    v *= 0.2
                if unit == "Export" and account == "Freight" and mi >= 8:
                    v *= 1.9  # a cost spike the questions ask about
                con.execute("insert into fact_dre values (?, ?, ?, ?)", (month, ui, ai, round(float(v), 2)))
    con.commit()
    return con


NET = "sum(case when a.line in ('gross_revenue','deductions') then f.value end)"
JOIN = "from fact_dre f join dim_account a using (account_id) join dim_unit u using (unit_id)"

# (id, question, reference SQL, kind) kind: number | percent | name
QUESTIONS = [
    ("q01", "What was total gross revenue in 2025, in thousands?", f"select sum(f.value) {JOIN} where a.line='gross_revenue'", "number"),
    ("q02", "What was net revenue (gross revenue minus deductions) in 2025, in thousands?", f"select {NET} {JOIN}", "number"),
    ("q03", "What was net revenue of the North unit in March 2025, in thousands?", f"select {NET} {JOIN} where u.unit='North' and f.month='2025-03'", "number"),
    ("q04", "How much did the Export unit spend on freight in 2025, in thousands (as a positive number)?", f"select -sum(f.value) {JOIN} where u.unit='Export' and a.account='Freight'", "number"),
    ("q05", "What was total salaries cost across all units in the second quarter of 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where a.account='Salaries' and f.month between '2025-04' and '2025-06'", "number"),
    ("q06", "Which business unit had the highest net revenue in 2025?", f"select u.unit {JOIN} group by u.unit order by {NET} desc limit 1", "name"),
    ("q07", "Which business unit had the lowest gross revenue in 2025?", f"select u.unit {JOIN} where a.line='gross_revenue' group by u.unit order by sum(f.value) asc limit 1", "name"),
    ("q08", "What was the operating result (net revenue plus cogs plus opex) of the South unit in 2025, in thousands?", f"select sum(case when a.line<>'financial' then f.value end) {JOIN} where u.unit='South'", "number"),
    ("q09", "What was the gross margin of the company in 2025, as a percentage of net revenue? Gross margin = net revenue + cogs.", f"select 100.0*sum(case when a.line in ('gross_revenue','deductions','cogs') then f.value end)/{NET} {JOIN}", "percent"),
    ("q10", "Which cost account was the largest in 2025 for the whole company?", f"select a.account {JOIN} where f.value<0 group by a.account order by sum(f.value) asc limit 1", "name"),
    ("q11", "By how many percent did Export freight cost change from the first eight months' monthly average to the last four months' monthly average?", f"select 100.0*(avg(case when f.month>='2025-09' then -f.value end)/avg(case when f.month<'2025-09' then -f.value end)-1) {JOIN} where u.unit='Export' and a.account='Freight'", "percent"),
    ("q12", "What was the total marketing spend of the Services unit in 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where u.unit='Services' and a.account='Marketing'", "number"),
    ("q13", "What was net income (all lines) of the whole company in December 2025, in thousands?", f"select sum(f.value) {JOIN} where f.month='2025-12'", "number"),
    ("q14", "In which month of 2025 was company-wide gross revenue highest?", f"select f.month {JOIN} where a.line='gross_revenue' group by f.month order by sum(f.value) desc limit 1", "name"),
    ("q15", "What share of 2025 gross revenue came from service revenue, in percent?", f"select 100.0*sum(case when a.account='Service revenue' then f.value end)/sum(case when a.line='gross_revenue' then f.value end) {JOIN}", "percent"),
    ("q16", "What were the North unit's total opex in 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where u.unit='North' and a.line='opex'", "number"),
    ("q17", "How much were sales taxes for the whole company in the first half of 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where a.account='Sales taxes' and f.month<='2025-06'", "number"),
    ("q18", "What was the average monthly gross revenue of the South unit in 2025, in thousands?", f"select sum(f.value)/12 {JOIN} where u.unit='South' and a.line='gross_revenue'", "number"),
    ("q19", "Which unit spent the most on IT services in 2025?", f"select u.unit {JOIN} where a.account='IT services' group by u.unit order by sum(f.value) asc limit 1", "name"),
    ("q20", "What was the company's interest expense in 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where a.account='Interest'", "number"),
    ("q21", "What was the operating margin of the Export unit in 2025 as a percent of its net revenue? Operating result = net revenue + cogs + opex.", f"select 100.0*sum(case when a.line<>'financial' then f.value end)/{NET} {JOIN} where u.unit='Export'", "percent"),
    ("q22", "By how much did company-wide net revenue grow from January to December 2025, in percent?", f"select 100.0*(sum(case when f.month='2025-12' and a.line in ('gross_revenue','deductions') then f.value end)/sum(case when f.month='2025-01' and a.line in ('gross_revenue','deductions') then f.value end)-1) {JOIN}", "percent"),
    ("q23", "What was the total cost of goods sold (cogs) of the North unit in the fourth quarter, in thousands (positive)?", f"select -sum(f.value) {JOIN} where u.unit='North' and a.line='cogs' and f.month>='2025-10'", "number"),
    ("q24", "What was the combined direct labour and raw materials cost of the South unit in July 2025, in thousands (positive)?", f"select -sum(f.value) {JOIN} where u.unit='South' and a.account in ('Direct labour','Raw materials') and f.month='2025-07'", "number"),
]


def expected(con):
    return {qid: con.execute(sql).fetchone()[0] for qid, _, sql, _ in QUESTIONS}


SCHEMA = """Tables (SQLite):
  fact_dre(month TEXT 'YYYY-MM', unit_id INTEGER, account_id INTEGER, value REAL)  -- thousands; revenue positive, deductions and costs negative
  dim_unit(unit_id INTEGER, unit TEXT)        -- units: North, South, Export, Services
  dim_account(account_id INTEGER, account TEXT, line TEXT)  -- line in: gross_revenue, deductions, cogs, opex, financial
Accounts: """ + ", ".join(f"{a} ({l})" for a, l, _ in ACCOUNTS)


if __name__ == "__main__":
    con = build()
    for qid, value in expected(con).items():
        print(qid, value)
