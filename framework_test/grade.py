"""Grades a transcript of the five questions: {"Q1": "answer", ...} as JSON. Prints pass or fail per question.

  python framework_test/grade.py transcript.json
"""
import json
import os
import sqlite3
import sys

here = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect(os.path.join(here, "hr.sqlite"))
oldest = con.execute("select name from employees order by hire_date limit 1").fetchone()[0]
newest = con.execute("select name from employees order by hire_date desc limit 1").fetchone()[0]
checks = {
    "Q1": lambda a: "ana" in a.lower(),
    "Q2": lambda a: "ana" in a.lower(),
    "Q3": lambda a: "analyst" in a.lower(),
    "Q4": lambda a: oldest.lower() in a.lower(),
    "Q5": lambda a: newest.lower() in a.lower(),
}
answers = json.load(open(sys.argv[1], encoding="utf-8"))
for q, check in checks.items():
    print(q, "pass" if check(answers.get(q, "")) else "fail")
