"""A synthetic HR table for the framework test: 300 employees with hire dates. Writes hr.sqlite next to this file."""
import os
import sqlite3

import numpy as np

rng = np.random.default_rng(3)
first = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe", "Gabriela", "Hugo", "Isabel", "Joao", "Karina", "Lucas"]
last = ["Silva", "Souza", "Lima", "Costa", "Rocha", "Alves", "Pereira", "Gomes", "Martins", "Barros"]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hr.sqlite")
if os.path.exists(path):
    os.remove(path)
con = sqlite3.connect(path)
con.execute("create table employees (id integer primary key, name text, department text, hire_date text)")
days = np.sort(rng.integers(0, 12 * 365, 300))
for i, d in enumerate(days):
    date = np.datetime64("2013-03-01") + int(d)
    con.execute("insert into employees values (?, ?, ?, ?)",
                (i + 1, f"{rng.choice(first)} {rng.choice(last)} {i + 1}", str(rng.choice(["Sales", "Ops", "Finance", "IT"])), str(date)))
con.commit()
print(con.execute("select name, hire_date from employees order by hire_date limit 1").fetchone(),
      con.execute("select name, hire_date from employees order by hire_date desc limit 1").fetchone())
