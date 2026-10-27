"""SQLite storage of discovered groups.

One row per group: the method parameters (e.g. ``k``, ``n_neighbors``,
``tau``...), the member indices (JSON list), the semantic distance
(``compactness`` column, kept for compatibility with the original result files)
and the group size. The schema is created on the fly from the parameter names,
so the files are compatible with the ``RFW_*.db`` files produced for the paper.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

SQL_TYPES = {int: "INTEGER", float: "REAL", str: "TEXT"}


def init_db(path, params: Dict[str, type], fresh: bool = False) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    if fresh:
        conn.execute("DROP TABLE IF EXISTS results")
    cols = ",\n".join(f"{p} {SQL_TYPES[t]}" for p, t in params.items())
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            {cols},
            group_id INTEGER,
            indices TEXT,
            compactness REAL,
            size INTEGER)"""
    )
    conn.commit()
    return conn


def insert_groups(conn, model: str, params: Dict, groups: Sequence[Sequence[int]], compactness: Sequence[float]):
    keys = list(params)
    sql = (
        f"INSERT INTO results (model, {', '.join(keys)}, group_id, indices, compactness, size) "
        f"VALUES ({', '.join(['?'] * (len(keys) + 5))})"
    )
    rows = [
        (model, *[params[k] for k in keys], gid,
         json.dumps([int(x) for x in g], separators=(",", ":")),
         None if c is None else float(c), len(g))
        for gid, (g, c) in enumerate(zip(groups, compactness))
    ]
    with conn:
        conn.executemany(sql, rows)


def load_groups(path, model: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(str(path))
    if model is None:
        df = pd.read_sql_query("SELECT * FROM results", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM results WHERE model = ?", conn, params=(model,))
    conn.close()
    if df.empty:
        raise ValueError(f"No rows found in {path} for model={model}")
    df["indices"] = df["indices"].apply(json.loads)
    return df.drop(columns=["id"])
