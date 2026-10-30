"""Safe natural-language -> SQL over the employee database.

Safety model: the model never touches the real database. For each request we copy
ONLY allowlisted columns and ONLY the rows this principal may see into a throw-away
in-memory SQLite database, detach the real file, and run the model's SELECT there
(query_only, authorizer, row limit, time limit). Password hashes, logs and other
people's rows simply do not exist in that sandbox.
"""
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .access import DB_PATH, Principal, employee_scope, require_source
from .settings import SETTINGS

# table -> allowlisted columns. employee-keyed tables are filtered to the visible employees.
SANDBOX_TABLES = {
    "employees": ["employee_id", "full_name", "email", "department", "designation",
                  "location", "manager_id", "joining_date", "status"],
    "skills": ["skill_id", "skill_name", "category"],
    "employee_skills": ["employee_id", "skill_id", "proficiency_level",
                        "years_experience", "last_verified_date"],
    "certifications": ["employee_id", "cert_name", "issuing_body", "issue_date",
                       "expiry_date", "verified_flag"],
    "assessments": ["employee_id", "skill_id", "assessment_name", "score", "date",
                    "result_status"],
    "projects": ["project_id", "project_name", "client", "required_skills", "start_date",
                 "end_date", "status"],
    "project_allocation": ["employee_id", "project_id", "project_name", "manager_name"],
    # id + name only (never password_hash), and only managers of employees this user can see
    "managers": ["manager_id", "full_name"],
}
GLOBAL_TABLES = {"skills", "projects"}          # not tied to one employee
MAX_ROWS = SETTINGS.sql_max_rows
TIME_LIMIT_S = SETTINGS.sql_time_limit_s

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _allow_all(*_):
    return sqlite3.SQLITE_OK


@dataclass
class SqlResult:
    sql: str
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    error: str = None


def build_sandbox(principal: Principal, db_path=DB_PATH) -> sqlite3.Connection:
    require_source(principal, "employee_db")
    clause, params = employee_scope(principal)

    mem = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
    mem.execute("ATTACH DATABASE ? AS src", (f"{Path(db_path).resolve().as_uri()}?mode=ro",))
    present = {r[0] for r in mem.execute("SELECT name FROM src.sqlite_master WHERE type='table'")}

    # employees first: everything else is filtered against the visible employee set
    order = ["employees"] + [t for t in SANDBOX_TABLES if t != "employees"]
    for table in order:
        src_table = "manager" if table == "managers" else table
        if src_table not in present:
            continue
        cols = ", ".join(SANDBOX_TABLES[table])
        if table == "managers":
            mem.execute(f"CREATE TABLE managers AS SELECT {cols} FROM src.manager "
                        f"WHERE manager_id IN (SELECT manager_id FROM main.employees)")
        elif table == "employees":
            mem.execute(f"CREATE TABLE employees AS SELECT {cols} FROM src.employees "
                        f"WHERE {clause}", params)
        elif table in GLOBAL_TABLES:
            mem.execute(f"CREATE TABLE {table} AS SELECT {cols} FROM src.{table}")
        else:
            mem.execute(f"CREATE TABLE {table} AS SELECT {cols} FROM src.{table} "
                        f"WHERE employee_id IN (SELECT employee_id FROM main.employees)")
    mem.commit()
    mem.execute("DETACH DATABASE src")          # the real file is now unreachable
    mem.execute("PRAGMA query_only = ON")
    mem.set_authorizer(_authorizer)             # SELECT only; no ATTACH / PRAGMA / DDL / DML
    return mem


def validate_select(sql: str) -> str:
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s:
        raise ValueError("only one statement is allowed")
    if not re.match(r"(?is)^(select|with)\b", s):
        raise ValueError("only SELECT queries are allowed")
    return s


def run_query(mem: sqlite3.Connection, sql: str, max_rows: int = MAX_ROWS):
    s = validate_select(sql)
    deadline = time.monotonic() + TIME_LIMIT_S
    mem.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cur = mem.execute(f"SELECT * FROM ({s}) LIMIT {int(max_rows)}")
        return [d[0] for d in cur.description], cur.fetchall()
    finally:
        mem.set_progress_handler(None, 0)


def schema_doc(mem: sqlite3.Connection) -> str:
    """Schema text for the SQL prompt, built only from what this principal can see.

    This is trusted code (not model output), so the SELECT-only authorizer is lifted
    just for the duration of this function and always restored.
    """
    # NOTE: passing None to disable the authorizer needs Python 3.11+, so swap in an allow-all callback.
    mem.set_authorizer(_allow_all)
    try:
        return _schema_doc(mem)
    finally:
        mem.set_authorizer(_authorizer)


def _schema_doc(mem: sqlite3.Connection) -> str:
    lines = []
    for (table,) in mem.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [r[1] for r in mem.execute(f"PRAGMA table_info({table})")]
        lines.append(f"{table}({', '.join(cols)})")
    depts = [r[0] for r in mem.execute(
        "SELECT DISTINCT department FROM employees WHERE department IS NOT NULL LIMIT 30")]
    desigs = [r[0] for r in mem.execute(
        "SELECT DISTINCT designation FROM employees WHERE designation IS NOT NULL LIMIT 30")]
    skills = []
    if _has(mem, "employee_skills") and _has(mem, "skills"):
        skills = [r[0] for r in mem.execute(
            "SELECT s.skill_name FROM employee_skills es JOIN skills s ON s.skill_id = es.skill_id "
            "GROUP BY s.skill_name ORDER BY COUNT(*) DESC LIMIT 40")]
    return ("TABLES:\n" + "\n".join(lines) +
            f"\nDEPARTMENT VALUES: {depts}\nDESIGNATION VALUES: {desigs}"
            f"\nCOMMON SKILL NAMES: {skills}")


def _has(mem, table):
    return mem.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (table,)).fetchone() is not None


SQL_PROMPT = """You translate a question about employees into ONE read-only SQLite query.

{schema}

RULES
- Output only the SQL. No markdown, no explanation.
- Exactly one SELECT (a WITH ... SELECT is fine). Never modify data.
- Match text case-insensitively: LOWER(col) LIKE '%value%'.
- Skills live in employee_skills joined to skills on skill_id; years of experience is
  employee_skills.years_experience.
- Always include employee_id and full_name when listing people.
{history}
QUESTION: {question}
SQL:"""

REPAIR_PROMPT = """The SQL below failed.

{schema}

SQL: {sql}
ERROR: {error}

Return only a corrected single SELECT query."""


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:sql)?\s*|\s*```$", "", t, flags=re.I).strip()
    return t


def ask_sql(question: str, principal: Principal, llm, history: str = "",
            db_path=DB_PATH) -> SqlResult:
    """llm: callable(prompt: str) -> str. Returns rows for the question, scoped to principal."""
    mem = build_sandbox(principal, db_path)
    try:
        schema = schema_doc(mem)
        hist = f"CONVERSATION SO FAR:\n{history}\n" if history else ""
        sql = _clean(llm(SQL_PROMPT.format(schema=schema, history=hist, question=question)))
        for attempt in range(2):
            try:
                cols, rows = run_query(mem, sql)
                return SqlResult(sql, cols, rows)
            except (sqlite3.Error, ValueError) as e:
                if attempt == 1:
                    return SqlResult(sql, error=str(e))
                sql = _clean(llm(REPAIR_PROMPT.format(schema=schema, sql=sql, error=e)))
    finally:
        mem.close()
