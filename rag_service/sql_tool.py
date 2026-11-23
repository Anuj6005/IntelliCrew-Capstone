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
from datetime import date
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


def query_workforce(question: str, principal: Principal, db_path=DB_PATH, *,
                    include_skill: bool = True) -> SqlResult:
    """Run a small set of fixed, parameterized workforce queries.

    This is the chatbot path: the question selects a supported query shape, while
    values are always bound as parameters. The model never writes SQL here.
    """
    require_source(principal, "employee_db")
    clause, params = employee_scope(principal)
    if principal.role == "MANAGER":
        clause = "e.manager_id = ?"
    else:
        clause = "1 = 1"

    question = (question or "").lower()
    team_query = principal.role == "MANAGER" and bool(re.search(
        r"\b(team|direct reports?|reports? to|reporting to|under me)\b", question))
    department_match = None if team_query else re.search(
        r"\b(?:in|from)\s+(.+?)(?:\s+(?:department|dept)\b|[?!.,]*$)", question)
    department = department_match.group(1).strip() if department_match else None
    if department and re.search(r"\b(company|organization|team)\b", department):
        department = None
    skill_match = re.search(
        r"\b(?:know|knows|knowing|have|has|with|using|skilled\s+in)\s+([\w+#.-]+)",
        question)
    skill = skill_match.group(1).strip() if skill_match and include_skill else None
    designation_match = re.search(
        r"\b((?:[a-z][\w+#.-]*\s+)?(?:developers?|engineers?|analysts?|interns?|"
        r"designers?|scientists?|architects?|consultants?|specialists?|administrators?))\b",
        question)
    designation = designation_match.group(1) if designation_match else None
    if designation:
        designation = re.sub(r"\bdevelopers\b", "developer", designation)
        designation = re.sub(r"\bengineers\b", "engineer", designation)
        designation = re.sub(r"\banalysts\b", "analyst", designation)
    count_query = bool(re.search(r"\b(how many|count|number of)\b", question))

    if not count_query and not (department or skill or designation or team_query):
        return SqlResult("", error="unsupported structured question")

    if count_query:
        sql = "SELECT COUNT(DISTINCT e.employee_id) AS employee_count FROM employees e"
    else:
        sql = ("SELECT DISTINCT e.employee_id, e.full_name, e.department, e.designation "
               "FROM employees e")
    query_params = list(params) if principal.role == "MANAGER" else []
    where = [clause]
    if department:
        where.append("instr(LOWER(e.department), LOWER(?)) > 0")
        query_params.append(department)
    if skill:
        sql += (" JOIN employee_skills es ON es.employee_id = e.employee_id "
                "JOIN skills s ON s.skill_id = es.skill_id")
        where.append("LOWER(s.skill_name) = LOWER(?)")
        query_params.append(skill)
    if designation:
        where.append("instr(LOWER(e.designation), LOWER(?)) > 0")
        query_params.append(designation)
    sql += " WHERE " + " AND ".join(where)
    if not count_query:
        sql += " ORDER BY e.full_name LIMIT ?"
        query_params.append(MAX_ROWS)

    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    deadline = time.monotonic() + TIME_LIMIT_S
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cursor = conn.execute(sql, query_params)
        return SqlResult(sql, [col[0] for col in cursor.description], cursor.fetchall())
    except sqlite3.Error as error:
        return SqlResult("", error=str(error))
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def query_person_skills(employee_id: str, principal: Principal, db_path=DB_PATH) -> SqlResult:
    """Read one employee's structured skills while enforcing the caller's scope."""
    require_source(principal, "employee_db")
    clause, params = employee_scope(principal)
    if principal.role == "MANAGER":
        clause = "e.manager_id = ?"
    else:
        clause = "1 = 1"
    sql = ("SELECT e.employee_id, e.full_name, s.skill_name, es.proficiency_level, "
           "es.years_experience FROM employees e "
           "JOIN employee_skills es ON es.employee_id = e.employee_id "
           "JOIN skills s ON s.skill_id = es.skill_id "
           f"WHERE ({clause}) AND e.employee_id = ? ORDER BY s.skill_name LIMIT ?")
    query_params = [*(params if principal.role == "MANAGER" else []), employee_id, MAX_ROWS]
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql, query_params)
        return SqlResult(sql, [column[0] for column in cursor.description], cursor.fetchall())
    except sqlite3.Error as error:
        return SqlResult("", error=str(error))
    finally:
        conn.close()


ACTIVE_ALLOCATION_RULE = (
    "An active allocation is a project_allocation row linked to a project whose status is "
    "In Progress and whose start/end dates include today; a missing date bound is allowed. "
    "An employee is available when employees.status is Active and they have no active allocation."
)


def query_person_work_status(employee_id: str, principal: Principal, db_path=DB_PATH,
                             *, today=None) -> SqlResult:
    """Return one in-scope employee's status and all allocation/project details.

    Allocation activity follows ACTIVE_ALLOCATION_RULE. Values and dates are bound parameters;
    the manager predicate is applied before a named employee id can match.
    """
    require_source(principal, "employee_db")
    scope = "e.manager_id = ?" if principal.role == "MANAGER" else "1 = 1"
    scope_params = [principal.user_id] if principal.role == "MANAGER" else []
    as_of = (today or date.today()).isoformat()
    sql = (
        "SELECT e.employee_id, e.full_name, e.status AS employee_status, "
        "COALESCE(p.project_name, a.project_name) AS project_name, p.client, "
        "p.status AS project_status, p.start_date, p.end_date, a.manager_name, "
        "CASE WHEN LOWER(COALESCE(p.status, '')) = 'in progress' "
        "AND (p.start_date IS NULL OR date(p.start_date) <= date(?)) "
        "AND (p.end_date IS NULL OR date(p.end_date) >= date(?)) "
        "THEN 1 ELSE 0 END AS active_allocation "
        "FROM employees e "
        "LEFT JOIN project_allocation a ON a.employee_id = e.employee_id "
        "LEFT JOIN projects p ON (p.project_id = a.project_id OR "
        "(a.project_id IS NULL AND LOWER(p.project_name) = LOWER(a.project_name))) "
        f"WHERE ({scope}) AND e.employee_id = ? "
        "ORDER BY active_allocation DESC, project_name LIMIT ?")
    params = [as_of, as_of, *scope_params, employee_id, MAX_ROWS]
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql, params)
        return SqlResult(sql, [column[0] for column in cursor.description], cursor.fetchall())
    except sqlite3.Error as error:
        return SqlResult("", error=str(error))
    finally:
        conn.close()


def _skill_filter(question: str, conn: sqlite3.Connection):
    """Find a database skill mentioned in the query; unknown requested skills match nothing."""
    text = (question or "").lower()
    skills = [row[0] for row in conn.execute(
        "SELECT skill_name FROM skills WHERE skill_name IS NOT NULL")]
    for skill in sorted(skills, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])", text):
            return skill
    role = re.search(
        r"\b(developers?|engineers?|analysts?|designers?|scientists?|architects?|"
        r"consultants?|specialists?)\b", text)
    if role:
        prefix = text[:role.start()]
        prefix = re.sub(r"\b(who|is|are|list|show|find|any|available|availability|"
                        r"free|unallocated|bench|the|a|an|in|from|my|our|their|team)\b", " ", prefix)
        prefix = re.sub(r"\s+", " ", prefix).strip()
        if prefix:
            return prefix
    return None


def query_workforce_status(question: str, principal: Principal, db_path=DB_PATH,
                           *, today=None) -> SqlResult:
    """Return scoped availability/allocation lists or active employee counts.

    List/count scope is limited to employees whose status is Active. Count categories are
    complementary: active allocation versus no active allocation, using ACTIVE_ALLOCATION_RULE.
    """
    require_source(principal, "employee_db")
    text = (question or "").lower()
    allocation_words = r"allocat\w*|unallocated|assigned|on\s+a\s+project"
    count_query = bool(re.search(r"\b(how many|count|number of)\b", text) and
                       re.search(r"\b(?:" + allocation_words + r")\b", text))
    available_query = bool(re.search(
        r"\b(available|availability|free|bench|unallocated)\b|\bnot\s+allocated\b", text))
    allocated_query = bool(re.search(r"\b(?:allocated|assigned)\b|\bon\s+a\s+project\b", text))
    if not (count_query or available_query or allocated_query):
        return SqlResult("", error="unsupported structured question")

    as_of = (today or date.today()).isoformat()
    manager_scope = "e.manager_id = ?" if principal.role == "MANAGER" else "1 = 1"
    manager_params = [principal.user_id] if principal.role == "MANAGER" else []
    active_cte = (
        "WITH active_allocations AS ("
        "SELECT a.employee_id, COALESCE(p.project_name, a.project_name) AS project_name, "
        "p.client, p.status AS project_status, p.start_date, p.end_date, a.manager_name "
        "FROM project_allocation a JOIN projects p ON (p.project_id = a.project_id OR "
        "(a.project_id IS NULL AND LOWER(p.project_name) = LOWER(a.project_name))) "
        "WHERE LOWER(COALESCE(p.status, '')) = 'in progress' "
        "AND (p.start_date IS NULL OR date(p.start_date) <= date(?)) "
        "AND (p.end_date IS NULL OR date(p.end_date) >= date(?))) ")
    params = [as_of, as_of]

    if count_query:
        sql = (active_cte +
               "SELECT COUNT(DISTINCT CASE WHEN aa.employee_id IS NOT NULL "
               "THEN e.employee_id END) AS allocated_count, "
               "COUNT(DISTINCT CASE WHEN aa.employee_id IS NULL "
               "THEN e.employee_id END) AS unallocated_count "
               "FROM employees e LEFT JOIN active_allocations aa "
               "ON aa.employee_id = e.employee_id "
               f"WHERE ({manager_scope}) AND LOWER(TRIM(COALESCE(e.status, ''))) = 'active'")
        params.extend(manager_params)
    else:
        sql = (active_cte +
               "SELECT DISTINCT e.employee_id, e.full_name, e.designation, e.status, "
               "aa.project_name, aa.client, aa.project_status, aa.start_date, aa.end_date, "
               "aa.manager_name, CASE WHEN aa.employee_id IS NULL THEN 0 ELSE 1 END "
               "AS active_allocation FROM employees e "
               "LEFT JOIN active_allocations aa ON aa.employee_id = e.employee_id ")
        conditions = [f"({manager_scope})",
                      "LOWER(TRIM(COALESCE(e.status, ''))) = 'active'"]
        params.extend(manager_params)
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
        try:
            skill = _skill_filter(text, conn)
        except sqlite3.Error as error:
            return SqlResult("", error=str(error))
        finally:
            conn.close()
        if skill:
            conditions.append(
                "EXISTS (SELECT 1 FROM employee_skills es JOIN skills s "
                "ON s.skill_id = es.skill_id WHERE es.employee_id = e.employee_id "
                "AND LOWER(s.skill_name) = LOWER(?))")
            params.append(skill)
        role = re.search(
            r"\b(developers?|engineers?|analysts?|designers?|scientists?|architects?|"
            r"consultants?|specialists?)\b", text)
        if role:
            designation = re.sub(r"s$", "", role.group(1))
            conditions.append("instr(LOWER(e.designation), LOWER(?)) > 0")
            params.append(designation)
        if available_query:
            conditions.append("aa.employee_id IS NULL")
        elif allocated_query:
            conditions.append("aa.employee_id IS NOT NULL")
        else:
            return SqlResult("", error="unsupported structured question")
        sql += "WHERE " + " AND ".join(conditions)
        sql += " ORDER BY e.full_name, aa.project_name LIMIT ?"
        params.append(MAX_ROWS)

    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql, params)
        return SqlResult(sql, [column[0] for column in cursor.description], cursor.fetchall())
    except sqlite3.Error as error:
        return SqlResult("", error=str(error))
    finally:
        conn.close()


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
