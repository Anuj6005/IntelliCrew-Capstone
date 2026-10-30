"""Single place that decides WHO can search WHAT.

Rules (portal decisions):
  - HR sees every employee.
  - A Manager sees only employees whose employees.manager_id equals their own id.
  - Employees get no resume / employee-database access (company_docs comes in the portal).

Scoping is derived from SQLite (the source of truth), never from the prompt.
"""
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_records.db"

ROLE_HR = "HR"
ROLE_MANAGER = "MANAGER"


@dataclass(frozen=True)
class Principal:
    role: str       # "HR" / "MANAGER" (as stored in the session)
    user_id: str    # hr_id or manager_id from the session

    def __post_init__(self):
        object.__setattr__(self, "role", (self.role or "").upper())


def allowed_sources(role: str) -> set:
    """Knowledge sources a role may query. The portal adds 'company_docs' for everyone."""
    role = (role or "").upper()
    if role in (ROLE_HR, ROLE_MANAGER):
        return {"resumes", "employee_db"}
    return set()


def require_source(principal: Principal, source: str) -> None:
    if source not in allowed_sources(principal.role):
        raise PermissionError(f"role '{principal.role}' may not access '{source}'")


def employee_scope(principal: Principal):
    """(sql_where_clause, params) restricting the employees table for this principal."""
    if principal.role == ROLE_HR:
        return "1=1", ()
    if principal.role == ROLE_MANAGER:
        return "manager_id = ?", (principal.user_id,)
    raise PermissionError(f"role '{principal.role}' has no employee access")


def scoped_employee_ids(principal: Principal, db_path=DB_PATH):
    """Employee ids this principal may see. None means 'all' (HR): no id filter needed."""
    if principal.role == ROLE_HR:
        return None
    clause, params = employee_scope(principal)
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute(
            f"SELECT employee_id FROM employees WHERE {clause}", params)]
    finally:
        conn.close()
