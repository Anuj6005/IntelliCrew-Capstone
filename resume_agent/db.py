"""Raw SQLite insert helpers — no ORM classes, just functions."""
import sqlite3
from datetime import date
from pathlib import Path

# db.py is inside agent/, so go up ONE level to reach the project root, then data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_FILE = BASE_DIR / "data" / "employee_records.db"   # ✅ matches data_seed.py

# employees.manager_id is a self-FK to employees(employee_id).
# Business tables start empty, so keep it NULL for now (FK is skipped when None).
DEFAULT_MANAGER_ID = None


def get_conn():
    # open a connection, rows accessible by column name
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_employee(full_name, email, department, designation,
                    employee_id=None, manager_id=DEFAULT_MANAGER_ID,
                    location=None, joining_date=None):
    """Return employee_id (string / nvarchar).

    Priority for finding an existing row:
      1. by extracted employee_id (if provided)
      2. by email
    If not found, insert — using the provided employee_id when available,
    otherwise let SQLite auto-generate it.
    """
    conn = get_conn()
    cur = conn.cursor()

    # normalize: treat empty string as missing
    employee_id = employee_id.strip() if isinstance(employee_id, str) and employee_id.strip() else None
    manager_id = manager_id.strip() if isinstance(manager_id, str) and manager_id.strip() else manager_id
    location = location.strip() if isinstance(location, str) and location.strip() else None
    joining_date = joining_date.strip() if isinstance(joining_date, str) and joining_date.strip() else None

    row = None

    # 1. try matching by the given employee_id first
    if employee_id:
        row = cur.execute(
            "SELECT employee_id FROM employees WHERE employee_id = ?", (employee_id,)
        ).fetchone()

    # 2. fall back to matching by email
    if not row and email:
        row = cur.execute(
            "SELECT employee_id FROM employees WHERE email = ?", (email,)
        ).fetchone()

    if row:
        emp_id = row["employee_id"]          # already exists
    elif employee_id:
        # insert WITH the provided id — column order matches the schema
        cur.execute(
            """INSERT INTO employees
               (employee_id, full_name, email, department, designation, location,
                manager_id, joining_date, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', datetime('now'), datetime('now'))""",
            (employee_id, full_name, email, department, designation, location,
             manager_id, joining_date),
        )
        emp_id = employee_id
        conn.commit()
    else:
        # no id given → let SQLite auto-generate
        cur.execute(
            """INSERT INTO employees
               (full_name, email, department, designation, location,
                manager_id, joining_date, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'active', datetime('now'), datetime('now'))""",
            (full_name, email, department, designation, location,
             manager_id, joining_date),
        )
        emp_id = cur.lastrowid
        conn.commit()

    conn.close()
    return emp_id


def get_or_create_skill(skill_name, category=None, description=None):
    """Return skill_id — insert into skills only if it does not exist."""
    conn = get_conn()
    cur = conn.cursor()

    row = cur.execute(
        "SELECT skill_id FROM skills WHERE skill_name = ?", (skill_name,)
    ).fetchone()

    if row:
        skill_id = row["skill_id"]
    else:
        cur.execute(
            "INSERT INTO skills (skill_name, category, description) VALUES (?, ?, ?)",
            (skill_name, category, description),
        )
        skill_id = cur.lastrowid
        conn.commit()

    conn.close()
    return skill_id


def add_employee_skill(employee_id, skill_id, proficiency_level, years_experience):
    """Link employee <-> skill in employee_skills (skip if already linked)."""
    conn = get_conn()
    cur = conn.cursor()

    exists = cur.execute(
        "SELECT id FROM employee_skills WHERE employee_id = ? AND skill_id = ?",
        (employee_id, skill_id),
    ).fetchone()

    if not exists:
        cur.execute(
            """INSERT INTO employee_skills
               (employee_id, skill_id, proficiency_level, years_experience, source, last_verified_date)
               VALUES (?, ?, ?, ?, 'resume', ?)""",
            (employee_id, skill_id, proficiency_level, years_experience, date.today().isoformat()),
        )
        conn.commit()

    conn.close()
    
    
    
def get_resume_log_by_name(file_name):
    """Return the resume_logs row for a file_name, or None if not logged yet."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT log_id, file_name, file_path, ingested FROM resume_logs WHERE file_name = ?",
        (file_name,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def log_resume(file_name, file_path, ingested=0):
    """Insert a row into resume_logs. Returns log_id."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO resume_logs (file_name, file_path, ingested) VALUES (?, ?, ?)",
        (file_name, file_path, ingested),
    )
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id


def mark_resume_ingested(log_id):
    """Flip ingested to 1 after embedding succeeds."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE resume_logs SET ingested = 1 WHERE log_id = ?", (log_id,))
    conn.commit()
    conn.close()


def get_employee_name(employee_id):
    """Look up full_name for embed metadata (returns None if not found)."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT full_name FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone()
    conn.close()
    return row["full_name"] if row else None

# ---- paste at the END of resume_agent/db.py ----

def get_employee_profile(employee_id):
    """Profile fields used as Chroma chunk metadata (None if the employee is unknown)."""
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT full_name, department, designation, manager_id "
        "FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None