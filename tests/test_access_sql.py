import sqlite3, pytest
from rag_service.access import Principal, allowed_sources, scoped_employee_ids
from rag_service import sql_tool as st

import shutil, tempfile
# Tests run on a throw-away COPY of the database; the real file is never modified.
DB = shutil.copy("data/employee_records.db", tempfile.mkdtemp())

@pytest.fixture(scope="module", autouse=True)
def seed():
    c = sqlite3.connect(DB)
    for t in ("employee_skills", "skills", "employees", "certifications", "assessments", "project_allocation"):
        c.execute(f"DELETE FROM {t}")
    c.executemany("INSERT INTO employees (employee_id, full_name, email, department, designation, location, manager_id, status) VALUES (?,?,?,?,?,?,?,?)", [
        ("E001", "Aarav Mehta", "a@x.com", "Engineering", "Python Developer", "Pune", "M001", "Active"),
        ("E002", "Diya Sharma", "d@x.com", "Data Analytics", "Data Analyst", "Pune", "M001", "Active"),
        ("E003", "Kabir Rao", "k@x.com", "Cloud Ops", "DevOps Engineer", "Mumbai", "M002", "Active"),
    ])
    c.executemany("INSERT INTO skills (skill_id, skill_name, category) VALUES (?,?,?)",
                  [(1, "Python", "Language"), (2, "SQL", "Data"), (3, "Kubernetes", "Cloud")])
    c.executemany("INSERT INTO employee_skills (employee_id, skill_id, proficiency_level, years_experience) VALUES (?,?,?,?)",
                  [("E001", 1, "Intermediate", 2), ("E001", 2, "Beginner", 1), ("E002", 2, "Advanced", 3), ("E003", 3, "Advanced", 4), ("E003", 1, "Beginner", 1)])
    c.commit(); c.close()

HR, M1, M2, EMP = Principal("HR", "H001"), Principal("MANAGER", "M001"), Principal("manager", "M002"), Principal("EMPLOYEE", "E001")

def q(p, sql):
    mem = st.build_sandbox(p, DB)
    try: return st.run_query(mem, sql)[1]
    finally: mem.close()

def test_sources():
    assert allowed_sources("HR") == allowed_sources("manager") == {"resumes", "employee_db"}
    assert allowed_sources("EMPLOYEE") == set() and allowed_sources("") == set()

def test_scope_ids():
    assert scoped_employee_ids(HR, DB) is None
    assert sorted(scoped_employee_ids(M1, DB)) == ["E001", "E002"]
    assert scoped_employee_ids(M2, DB) == ["E003"]

def test_fixed_workforce_queries_are_parameterized_and_team_scoped():
    team_one = st.query_workforce("who knows Python", M1, DB)
    team_two = st.query_workforce("who knows Python", M2, DB)
    assert {row[0] for row in team_one.rows} == {"E001"}
    assert {row[0] for row in team_two.rows} == {"E003"}

    hostile_department = st.query_workforce("people in x' OR 1=1 -- dept", M1, DB)
    assert hostile_department.rows == []
    assert "OR 1=1" not in hostile_department.sql

def test_team_and_designation_queries_use_the_signed_in_manager_scope():
    team = st.query_workforce("who is in my team", M1, DB)
    python_developers = st.query_workforce("python developer in my team", M1, DB)
    company_wording = st.query_workforce("list python developers in the company", M1, DB)

    assert {row[0] for row in team.rows} == {"E001", "E002"}
    assert {row[0] for row in python_developers.rows} == {"E001"}
    assert {row[0] for row in company_wording.rows} == {"E001"}
    assert "designation" in python_developers.sql.lower()

def test_row_scoping():
    assert len(q(HR, "SELECT * FROM employees")) == 3
    assert {r[0] for r in q(M1, "SELECT employee_id FROM employees")} == {"E001", "E002"}
    assert {r[0] for r in q(M2, "SELECT employee_id FROM employees")} == {"E003"}
    # employee-keyed tables are scoped too, even without a join to employees
    assert {r[0] for r in q(M1, "SELECT DISTINCT employee_id FROM employee_skills")} == {"E001", "E002"}
    assert {r[0] for r in q(M2, "SELECT DISTINCT employee_id FROM employee_skills")} == {"E003"}

def test_employee_role_blocked():
    with pytest.raises(PermissionError): st.build_sandbox(EMP, DB)

@pytest.mark.parametrize("sql", [
    "SELECT password_hash FROM manager",            # table not in sandbox
    "SELECT * FROM hr",
    "SELECT * FROM src.employees",                  # real db detached
    "SELECT * FROM audit_log",
    "SELECT created_at FROM employees",             # column not copied (email is now allowed on purpose)
    "DROP TABLE employees",
    "DELETE FROM employees",
    "INSERT INTO employees (employee_id) VALUES ('X')",
    "UPDATE employees SET department='x'",
    "PRAGMA table_info(employees)",
    "ATTACH DATABASE 'data/employee_records.db' AS z",
    "SELECT 1; SELECT 2",
    "SELECT load_extension('x')",
    "",
])
def test_blocked(sql):
    with pytest.raises(Exception): q(M1, sql)

def test_attach_inside_select_wrapper_blocked():
    with pytest.raises(Exception): q(M1, "WITH x AS (SELECT 1) SELECT * FROM x; ATTACH DATABASE 'a' AS b")

def test_cte_and_join_allowed():
    rows = q(M1, "WITH s AS (SELECT employee_id FROM employee_skills es JOIN skills k ON k.skill_id=es.skill_id WHERE k.skill_name='SQL') SELECT e.full_name FROM employees e JOIN s USING(employee_id) ORDER BY 1")
    assert [r[0] for r in rows] == ["Aarav Mehta", "Diya Sharma"]

def test_row_limit():
    mem = st.build_sandbox(HR, DB)
    try:
        rows = st.run_query(mem, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<1000) SELECT i FROM n")[1]
        assert len(rows) == st.MAX_ROWS
    finally: mem.close()

def test_time_limit():
    mem = st.build_sandbox(HR, DB)
    try:
        with pytest.raises(Exception):
            st.run_query(mem, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n) SELECT COUNT(*) FROM n")
    finally: mem.close()

def test_schema_doc_leaks_nothing_out_of_scope():
    mem = st.build_sandbox(M2, DB)
    doc = st.schema_doc(mem); mem.close()
    assert "Cloud Ops" in doc and "Data Analytics" not in doc and "Engineering" not in doc
    assert "password" not in doc and "manager(" not in doc
    assert "Kubernetes" in doc

def test_ask_sql_with_fenced_output():
    llm = lambda p: "```sql\nSELECT e.employee_id, e.full_name FROM employees e JOIN employee_skills es USING(employee_id) JOIN skills s ON s.skill_id=es.skill_id WHERE LOWER(s.skill_name) LIKE '%sql%' AND es.years_experience >= 2;\n```"
    r = st.ask_sql("who knows SQL for 2+ years", M1, llm, db_path=DB)
    assert r.error is None and r.rows == [("E002", "Diya Sharma")]
    # same question, other manager: sees nobody from M001's team
    assert st.ask_sql("who knows SQL", M2, lambda p: "SELECT full_name FROM employees e JOIN employee_skills es USING(employee_id) JOIN skills s USING(skill_id) WHERE s.skill_name='SQL'", db_path=DB).rows == []

def test_repair_loop():
    calls = []
    def llm(p):
        calls.append(p)
        return "SELEKT broken" if len(calls) == 1 else "SELECT COUNT(*) FROM employees"
    r = st.ask_sql("how many people", HR, llm, db_path=DB)
    assert r.error is None and r.rows == [(3,)] and len(calls) == 2 and "ERROR" in calls[1]

def test_gives_up_after_one_repair():
    r = st.ask_sql("x", HR, lambda p: "DROP TABLE employees", db_path=DB)
    assert r.error and r.rows == []
    assert sqlite3.connect(DB).execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 3  # real db untouched


def test_python310_compatible_authorizer():
    """set_authorizer(None) needs Python 3.11+; the project runs on 3.10."""
    import inspect
    assert "set_authorizer(None" not in inspect.getsource(st)
