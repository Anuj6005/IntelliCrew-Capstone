import sqlite3
from datetime import date

import pytest

from data_seed import SCHEMA_SQL
from rag_service.access import Principal
from rag_service import pipeline
from rag_service import service as service_module
from rag_service.service import RagService
from rag_service.sql_tool import (
    ACTIVE_ALLOCATION_RULE,
    query_person_work_status,
    query_workforce_status,
)


HR = Principal("HR", "H001")
M1 = Principal("MANAGER", "M001")
M2 = Principal("MANAGER", "M002")
EMPLOYEE = Principal("EMPLOYEE", "E001")
AS_OF = date(2026, 11, 23)


@pytest.fixture
def work_db(tmp_path):
    path = tmp_path / "work_status.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            "INSERT INTO manager (manager_id, full_name, email, password_hash) "
            "VALUES (?, ?, ?, ?)", [
                ("M001", "Manager One", "m1@example.invalid", "test"),
                ("M002", "Manager Two", "m2@example.invalid", "test"),
            ])
        connection.executemany(
            "INSERT INTO employees (employee_id, full_name, email, department, designation, "
            "manager_id, status) VALUES (?, ?, ?, ?, ?, ?, ?)", [
                ("E001", "Aarav Mehta", "a@example.invalid", "Engineering",
                 "Python Developer", "M001", "active"),
                ("E002", "Asha Verma", "b@example.invalid", "Engineering",
                 "Python Developer", "M001", "ACTIVE"),
                ("E003", "Arjun Shah", "c@example.invalid", "Data", "Data Analyst",
                 "M001", "active"),
                ("E004", "Dev Patel", "d@example.invalid", "Engineering",
                 "Python Developer", "M002", "active"),
                ("E005", "Inactive Person", "e@example.invalid", "Engineering",
                 "Developer", "M001", "inactive"),
            ])
        connection.executemany(
            "INSERT INTO skills (skill_id, skill_name, category) VALUES (?, ?, ?)", [
                (1, "Python", "Language"), (2, "SQL", "Data")])
        connection.executemany(
            "INSERT INTO employee_skills (employee_id, skill_id, proficiency_level, "
            "years_experience) VALUES (?, ?, ?, ?)", [
                ("E001", 1, "Advanced", 4), ("E002", 1, "Intermediate", 2),
                ("E003", 1, "Beginner", 1), ("E004", 1, "Advanced", 5)])
        connection.executemany(
            "INSERT INTO projects (project_id, project_name, client, start_date, end_date, "
            "status) VALUES (?, ?, ?, ?, ?, ?)", [
                (1, "Current Platform", "Example Client", None, None, "In Progress"),
                (2, "Future Upgrade", "Future Client", "2999-01-01", "2999-12-31", "Planned"),
                (3, "Finished Work", "Past Client", "2000-01-01", "2000-12-31", "Completed"),
            ])
        connection.executemany(
            "INSERT INTO project_allocation (employee_id, employee_email, project_id, "
            "project_name, manager_name) VALUES (?, ?, ?, ?, ?)", [
                ("E001", "a@example.invalid", 1, "Current Platform", "Manager One"),
                ("E002", "b@example.invalid", 2, "Future Upgrade", "Manager One"),
                ("E004", "d@example.invalid", 1, "Current Platform", "Manager Two"),
            ])
    return str(path)


def test_active_allocation_rule_and_named_employee_details(work_db):
    result = query_person_work_status("E001", HR, work_db, today=AS_OF)
    columns = {name: index for index, name in enumerate(result.columns)}
    row = result.rows[0]

    assert ACTIVE_ALLOCATION_RULE.startswith("An active allocation is")
    assert row[columns["employee_status"]] == "active"
    assert row[columns["project_name"]] == "Current Platform"
    assert row[columns["client"]] == "Example Client"
    assert row[columns["project_status"]] == "In Progress"
    assert row[columns["manager_name"]] == "Manager One"
    assert row[columns["active_allocation"]] == 1
    assert "?" in result.sql


def test_planned_or_completed_assignment_is_reported_but_not_active(work_db):
    result = query_person_work_status("E002", HR, work_db, today=AS_OF)
    columns = {name: index for index, name in enumerate(result.columns)}
    row = result.rows[0]

    assert row[columns["project_name"]] == "Future Upgrade"
    assert row[columns["project_status"]] == "Planned"
    assert row[columns["active_allocation"]] == 0


def test_named_allocation_and_availability_answers_are_sql_only(work_db):
    def no_model(prompt):
        raise AssertionError("structured work-status questions must not call the model")

    allocated = pipeline.answer_question(
        "What is Aarav Mehta's project allocation status?", HR, no_model,
        db_path=work_db)
    assert allocated.route == "sql"
    assert "Current Platform" in allocated.answer
    assert "Example Client" in allocated.answer
    assert "Manager One" in allocated.answer
    assert "active allocation" in allocated.answer
    assert allocated.sources[0]["employee_ids"] == ["E001"]

    available = pipeline.answer_question(
        "Check Asha Verma's availability", HR, no_model, db_path=work_db)
    assert "Availability: available" in available.answer
    assert "no active project allocation" in available.answer
    assert "Planned" not in available.answer

    assignment = pipeline.answer_question(
        "What are Asha Verma's allocations?", HR, no_model, db_path=work_db)
    assert "Future Upgrade" in assignment.answer
    assert "Future Client" in assignment.answer
    assert "Planned" in assignment.answer
    assert "2999-01-01 to 2999-12-31" in assignment.answer
    assert "allocating manager: Manager One" in assignment.answer


def test_status_and_availability_person_queries_resolve_names(work_db):
    def no_model(prompt):
        raise AssertionError("named status queries must not call the model")

    status = pipeline.answer_question("Check Aarav Mehta's status", HR, no_model,
                                      db_path=work_db)
    assert "Employee status: active" in status.answer
    assert status.route == "sql"

    out_of_scope = pipeline.answer_question("Check Dev Patel's status", M1, no_model,
                                            db_path=work_db)
    assert "couldn't find" in out_of_scope.answer
    assert out_of_scope.sources == []


def test_manager_scoping_applies_inside_person_status_sql(work_db):
    assert query_person_work_status("E004", M1, work_db, today=AS_OF).rows == []
    assert query_person_work_status("E004", M2, work_db, today=AS_OF).rows
    with pytest.raises(PermissionError):
        query_person_work_status("E001", EMPLOYEE, work_db, today=AS_OF)


def test_available_python_developers_and_manager_scope(work_db):
    result = query_workforce_status(
        "available Python developers in my team", M1, work_db, today=AS_OF)

    assert {row[0] for row in result.rows} == {"E002"}
    assert "e.manager_id = ?" in result.sql
    assert "E004" not in repr(result.rows)


def test_active_allocation_counts_are_complementary_and_team_scoped(work_db):
    result = query_workforce_status(
        "How many people are allocated or unallocated in my team?", M1,
        work_db, today=AS_OF)

    assert result.rows == [(1, 2)]
    assert "M001" in result.sql or "e.manager_id = ?" in result.sql


def test_allocation_lists_exclude_planned_and_foreign_team_rows(work_db):
    result = query_workforce_status("Who is allocated to a project?", M1,
                                    work_db, today=AS_OF)

    assert [row[0] for row in result.rows] == ["E001"]
    assert "Current Platform" in result.rows[0]


def test_named_work_question_parser_handles_natural_wording():
    from rag_service.people import parse_person_work_query

    assert parse_person_work_query("check Aarav Mehta's availability") == (
        "aarav mehta", "availability")
    assert parse_person_work_query("check Aarav Mehta availability") == (
        "aarav mehta", "availability")
    assert parse_person_work_query("Aarav Mehta's project allocation status") == (
        "aarav mehta", "allocation")
    assert parse_person_work_query("status of E001") == ("e001", "status")
    assert parse_person_work_query("availability for employee E001") == (
        "e001", "availability")
    assert parse_person_work_query("what project is Dev Patel allocated to") == (
        "dev patel", "allocation")


def test_rag_service_followup_uses_resolved_id_and_normal_manager_scope(work_db, monkeypatch):
    def answer_with_fixture(question, principal, llm, history):
        return pipeline.answer_question(question, principal, llm, history,
                                        db_path=work_db)

    monkeypatch.setattr(service_module, "answer_question", answer_with_fixture)
    chat = RagService(llm=lambda prompt: "must not call model")

    first = chat.chat("conversation", "Check Aarav Mehta's availability", M1)
    followup = chat.chat("conversation", "Check his status", M1)
    assert first.route == followup.route == "sql"
    assert "Aarav Mehta" in followup.answer
    assert "Employee status: active" in followup.answer

    foreign = chat.chat("conversation", "Check Dev Patel's status", M1)
    assert "couldn't find" in foreign.answer
    clarified = chat.chat("conversation", "Check his availability", M1)
    assert clarified.route == "clarify"
    assert "Which employee do you mean?" in clarified.answer


def test_unknown_structured_status_shape_gets_supported_capabilities_reply(work_db):
    result = pipeline.answer_question("What is the salary status of the company?", HR,
                                      lambda prompt: "not called", db_path=work_db)

    assert result.route == "sql"
    assert "I can answer employee profiles" in result.answer
    assert "could you rephrase" not in result.answer.lower()
