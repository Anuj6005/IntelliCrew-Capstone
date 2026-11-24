import json

import pytest

from rag_service import pipeline, service as service_module, trace
from rag_service.access import Principal
from rag_service.pipeline import RagAnswer
from rag_service.settings import RagSettings
from rag_service.service import RagService
from rag_service.sql_tool import SqlResult


HR = Principal("HR", "H001")


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pipeline_logs_structured_fields_without_question_or_content(
        monkeypatch, tmp_path):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(trace, "LOG_PATH", log_path)
    monkeypatch.setattr(trace, "SETTINGS", RagSettings.from_env({}))
    monkeypatch.setattr(
        pipeline, "query_workforce_status",
        lambda *args, **kwargs: SqlResult(
            "SELECT employee_id FROM employees",
            ["employee_id", "full_name", "designation"],
            [("E123", "Test Person", "Engineer")]))

    question = "PRIVATE_QUESTION_MARKER who is available?"
    answer = pipeline.answer_question(question, HR, lambda prompt: "PRIVATE_ANSWER_MARKER")

    [event] = _events(log_path)
    assert answer.route == "sql"
    assert event["request_id"]
    assert event["role"] == "HR"
    assert event["route"] == "sql"
    assert event["stage_latencies_ms"]["pipeline"] >= 0
    assert event["stage_latencies_ms"]["sql"] >= 0
    assert event["sql_row_count"] == 1
    assert event["retrieved_employee_ids"] == ["E123"]
    assert event["answer_length"] == len(answer.answer)
    assert event["error"] is None
    assert "question" not in event
    serialized = log_path.read_text(encoding="utf-8")
    assert "PRIVATE_QUESTION_MARKER" not in serialized
    assert "PRIVATE_ANSWER_MARKER" not in serialized
    assert "Test Person" not in serialized


def test_pipeline_logs_question_only_when_local_debug_setting_is_enabled(
        monkeypatch, tmp_path):
    log_path = tmp_path / "debug.jsonl"
    monkeypatch.setattr(trace, "LOG_PATH", log_path)
    monkeypatch.setattr(
        trace, "SETTINGS", RagSettings.from_env({"RAG_TRACE_LOG_QUESTIONS": "true"}))

    pipeline.answer_question("LOCAL_DEBUG_QUESTION", Principal("EMPLOYEE", "E001"), None)

    [event] = _events(log_path)
    assert event["question"] == "LOCAL_DEBUG_QUESTION"
    assert event["route"] == "denied"


def test_trace_never_writes_answer_or_resume_text(monkeypatch, tmp_path):
    log_path = tmp_path / "content.jsonl"
    monkeypatch.setattr(trace, "LOG_PATH", log_path)
    monkeypatch.setattr(trace, "SETTINGS", RagSettings.from_env({}))

    with trace.request_trace("HR", "question") as event:
        event.set_result(RagAnswer(
            "PRIVATE_FULL_ANSWER",
            "resumes",
            sources=[{"type": "resume", "employee_id": "E123",
                      "text": "PRIVATE_RAW_RESUME_TEXT"}],
        ))

    serialized = log_path.read_text(encoding="utf-8")
    assert "PRIVATE_FULL_ANSWER" not in serialized
    assert "PRIVATE_RAW_RESUME_TEXT" not in serialized
    assert "E123" in serialized


def test_pipeline_logs_error_type_without_exception_details(monkeypatch, tmp_path):
    log_path = tmp_path / "errors.jsonl"
    monkeypatch.setattr(trace, "LOG_PATH", log_path)
    monkeypatch.setattr(trace, "SETTINGS", RagSettings.from_env({}))

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE_EXCEPTION_DETAILS")

    monkeypatch.setattr(pipeline, "_answer_question_impl", fail)
    with pytest.raises(RuntimeError, match="PRIVATE_EXCEPTION_DETAILS"):
        pipeline.answer_question("PRIVATE_ERROR_QUESTION", HR, lambda prompt: "unused")

    [event] = _events(log_path)
    assert event["error"] == "RuntimeError"
    assert event["stage_latencies_ms"]["pipeline"] >= 0
    serialized = log_path.read_text(encoding="utf-8")
    assert "PRIVATE_EXCEPTION_DETAILS" not in serialized
    assert "PRIVATE_ERROR_QUESTION" not in serialized


def test_service_retry_writes_one_event_for_the_chat_request(monkeypatch, tmp_path):
    log_path = tmp_path / "service.jsonl"
    monkeypatch.setattr(trace, "LOG_PATH", log_path)
    monkeypatch.setattr(trace, "SETTINGS", RagSettings.from_env({}))
    monkeypatch.setattr(service_module.time, "sleep", lambda seconds: None)
    calls = []

    def answer(*args):
        calls.append(args)
        if len(calls) == 1:
            raise TimeoutError("PRIVATE_TIMEOUT_DETAILS")
        return RagAnswer("A safe answer", "resumes", sources=[
            {"type": "resume", "employee_id": "E123", "section": "experience"}])

    monkeypatch.setattr(service_module, "answer_question", answer)
    chat = RagService(llm=lambda prompt: "unused")
    result = chat.chat("conversation", "PRIVATE_CHAT_QUESTION", HR)

    events = _events(log_path)
    assert result.answer == "A safe answer"
    assert len(calls) == 2
    assert len(events) == 1
    assert events[0]["route"] == "resumes"
    assert events[0]["retrieved_employee_ids"] == ["E123"]
    assert events[0]["error"] is None
    serialized = log_path.read_text(encoding="utf-8")
    assert "PRIVATE_CHAT_QUESTION" not in serialized
    assert "PRIVATE_TIMEOUT_DETAILS" not in serialized
