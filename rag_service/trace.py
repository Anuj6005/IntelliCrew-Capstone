"""Privacy-conscious JSONL tracing for RAG requests."""
import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from .settings import SETTINGS

LOG_PATH = Path(__file__).resolve().parent / "logs" / "requests.jsonl"
_LOG_LOCK = threading.Lock()
_ACTIVE_TRACE = ContextVar("rag_active_trace", default=None)


class RequestTrace:
    """Collect fields for one request without retaining answer or resume contents."""

    def __init__(self, role: str, question: str):
        self.request_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.role = role
        self.question = str(question or "")[:2000]
        self.route = None
        self.stage_latencies_ms = {}
        self.sql_row_count = 0
        self.retrieved_employee_ids = set()
        self.answer_length = 0
        self.error = None

    def add_stage_time(self, name: str, elapsed: float) -> None:
        self.stage_latencies_ms[name] = round(
            self.stage_latencies_ms.get(name, 0.0) + elapsed * 1000, 3)

    def set_result(self, result) -> None:
        self.route = result.route
        self.answer_length = len(result.answer or "")
        if result.route == "error":
            self.error = "service_error"
        for source in result.sources or ():
            if source.get("type") == "sql":
                self.sql_row_count += int(source.get("rows") or 0)
            employee_id = source.get("employee_id")
            if employee_id:
                self.retrieved_employee_ids.add(str(employee_id))
            self.retrieved_employee_ids.update(
                str(value) for value in source.get("employee_ids", ()) if value)

    def as_dict(self) -> dict:
        event = {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "role": self.role,
            "route": self.route,
            "stage_latencies_ms": self.stage_latencies_ms,
            "sql_row_count": self.sql_row_count,
            "retrieved_employee_ids": sorted(self.retrieved_employee_ids),
            "answer_length": self.answer_length,
            "error": self.error,
        }
        if SETTINGS.trace_log_questions:
            event["question"] = self.question
        return event


def _write_event(event: dict) -> None:
    """Append one JSON object; logging failures must not fail a chat request."""
    try:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with _LOG_LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8", newline="\n") as log_file:
                log_file.write(line + "\n")
    except (OSError, TypeError, ValueError):
        return


@contextmanager
def request_trace(role: str, question: str):
    """Write one event at the outer request boundary; nested stages reuse it."""
    active = _ACTIVE_TRACE.get()
    if active is not None:
        yield active
        return

    trace = RequestTrace(role, question)
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    except Exception as error:
        trace.error = type(error).__name__
        raise
    finally:
        _ACTIVE_TRACE.reset(token)
        _write_event(trace.as_dict())


@contextmanager
def stage(name: str):
    """Accumulate elapsed milliseconds for a named pipeline stage."""
    trace = _ACTIVE_TRACE.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if trace is not None:
            trace.add_stage_time(name, time.perf_counter() - started)
