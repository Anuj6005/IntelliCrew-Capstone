import threading

import pytest

from rag_service.access import Principal
from rag_service.pipeline import RagAnswer
from rag_service import service as service_module
from rag_service.service import InMemoryConversationStore, RagService, UNAVAILABLE_REPLY


HR = Principal("HR", "H001")
MANAGER = Principal("MANAGER", "M001")


def test_conversation_history_is_isolated_by_principal(monkeypatch):
    calls = []

    def fake_answer(question, principal, llm, history):
        calls.append((question, principal, list(history)))
        return RagAnswer("answer", "other")

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    chat = RagService(llm=lambda prompt: "unused")

    chat.chat("session-conversation", "first", HR)
    chat.chat("session-conversation", "manager message", MANAGER)

    assert calls[0][2] == []
    assert calls[1][2] == []
    assert chat._store.get("session-conversation", HR) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]
    assert chat._store.get("session-conversation", MANAGER) == [
        {"role": "user", "content": "manager message"},
        {"role": "assistant", "content": "answer"},
    ]


def test_history_accessor_returns_only_owner_messages_and_a_copy():
    chat = RagService(llm=lambda prompt: "unused")
    chat._store.append("conversation", HR, "user", "HR question")
    chat._store.append("conversation", HR, "assistant", "HR answer")
    chat._store.append("conversation", MANAGER, "user", "Manager question")

    history = chat.history("conversation", HR)
    history[0]["content"] = "changed by caller"

    assert chat.history("conversation", HR) == [
        {"role": "user", "content": "HR question"},
        {"role": "assistant", "content": "HR answer"},
    ]
    assert chat.history("conversation", MANAGER) == [
        {"role": "user", "content": "Manager question"},
    ]
    assert chat.history("missing", HR) == []


def test_history_is_limited_to_existing_history_turn_setting():
    store = InMemoryConversationStore(history_turns=2)
    for index in range(4):
        store.append("c1", HR, "user", f"q{index}")
        store.append("c1", HR, "assistant", f"a{index}")

    assert store.get("c1", HR) == [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_expired_conversation_is_removed(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(service_module.time, "monotonic", lambda: now[0])
    store = InMemoryConversationStore(ttl_seconds=5, history_turns=2)
    store.append("c1", HR, "user", "old")

    now[0] = 14.9
    assert store.get("c1", HR)[0]["content"] == "old"
    now[0] = 20.0
    assert store.get("c1", HR) == []


def test_clear_only_removes_the_callers_conversation():
    store = InMemoryConversationStore()
    store.append("c1", HR, "user", "HR history")
    store.append("c1", MANAGER, "user", "Manager history")

    store.clear("c1", MANAGER)

    assert store.get("c1", MANAGER) == []
    assert store.get("c1", HR) == [{"role": "user", "content": "HR history"}]


def test_store_append_is_thread_safe():
    store = InMemoryConversationStore(history_turns=100)
    workers = [threading.Thread(target=store.append,
                                args=("c1", HR, "user", str(index)))
               for index in range(60)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(store.get("c1", HR)) == 60


def test_message_guards_remove_controls_and_reject_empty_or_oversized(monkeypatch):
    questions = []
    monkeypatch.setattr(
        service_module, "answer_question",
        lambda question, principal, llm, history: (
            questions.append(question) or RagAnswer("ok", "other")))
    chat = RagService(llm=lambda prompt: "unused", max_message_length=5)

    chat.chat("c1", "  he\x07llo  ", HR)
    assert questions == ["hello"]
    with pytest.raises(ValueError, match="Enter a message"):
        chat.chat("c1", " \x00\x07 ", HR)
    with pytest.raises(ValueError, match="5 characters"):
        chat.chat("c1", "123456", HR)
    assert len(questions) == 1


def test_transient_failure_retries_once_and_does_not_leak_error(monkeypatch):
    calls = []

    def fake_answer(question, principal, llm, history):
        calls.append(question)
        if len(calls) == 1:
            raise TimeoutError("private prompt and secret")
        return RagAnswer("success", "other")

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    monkeypatch.setattr(service_module.time, "sleep", lambda seconds: None)
    chat = RagService(llm=lambda prompt: "unused")

    result = chat.chat("c1", "hello", HR)

    assert result.answer == "success"
    assert len(calls) == 2
    assert chat._store.get("c1", HR)[-1]["content"] == "success"


@pytest.mark.parametrize("error", [
    PermissionError("403 private credential"),
    RuntimeError("quota details include a secret"),
])
def test_permanent_llm_failures_return_generic_reply_without_retry_or_history(
        monkeypatch, error):
    calls = []

    def fake_answer(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    chat = RagService(llm=lambda prompt: "unused")

    result = chat.chat("c1", "hello", HR)

    assert result.answer == UNAVAILABLE_REPLY
    assert result.route == "error"
    assert len(calls) == 1
    assert chat._store.get("c1", HR) == []


def test_service_resolves_followup_pronoun_to_last_employee_id(monkeypatch):
    questions = []

    def fake_answer(question, principal, llm, history):
        questions.append(question)
        return RagAnswer("structured answer", "sql", sources=[
            {"type": "sql", "employee_ids": ["E001"]}])

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    chat = RagService(llm=lambda prompt: "unused")

    chat.chat("c1", "Who is Aarav Mehta?", HR)
    chat.chat("c1", "Check his availability", HR)

    assert questions == ["Who is Aarav Mehta?", "Check E001's availability"]


def test_service_asks_which_employee_when_context_is_ambiguous(monkeypatch):
    calls = []

    def fake_answer(question, principal, llm, history):
        calls.append(question)
        return RagAnswer("two employees", "sql", sources=[
            {"type": "sql", "employee_ids": ["E001", "E002"]}])

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    chat = RagService(llm=lambda prompt: "unused")
    chat.chat("c1", "Who is in my team?", HR)
    result = chat.chat("c1", "Check her status", HR)

    assert result.route == "clarify"
    assert "Which employee do you mean?" in result.answer
    assert "E001, E002" in result.answer
    assert calls == ["Who is in my team?"]


def test_service_does_not_guess_a_pronoun_without_context(monkeypatch):
    called = []
    monkeypatch.setattr(service_module, "answer_question",
                        lambda *args: called.append(args))
    chat = RagService(llm=lambda prompt: "unused")

    result = chat.chat("c1", "Check their availability", HR)

    assert result.route == "clarify"
    assert result.answer == "Which employee do you mean? Please provide a name or employee ID."
    assert called == []


def test_service_clears_followup_state_and_isolates_principals(monkeypatch):
    questions = []

    def fake_answer(question, principal, llm, history):
        questions.append((question, principal.user_id))
        return RagAnswer("answer", "sql", sources=[
            {"type": "sql", "employee_ids": ["E001"]}])

    monkeypatch.setattr(service_module, "answer_question", fake_answer)
    chat = RagService(llm=lambda prompt: "unused")
    chat.chat("c1", "Find Aarav", HR)
    no_context = chat.chat("c1", "Check his status", MANAGER)
    assert no_context.route == "clarify"

    chat.clear("c1", HR)
    cleared = chat.chat("c1", "Check her availability", HR)
    assert cleared.route == "clarify"
    assert questions == [("Find Aarav", "H001")]


def test_followup_employee_state_expires_with_conversation_ttl(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(service_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        service_module, "answer_question",
        lambda *args: RagAnswer("answer", "sql", sources=[
            {"type": "sql", "employee_ids": ["E001"]}]))
    chat = RagService(llm=lambda prompt: "unused")
    chat.chat("c1", "Find Aarav", HR)

    now[0] += service_module.CONVERSATION_TTL_SECONDS
    result = chat.chat("c1", "Check his availability", HR)

    assert result.route == "clarify"
    assert "Which employee do you mean?" in result.answer

