from datetime import datetime, timedelta, timezone
import sys
from types import ModuleType

from fastapi.testclient import TestClient

# app.py imports unrelated agents at module load. Keep API tests isolated from
# their model/network initialization; the chat routes themselves use a fake RAG.
_STUBBED_MODULES = []


def _stub_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    _STUBBED_MODULES.append(name)
    return module


_stub_module("security", verify_password=lambda *args: True)
_stub_module("fetch_sqlite_data", __path__=[])
_stub_module("fetch_sqlite_data.dashboard_data",
             get_manager_dashboard=lambda *args: {},
             get_hr_dashboard=lambda *args: {}, _initials=lambda *args: "")
_stub_module("fetch_sqlite_data.requirement_data", get_projects=lambda: [])
_stub_module("resume_agent", __path__=[])
_stub_module("resume_agent.embed_agent", run_embedding_agent=lambda *args: {})
_stub_module("orchestrator", __path__=[])
_stub_module("orchestrator.orchestrator", run=lambda *args, **kwargs: {})
_stub_module("skill_analyze", __path__=[])
_stub_module("skill_analyze.allocation_helper", save_project_allocations=lambda *args: {})

import app as app_module
for _module_name in _STUBBED_MODULES:
    sys.modules.pop(_module_name, None)
from rag_service.pipeline import RagAnswer


class FakeRagService:
    def __init__(self):
        self.chats = []
        self.clears = []
        self.histories = []

    def chat(self, conversation_id, message, principal):
        self.chats.append((conversation_id, message, principal))
        return RagAnswer("A safe answer", "sql", sources=[{"source": "SQL"}])

    def clear(self, conversation_id, principal):
        self.clears.append((conversation_id, principal))

    def history(self, conversation_id, principal):
        self.histories.append((conversation_id, principal))
        return [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]


def _session(role="HR", user_id="H001"):
    sid = "test-session-token"
    app_module.SESSIONS[sid] = {
        "role": role,
        "user_id": user_id,
        "name": "Test User",
        "expires": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return sid


def test_chat_uses_session_identity_and_returns_answer(monkeypatch):
    service = FakeRagService()
    monkeypatch.setattr(app_module, "_get_rag_service", lambda: service)
    app_module._chat_request_times.clear()
    sid = _session("MANAGER", "M001")

    response = TestClient(app_module.app).post(
        "/api/chat", cookies={app_module.COOKIE: sid},
        json={"message": "Who is in my team?", "role": "HR", "user_id": "H999"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A safe answer", "route": "sql", "sources": [{"source": "SQL"}]}
    conversation_id, message, principal = service.chats[0]
    assert conversation_id != sid
    assert message == "Who is in my team?"
    assert principal.role == "MANAGER"
    assert principal.user_id == "M001"


def test_chat_requires_login_and_allowed_role(monkeypatch):
    client = TestClient(app_module.app)
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    sid = _session("EMPLOYEE", "E001")
    assert client.post("/api/chat", cookies={app_module.COOKIE: sid},
                       json={"message": "hello"}).status_code == 403


def test_chat_requires_json_and_valid_message(monkeypatch):
    sid = _session()
    client = TestClient(app_module.app)
    assert client.post("/api/chat", cookies={app_module.COOKIE: sid},
                       content="message=hello").status_code == 415
    assert client.post("/api/chat", cookies={app_module.COOKIE: sid},
                       json={"message": 3}).status_code == 400


def test_chat_rate_limit_is_twenty_per_session(monkeypatch):
    service = FakeRagService()
    monkeypatch.setattr(app_module, "_get_rag_service", lambda: service)
    app_module._chat_request_times.clear()
    sid = _session()
    client = TestClient(app_module.app)

    for _ in range(20):
        assert client.post("/api/chat", cookies={app_module.COOKIE: sid},
                           json={"message": "hello"}).status_code == 200
    response = client.post("/api/chat", cookies={app_module.COOKIE: sid},
                           json={"message": "hello"})
    assert response.status_code == 429


def test_clear_endpoint_clears_session_owned_conversation(monkeypatch):
    service = FakeRagService()
    monkeypatch.setattr(app_module, "_get_rag_service", lambda: service)
    sid = _session()

    response = TestClient(app_module.app).post(
        "/api/chat/clear", cookies={app_module.COOKIE: sid}, json={})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    conversation_id, principal = service.clears[0]
    assert conversation_id != sid
    assert principal.user_id == "H001"


def test_chat_history_returns_only_current_session_history(monkeypatch):
    service = FakeRagService()
    monkeypatch.setattr(app_module, "_get_rag_service", lambda: service)
    sid = _session("MANAGER", "M001")

    response = TestClient(app_module.app).get(
        "/api/chat/history", cookies={app_module.COOKIE: sid})

    assert response.status_code == 200
    assert response.json() == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    conversation_id, principal = service.histories[0]
    assert conversation_id == app_module._session_key(sid)
    assert principal.role == "MANAGER"
    assert principal.user_id == "M001"


def test_chat_history_requires_login_and_allowed_role():
    client = TestClient(app_module.app)
    assert client.get("/api/chat/history").status_code == 401
    sid = _session("EMPLOYEE", "E001")
    assert client.get("/api/chat/history", cookies={app_module.COOKIE: sid}).status_code == 403


def test_chat_page_requires_login_and_allowed_role():
    client = TestClient(app_module.app)
    response = client.get("/ask-intellicrew", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"

    sid = _session("EMPLOYEE", "E001")
    assert client.get("/ask-intellicrew", cookies={app_module.COOKIE: sid}).status_code == 403


def test_chat_page_renders_role_specific_prompts_for_manager():
    sid = _session("MANAGER", "M001")

    response = TestClient(app_module.app).get(
        "/ask-intellicrew", cookies={app_module.COOKIE: sid})

    assert response.status_code == 200
    assert "Ask IntelliCrew" in response.text
    assert "Who is in my team?" in response.text
    assert 'aria-current="page"' in response.text

    hr_sid = _session("HR", "H001")
    hr_response = TestClient(app_module.app).get(
        "/ask-intellicrew", cookies={app_module.COOKIE: hr_sid})
    assert hr_response.status_code == 200
    assert "Who knows Python?" in hr_response.text


def test_home_sidebar_shows_chat_link_only_for_hr_and_manager():
    client = TestClient(app_module.app)
    for role, user_id in (("HR", "H001"), ("MANAGER", "M001")):
        sid = _session(role, user_id)
        response = client.get("/home", cookies={app_module.COOKIE: sid})
        assert response.status_code == 200
        assert 'href="/ask-intellicrew"' in response.text

    sid = _session("EMPLOYEE", "E001")
    response = client.get("/home", cookies={app_module.COOKIE: sid})
    assert response.status_code == 200
    assert 'href="/ask-intellicrew"' not in response.text
