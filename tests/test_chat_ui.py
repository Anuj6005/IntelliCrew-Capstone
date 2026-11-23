from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dashboards_replace_floating_chat_widget_with_navigation_link():
    for filename in ("hr_dashboard.html", "manager_dashboard.html"):
        template = (ROOT / "frontend" / "templates" / filename).read_text(
            encoding="utf-8")
        assert 'href="/ask-intellicrew"' in template
        assert "Ask IntelliCrew" in template
        assert "rag-chat-launcher" not in template
        assert "rag-chat-panel" not in template
        assert 'href="/static/chat.css"' not in template
        assert 'src="/static/chat.js"' not in template


def test_home_sidebar_links_to_chat_for_hr_and_manager_only():
    template = (ROOT / "frontend" / "templates" / "home.html").read_text(
        encoding="utf-8")
    assert 'href="/ask-intellicrew"' in template
    assert '{% if userType | lower in ("hr", "manager") %}' in template
    assert "Ask IntelliCrew" in template


def test_full_page_chat_has_history_composer_examples_and_active_navigation():
    template = (ROOT / "frontend" / "templates" / "chat.html").read_text(
        encoding="utf-8")
    assert 'id="rag-chat-messages"' in template
    assert 'id="rag-chat-form"' in template
    assert 'id="rag-chat-input"' in template
    assert 'id="rag-chat-send"' in template
    assert 'id="rag-chat-new"' in template
    assert 'data-prompt="{{ example }}"' in template
    assert 'aria-current="page"' in template
    assert 'href="/static/chat.css"' in template
    assert 'src="/static/chat.js" defer' in template


def test_chat_renderer_uses_safe_text_nodes_and_supports_history():
    script = (ROOT / "frontend" / "static" / "chat.js").read_text(
        encoding="utf-8")
    assert ".innerHTML" not in script
    assert "textContent" in script
    assert "document.createTextNode" in script
    assert 'fetch("/api/chat/history"' in script
    assert 'fetch("/api/chat/clear"' in script
    assert 'fetch("/api/chat"' in script


def test_chat_page_styles_cover_responsive_full_height_layout():
    stylesheet = (ROOT / "frontend" / "static" / "chat.css").read_text(
        encoding="utf-8")
    assert "overflow: hidden" in stylesheet
    assert ".chat-sidebar" in stylesheet
    assert ".chat-composer" in stylesheet
    assert "@media (max-width: 760px)" in stylesheet
