import pytest

from rag_service.settings import RagSettings


def test_settings_defaults_preserve_existing_limits():
    settings = RagSettings.from_env({})

    assert settings.max_distance == 0.9275
    assert settings.rerank is True
    assert settings.sql_max_rows == 200
    assert settings.sql_time_limit_s == 5.0
    assert settings.history_turns == 6
    assert settings.top_k == 20
    assert settings.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.answer_table_max_rows == 50


def test_settings_read_all_supported_environment_overrides():
    settings = RagSettings.from_env({
        "RAG_MAX_DISTANCE": "0.9",
        "RAG_RERANK": "false",
        "RAG_MAX_ROWS": "120",
        "RAG_TIME_LIMIT_S": "2.5",
        "RAG_HISTORY_TURNS": "4",
        "RAG_TOP_K": "12",
        "RAG_RERANK_MODEL": "test/reranker",
        "RAG_TABLE_MAX_ROWS": "25",
    })

    assert settings.max_distance == 0.9
    assert settings.rerank is False
    assert settings.sql_max_rows == 120
    assert settings.sql_time_limit_s == 2.5
    assert settings.history_turns == 4
    assert settings.top_k == 12
    assert settings.rerank_model == "test/reranker"
    assert settings.answer_table_max_rows == 25


@pytest.mark.parametrize("name,value", [
    ("RAG_MAX_DISTANCE", "0"),
    ("RAG_RERANK", "sometimes"),
    ("RAG_MAX_ROWS", "-1"),
    ("RAG_TIME_LIMIT_S", "nan"),
    ("RAG_HISTORY_TURNS", "0"),
    ("RAG_TOP_K", "invalid"),
    ("RAG_RERANK_MODEL", " "),
    ("RAG_TABLE_MAX_ROWS", "0"),
])
def test_settings_reject_invalid_environment_values(name, value):
    with pytest.raises(ValueError):
        RagSettings.from_env({name: value})
