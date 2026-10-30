"""Environment-backed settings for the RAG service.

Values are read once when the package imports this module. Set environment
variables before starting the service to override the defaults.
"""
import os
import math
from dataclasses import dataclass
from typing import Mapping


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = int(env.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = float(env.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _boolean(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


@dataclass(frozen=True)
class RagSettings:
    """Runtime thresholds and limits, with defaults matching the original service."""

    max_distance: float = 0.9275
    rerank: bool = True
    sql_max_rows: int = 200
    sql_time_limit_s: float = 5.0
    history_turns: int = 6
    top_k: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    answer_table_max_rows: int = 50

    @classmethod
    def from_env(cls, env: Mapping[str, str] = None) -> "RagSettings":
        """Read settings from an environment mapping (or ``os.environ``)."""
        env = os.environ if env is None else env
        model = env.get("RAG_RERANK_MODEL", cls.rerank_model).strip()
        if not model:
            raise ValueError("RAG_RERANK_MODEL must not be empty")
        return cls(
            max_distance=_positive_float(env, "RAG_MAX_DISTANCE", cls.max_distance),
            rerank=_boolean(env, "RAG_RERANK", cls.rerank),
            sql_max_rows=_positive_int(env, "RAG_MAX_ROWS", cls.sql_max_rows),
            sql_time_limit_s=_positive_float(env, "RAG_TIME_LIMIT_S", cls.sql_time_limit_s),
            history_turns=_positive_int(env, "RAG_HISTORY_TURNS", cls.history_turns),
            top_k=_positive_int(env, "RAG_TOP_K", cls.top_k),
            rerank_model=model,
            answer_table_max_rows=_positive_int(
                env, "RAG_TABLE_MAX_ROWS", cls.answer_table_max_rows),
        )


SETTINGS = RagSettings.from_env()
