"""Small chat facade with owner-isolated, in-memory conversation history."""
import time
import unicodedata
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Protocol
import re

from .access import Principal
from .pipeline import RagAnswer, answer_question
from .settings import SETTINGS

CONVERSATION_TTL_SECONDS = 30 * 60
UNAVAILABLE_REPLY = "The assistant is temporarily unavailable, please try again."


class ConversationStore(Protocol):
    """Storage interface that can be replaced by a shared store such as Redis."""

    def get(self, conversation_id: str, principal: Principal) -> list:
        ...

    def append(self, conversation_id: str, principal: Principal,
               role: str, content: str) -> None:
        ...

    def clear(self, conversation_id: str, principal: Principal) -> None:
        ...


@dataclass
class _Conversation:
    messages: list
    updated_at: float


class InMemoryConversationStore:
    """Thread-safe conversation history partitioned by conversation and principal."""

    def __init__(self, *, ttl_seconds: float = CONVERSATION_TTL_SECONDS,
                 history_turns: int = SETTINGS.history_turns):
        if ttl_seconds <= 0 or history_turns <= 0:
            raise ValueError("Conversation TTL and history turns must be positive")
        self._ttl_seconds = ttl_seconds
        # answer_question currently treats the setting as a cap on history entries.
        self._max_messages = history_turns
        self._records = {}
        self._lock = RLock()

    @staticmethod
    def _key(conversation_id: str, principal: Principal):
        return conversation_id, principal.role, principal.user_id

    def _remove_expired(self, now: float):
        expired = [key for key, record in self._records.items()
                   if now - record.updated_at >= self._ttl_seconds]
        for key in expired:
            self._records.pop(key, None)

    def get(self, conversation_id: str, principal: Principal) -> list:
        now = time.monotonic()
        key = self._key(conversation_id, principal)
        with self._lock:
            self._remove_expired(now)
            record = self._records.get(key)
            if record is None:
                return []
            record.updated_at = now
            return [dict(message) for message in record.messages]

    def append(self, conversation_id: str, principal: Principal,
               role: str, content: str) -> None:
        now = time.monotonic()
        key = self._key(conversation_id, principal)
        with self._lock:
            self._remove_expired(now)
            record = self._records.setdefault(key, _Conversation([], now))
            record.messages.append({"role": role, "content": content})
            record.messages = record.messages[-self._max_messages:]
            record.updated_at = now

    def clear(self, conversation_id: str, principal: Principal) -> None:
        now = time.monotonic()
        key = self._key(conversation_id, principal)
        with self._lock:
            self._remove_expired(now)
            self._records.pop(key, None)


class RagService:
    """Application facade for validated chat messages and conversation history."""

    def __init__(self, llm: Callable[[str], str] = None,
                 store: ConversationStore = None, *, max_message_length: int = None):
        if llm is None:
            from .llm import gemini_llm
            llm = gemini_llm()
        self._llm = llm
        self._store = store or InMemoryConversationStore()
        self._max_message_length = (SETTINGS.max_message_length if max_message_length is None
                                    else max_message_length)
        self._last_people = {}
        self._state_lock = RLock()

    @staticmethod
    def _state_key(conversation_id: str, principal: Principal):
        return conversation_id, principal.role, principal.user_id

    def _recent_people(self, key):
        now = time.monotonic()
        with self._state_lock:
            expired = [state_key for state_key, (updated, _) in self._last_people.items()
                       if now - updated >= CONVERSATION_TTL_SECONDS]
            for state_key in expired:
                self._last_people.pop(state_key, None)
            record = self._last_people.get(key)
            if record is None:
                return []
            self._last_people[key] = (now, record[1])
            return list(record[1])

    def _remember_people(self, key, answer):
        employee_ids = set()
        for source in getattr(answer, "sources", ()) or ():
            if source.get("employee_id"):
                employee_ids.add(str(source["employee_id"]))
            employee_ids.update(str(value) for value in source.get("employee_ids", ())
                                if value)
        if employee_ids:
            with self._state_lock:
                self._last_people[key] = (time.monotonic(), sorted(employee_ids))
            return True
        return False

    def _forget_people(self, key):
        with self._state_lock:
            self._last_people.pop(key, None)

    @staticmethod
    def _has_person_pronoun(message: str) -> bool:
        return bool(re.search(
            r"\b(his|her|their|he|she|him|them|they|that\s+person)\b", message, re.I))

    @staticmethod
    def _replace_person_pronouns(message: str, employee_id: str) -> str:
        def replacement(match):
            word = match.group(0)
            value = employee_id + "'s" if word.lower() in ("his", "her", "their") else employee_id
            return value
        return re.sub(
            r"\b(?:his|her|their|he|she|him|them|they|that\s+person)\b",
            replacement, message, flags=re.I)

    def _clean_message(self, message: str) -> str:
        if not isinstance(message, str):
            raise ValueError("Message must be text.")
        cleaned = "".join(char for char in message
                          if unicodedata.category(char) != "Cc").strip()
        if not cleaned:
            raise ValueError("Enter a message before sending.")
        if len(cleaned) > self._max_message_length:
            raise ValueError(
                f"Message must be {self._max_message_length} characters or fewer.")
        return cleaned

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        error_name = type(error).__name__.lower()
        error_text = str(error).lower()
        if any(term in error_name or term in error_text
               for term in ("403", "forbidden", "permission denied", "quota")):
            return False
        transient_terms = ("timeout", "connection", "network", "temporarily unavailable", "429",
                           "500", "502", "503", "504")
        return (isinstance(error, (TimeoutError, ConnectionError)) or
                (isinstance(error, OSError) and not isinstance(error, PermissionError)) or any(
                    term in error_name or term in error_text for term in transient_terms))

    def chat(self, conversation_id: str, message: str,
             principal: Principal) -> RagAnswer:
        """Answer a user message using only the history owned by that principal."""
        if not conversation_id:
            raise ValueError("A conversation id is required.")
        cleaned = self._clean_message(message)
        history = self._store.get(conversation_id, principal)
        state_key = self._state_key(conversation_id, principal)
        effective_message = cleaned
        if self._has_person_pronoun(cleaned):
            previous_people = self._recent_people(state_key)
            if len(previous_people) != 1:
                if previous_people:
                    choices = ", ".join(previous_people)
                    reply = (f"I have several employees in context ({choices}). Which employee "
                             "do you mean?")
                else:
                    reply = "Which employee do you mean? Please provide a name or employee ID."
                self._store.append(conversation_id, principal, "user", cleaned)
                self._store.append(conversation_id, principal, "assistant", reply)
                return RagAnswer(reply, "clarify", cleaned)
            effective_message = self._replace_person_pronouns(cleaned, previous_people[0])
        try:
            answer = answer_question(effective_message, principal, self._llm, history)
        except Exception as error:
            if self._is_transient(error):
                time.sleep(0.1)
                try:
                    answer = answer_question(effective_message, principal, self._llm, history)
                except Exception:
                    return RagAnswer(UNAVAILABLE_REPLY, "error", cleaned)
            else:
                return RagAnswer(UNAVAILABLE_REPLY, "error", cleaned)
        self._store.append(conversation_id, principal, "user", effective_message)
        self._store.append(conversation_id, principal, "assistant", answer.answer)
        remembered = self._remember_people(state_key, answer)
        if not remembered:
            from .people import parse_person_work_query
            if parse_person_work_query(effective_message)[0]:
                self._forget_people(state_key)
        return answer

    def clear(self, conversation_id: str, principal: Principal) -> None:
        """Clear history only for the owning principal."""
        if not conversation_id:
            raise ValueError("A conversation id is required.")
        self._store.clear(conversation_id, principal)
        with self._state_lock:
            self._last_people.pop(self._state_key(conversation_id, principal), None)

    def history(self, conversation_id: str, principal: Principal) -> list:
        """Return a copy of conversation messages owned by this principal."""
        if not conversation_id:
            return []
        return self._store.get(conversation_id, principal)
