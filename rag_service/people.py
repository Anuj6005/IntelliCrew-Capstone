"""Basic profile lookup by name: no model-written SQL, minimal fields, team-scoped."""
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path

from .access import DB_PATH, Principal, employee_scope

MAX_MATCHES = 6
FUZZY_MIN = 0.65       # per-token similarity needed to accept a typo'd name

_PREFIXES = re.compile(
    r"^(?:who\s+is|who's|who\s+was|tell\s+me\s+about|details\s+(?:of|for|about)|"
    r"info(?:rmation)?\s+(?:on|about|of)|profile\s+of|contact\s+(?:details\s+)?(?:of|for)|"
    r"show\s+me|show|find|search\s+for|look\s*up|get)\s+(.+)$")
_SUFFIXES = re.compile(r"(?:'s|s')?\s+(?:manager|email|contact|designation|details|profile|id)$")
_TITLES = re.compile(r"^(?:mr|mrs|ms|dr|shri|smt)\.?\s+")
_NAME_TOKEN = re.compile(r"[^\W\d_][\w'.-]*")
_STOP = {
    "who", "what", "which", "how", "when", "where", "why", "is", "are", "was", "were", "do",
    "does", "did", "can", "could", "should", "will", "would", "a", "an", "the", "in", "on",
    "at", "for", "of", "to", "with", "and", "or", "from", "by", "about", "me", "my", "our",
    "list", "all", "any", "every", "employee", "employees", "people", "person", "team",
    "skill", "skills", "experience", "years", "has", "have", "knows", "know", "many", "much",
    "give", "tell", "count", "number", "hello", "hi", "hey", "thanks", "thank", "please",
}


def parse_name_query(message: str):
    """-> (name or None, explicit).

    A name is found when the message is just a name ("aarav mehta", "E001") or a simple lookup
    phrase ("who is X", "tell me about X", "contact of X", "X's manager"). `explicit` is True
    when a lookup phrase was used, which is when typo tolerance is safe to apply. Questions
    about skills, counts, departments etc. contain stop-words and are NOT treated as names.
    """
    text = re.sub(r"[?!.,]+\s*$", "", (message or "").strip().lower())
    if re.fullmatch(r"[a-z]\d{2,}", text):
        return text, True
    explicit = False
    m = _PREFIXES.match(text)
    if m:
        text, explicit = m.group(1), True
    stripped = _SUFFIXES.sub("", text).strip()
    explicit = explicit or stripped != text
    text = _TITLES.sub("", stripped).strip()
    tokens = _NAME_TOKEN.findall(text)
    if not 1 <= len(tokens) <= 4 or " ".join(tokens) != re.sub(r"\s+", " ", text):
        return None, False
    if any(t in _STOP for t in tokens):
        return None, False
    return " ".join(tokens), explicit


def name_candidate(message: str):
    return parse_name_query(message)[0]


def manager_of_query(message: str):
    """Return the employee name in a direct 'who is X's manager?' question."""
    match = re.search(
        r"\b(?:who\s+is|who's)\s+(?:the\s+)?manager\s+(?:of|for)\s+(.+?)[?!.,]*$",
        (message or "").strip(), re.I)
    return match.group(1).strip() if match else None


def team_owner_query(message: str):
    """Return a named team owner, if a question explicitly names one."""
    text = (message or "").strip()
    match = re.search(r"\b(?:in|of|for|under)\s+(.+?)['’]s\s+team\b", text, re.I)
    if match:
        owner = match.group(1).strip()
        if owner.lower() not in {"my", "our", "your", "their", "this", "the"}:
            return owner
    match = re.search(r"\b(?:reports?\s+to|reporting\s+to)\s+(.+?)[?!.,]*$", text, re.I)
    if match and match.group(1).strip().lower() not in {"me", "you", "myself"}:
        return match.group(1).strip()
    return None


def name_matches(candidate: str, full_name: str) -> bool:
    """Allow modest spelling errors when checking a named owner against the session user."""
    candidate_tokens = [t.lower() for t in re.findall(r"[\w'.-]+", candidate or "") if len(t) >= 3]
    name_tokens = [t.lower() for t in re.findall(r"[\w'.-]+", full_name or "")]
    return bool(candidate_tokens and name_tokens) and all(
        max((SequenceMatcher(None, token, name).ratio() for name in name_tokens), default=0) >= FUZZY_MIN
        for token in candidate_tokens)


def find_own_manager(principal: Principal, db_path=DB_PATH):
    """Return the signed-in manager's basic profile; never looks up another manager."""
    if principal.role != "MANAGER":
        return None
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT manager_id, full_name, email FROM manager WHERE manager_id = ?",
            (principal.user_id,)).fetchone()
        if not row:
            return None
        return {"manager_id": row[0], "name": row[1], "email": row[2]}
    finally:
        conn.close()


def is_self_profile_query(message: str) -> bool:
    """Recognize basic questions about the signed-in manager account."""
    text = re.sub(r"[?!.,]+\s*$", "", (message or "").strip().lower())
    return bool(re.fullmatch(
        r"(?:who am i|what is my name|what's my name|my profile|show my profile|"
        r"what is my email|what's my email|my email|my manager id)", text))


def _esc(token: str) -> str:
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def find_person(name: str, principal: Principal, db_path=DB_PATH, limit: int = MAX_MATCHES,
                fuzzy: bool = True):
    """Return up to `limit` dicts: employee_id, name, designation, department, email,
    manager_id, manager_name. Only people this principal may see can ever match."""
    text = (name or "").strip().lower()
    if re.fullmatch(r"[a-z]\d{2,}", text):                 # an employee id such as e001
        where, params = "LOWER(e.employee_id) = ?", [text]
    else:
        tokens = [t for t in re.findall(r"[\w'.-]+", text) if len(t) >= 2][:4]
        if not tokens:
            return []
        where = " AND ".join("LOWER(e.full_name) LIKE ? ESCAPE '\\'" for _ in tokens)
        params = [f"%{_esc(t)}%" for t in tokens]

    base = ("SELECT e.employee_id, e.full_name, e.designation, e.department, e.email, "
            "e.manager_id, m.full_name FROM employees e "
            "LEFT JOIN manager m ON m.manager_id = e.manager_id ")
    scope, scope_params = employee_scope(principal)
    if principal.role == "MANAGER":
        scope = "e.manager_id = ?"
    else:
        scope = "1 = 1"
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(base + f"WHERE ({scope}) AND ({where}) "
                            "ORDER BY e.full_name LIMIT ?",
                            [*scope_params, *params, int(limit)]).fetchall()
        if fuzzy and not rows and not re.fullmatch(r"[a-z]\d{2,}", text):    # typo tolerance
            rows = _fuzzy(conn.execute(base + f"WHERE {scope}", scope_params).fetchall(),
                          text, limit)
    finally:
        conn.close()
    keys = ("employee_id", "name", "designation", "department", "email",
            "manager_id", "manager_name")
    return [dict(zip(keys, r)) for r in rows]


def _fuzzy(rows: list, query: str, limit: int) -> list:
    """Typo-tolerant match among the people this principal can already see."""
    q_tokens = [t.lower() for t in re.findall(r"[\w'.-]+", query) if len(t) >= 3][:4]
    if not q_tokens:
        return []
    scored = []
    for r in rows:
        name_tokens = re.findall(r"[\w'.-]+", (r[1] or "").lower())
        best = [max((SequenceMatcher(None, qt, nt).ratio() for nt in name_tokens), default=0)
                for qt in q_tokens]
        if min(best) >= FUZZY_MIN:
            scored.append((sum(best) / len(best), r))
    scored.sort(key=lambda x: (-x[0], x[1][1] or ""))
    return [r for _, r in scored[:limit]]
