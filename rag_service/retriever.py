"""Resume retrieval: scope current Chroma chunks, optionally cut off, rerank and group."""
import re
from dataclasses import dataclass, field

from .access import Principal, require_source, scoped_employee_ids, DB_PATH
from .settings import SETTINGS

# Keep in sync with document_loader.doc_loader.CURRENT_ONLY (ingestion side).
CURRENT_ONLY = {"is_current": True}

# Distances use Chroma's squared L2 on normalised MiniLM vectors. The global open-question
# cutoff (0.9276) comes from synthetic MiniLM calibration; person-scoped questions skip it.
MAX_DISTANCE = SETTINGS.max_distance
RERANK_DEFAULT = SETTINGS.rerank
RERANK_MODEL = SETTINGS.rerank_model


@dataclass
class Hit:
    employee_id: str
    name: str
    department: str
    designation: str
    section: str
    version: int
    text: str
    distance: float
    rerank_score: float = None


@dataclass
class EmployeeEvidence:
    employee_id: str
    name: str
    department: str
    designation: str
    hits: list = field(default_factory=list)


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL)
        except Exception as e:                       # offline / not installed: skip reranking
            print(f"[rag] reranker unavailable ({e}); using vector order")
            _reranker = False
    return _reranker or None


def _default_vector_db():
    from document_loader.doc_loader import get_vector_db   # heavy import: only when needed
    return get_vector_db()


def _build_filter(ids):
    if ids is None:
        return dict(CURRENT_ONLY)
    return {"$and": [CURRENT_ONLY, {"employee_id": {"$in": sorted(ids)}}]}


def visible_ids(principal: Principal, restrict_ids=None, db_path=DB_PATH):
    """None = no id restriction (HR, nothing narrowed). Otherwise the allowed id set."""
    scope = scoped_employee_ids(principal, db_path)
    if scope is None:
        return None if restrict_ids is None else set(restrict_ids)
    scope = set(scope)
    return scope if restrict_ids is None else scope & set(restrict_ids)


def search_resumes(query: str, principal: Principal, *, k: int = SETTINGS.top_k, restrict_ids=None,
                   max_distance: float = None, use_distance_cutoff: bool = True,
                   rerank: bool = None, vector_db=None,
                   db_path=DB_PATH) -> list:
    """Return ranked Hits; enforce scope in retrieval and optionally skip distance filtering."""
    require_source(principal, "resumes")
    ids = visible_ids(principal, restrict_ids, db_path)
    if ids is not None and not ids:
        return []

    db = vector_db or _default_vector_db()
    if not use_distance_cutoff:
        collection = getattr(db, "_collection", None)
        count = getattr(collection, "count", None)
        if callable(count):
            # With a metadata filter, this asks the vector store for all chunks in scope
            # before reranking, rather than truncating a long resume at the ordinary top-k.
            k = max(k, int(count()))
    raw = db.similarity_search_with_score(query, k=k, filter=_build_filter(ids))

    cutoff = MAX_DISTANCE if max_distance is None else max_distance
    hits = []
    for doc, dist in raw:
        if use_distance_cutoff and dist > cutoff:
            continue
        m = doc.metadata
        hits.append(Hit(
            employee_id=m.get("employee_id", ""), name=m.get("name", ""),
            department=m.get("department", ""), designation=m.get("designation", ""),
            section=m.get("section", ""), version=m.get("version", 1),
            text=doc.page_content, distance=float(dist)))

    hits.sort(key=lambda h: h.distance)
    if hits and (RERANK_DEFAULT if rerank is None else rerank):
        model = _get_reranker()
        if model:
            scores = model.predict([(query, h.text) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = float(s)
            hits.sort(key=lambda h: -h.rerank_score)
    return hits


def prioritize_section(query: str, hits: list) -> list:
    """Move chunks matching an explicit resume-detail intent ahead of other sections."""
    text = (query or "").lower()
    priorities = (
        ("certifications", r"\b(certifications?|certificates?|licenses?)\b"),
        ("education", r"\b(education|degree|college|university|academics?)\b"),
        ("projects", r"\b(projects?|project work)\b"),
        ("skills", r"\b(skills?|proficiencies)\b"),
        ("experience", r"\b(intern(ship)?s?|trainees?|experience|worked|working|"
                        r"built|build|responsibilities|employment)\b"),
    )
    section = next((name for name, pattern in priorities if re.search(pattern, text)), None)
    if section is None:
        return hits
    return sorted(hits, key=lambda hit: (
        hit.section.lower() != section,
        -(hit.rerank_score if hit.rerank_score is not None else 0),
        hit.distance if hit.rerank_score is None else 0))


def group_by_employee(hits: list, per_employee: int = 3, max_employees: int = 5) -> list:
    """Chunks -> one evidence block per person, ordered by their best chunk."""
    order, groups = [], {}
    for h in hits:
        if h.employee_id not in groups:
            if len(groups) >= max_employees:
                continue
            groups[h.employee_id] = EmployeeEvidence(h.employee_id, h.name, h.department,
                                                      h.designation)
            order.append(h.employee_id)
        g = groups[h.employee_id]
        if len(g.hits) < per_employee:
            g.hits.append(h)
    return [groups[i] for i in order]
