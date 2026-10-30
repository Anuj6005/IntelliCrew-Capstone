# """Standalone embedding agent — scans the uploads folder, checks resume_logs,
# and embeds every file that is NOT ingested yet.

# Rules per file:
#   - ingested = 1  -> skip
#   - ingested = 0  -> embed, then set 1
#   - no row at all -> create row (0), embed, then set 1
# """
# import os
# from typing import TypedDict, List

# from langgraph.graph import StateGraph, END

# from document_loader.doc_loader import load_text, embed_text
# from resume_agent.db import (
#     get_resume_log_by_name,   # find row by file_name (or None)
#     log_resume,               # create a row if file has no entry
#     mark_resume_ingested,     # set ingested = 1
#     get_employee_name,        # lookup full_name for metadata
# )

# # uploads folder lives in project root (one level above Emp_DataAgent/)
# from pathlib import Path
# UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
# VALID_EXT = (".pdf", ".docx")


# class EmbedState(TypedDict):
#     pending: List[str]   # file paths still to process
#     embedded: int
#     skipped: int


# # 1. scan uploads folder for resume files
# def node_scan(state):
#     files = []
#     if UPLOAD_DIR.is_dir():
#         for name in os.listdir(UPLOAD_DIR):
#             if name.startswith("_tmp_"):            # ignore temp files
#                 continue
#             if name.lower().endswith(VALID_EXT):
#                 files.append(str(UPLOAD_DIR / name))

#     state["pending"] = files
#     state["embedded"] = 0
#     state["skipped"] = 0
#     return state


# # 2. process ONE file, then loop back
# def node_process_one(state):
#     file_path = state["pending"].pop(0)
#     file_name = os.path.basename(file_path)

#     # look this file up in resume_logs
#     row = get_resume_log_by_name(file_name)

#     # already ingested -> skip
#     if row and row["ingested"] == 1:
#         state["skipped"] += 1
#         return state

#     # no entry -> create one (ingested = 0)
#     if row is None:
#         log_id = log_resume(file_name=file_name, file_path=file_path, ingested=0)
#     else:
#         log_id = row["log_id"]

#     # file gone between scan and now -> mark handled, skip
#     if not os.path.exists(file_path):
#         state["skipped"] += 1
#         mark_resume_ingested(log_id)
#         return state

#     # employee_id = filename without extension (files renamed to <emp_id>.ext)
#     employee_id = os.path.splitext(file_name)[0]
#     full_name = get_employee_name(employee_id) or "unknown"   # avoid None in metadata

#     # load text + embed into Chroma
#     raw_text = load_text(file_path)
#     embed_text(raw_text, employee_id, full_name)

#     # flip flag so it never embeds twice
#     mark_resume_ingested(log_id)
#     state["embedded"] += 1
#     return state


# # loop control
# def route_more(state):
#     return "more" if state["pending"] else "done"


# def build_embed_graph():
#     g = StateGraph(EmbedState)
#     g.add_node("scan", node_scan)
#     g.add_node("process_one", node_process_one)

#     g.set_entry_point("scan")
#     g.add_conditional_edges("scan", route_more,
#                             {"more": "process_one", "done": END})
#     g.add_conditional_edges("process_one", route_more,
#                             {"more": "process_one", "done": END})
#     return g.compile()


# embed_agent = build_embed_graph()


# def run_embedding_agent():
#     """Call this after all uploads — no input needed."""
#     final = embed_agent.invoke({"pending": [], "embedded": 0, "skipped": 0})
#     return {"embedded": final["embedded"], "skipped": final["skipped"]}


"""Standalone embedding agent: scans the uploads folder and keeps Chroma in sync.

Rules per file (<employee_id>.pdf / .docx):
  - Chroma already holds chunks built from this exact file (same SHA-1)  -> skip
  - otherwise (new file, re-uploaded / changed file, or empty Chroma)    -> re-embed,
    replacing that employee's old chunks
resume_logs.ingested is still maintained so the rest of the app is unaffected.
"""
import os
from pathlib import Path
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from document_loader.doc_loader import (
    load_text, embed_resume, file_sha1, resume_is_current,
)
from resume_agent.db import (
    get_resume_log_by_name,
    log_resume,
    mark_resume_ingested,
    get_employee_profile,     # NEW: name / department / designation / manager_id
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
VALID_EXT = (".pdf", ".docx")


class EmbedState(TypedDict):
    pending: List[str]
    embedded: int
    skipped: int


def node_scan(state):
    files = []
    if UPLOAD_DIR.is_dir():
        for name in os.listdir(UPLOAD_DIR):
            if name.startswith("_tmp_"):
                continue
            if name.lower().endswith(VALID_EXT):
                files.append(str(UPLOAD_DIR / name))
    state["pending"] = files
    state["embedded"] = 0
    state["skipped"] = 0
    return state


def node_process_one(state):
    file_path = state["pending"].pop(0)
    file_name = os.path.basename(file_path)

    row = get_resume_log_by_name(file_name)
    log_id = row["log_id"] if row else log_resume(
        file_name=file_name, file_path=file_path, ingested=0)

    if not os.path.exists(file_path):          # vanished between scan and now
        state["skipped"] += 1
        mark_resume_ingested(log_id)
        return state

    employee_id = os.path.splitext(file_name)[0]   # files are renamed to <emp_id>.ext
    source_hash = file_sha1(file_path)

    if resume_is_current(employee_id, source_hash):
        state["skipped"] += 1
        mark_resume_ingested(log_id)
        return state

    profile = get_employee_profile(employee_id) or {"full_name": "unknown"}
    profile.update({
        "employee_id": employee_id,
        "source_file": file_name,
        "source_hash": source_hash,
    })
    embed_resume(load_text(file_path), profile)

    mark_resume_ingested(log_id)
    state["embedded"] += 1
    return state


def route_more(state):
    return "more" if state["pending"] else "done"


def build_embed_graph():
    g = StateGraph(EmbedState)
    g.add_node("scan", node_scan)
    g.add_node("process_one", node_process_one)
    g.set_entry_point("scan")
    g.add_conditional_edges("scan", route_more, {"more": "process_one", "done": END})
    g.add_conditional_edges("process_one", route_more,
                            {"more": "process_one", "done": END})
    return g.compile()


embed_agent = build_embed_graph()


def run_embedding_agent():
    """Call this after all uploads, no input needed."""
    final = embed_agent.invoke({"pending": [], "embedded": 0, "skipped": 0})
    return {"embedded": final["embedded"], "skipped": final["skipped"]}