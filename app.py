"""IntelliCrew — Login + HR Resume Upload + Skill Search (single app)."""

import os
import shutil
import secrets
import sqlite3
import hashlib
import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import (
    Cookie, FastAPI, HTTPException, Request, Response, status,
    UploadFile, File, Form,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from security import verify_password
from fetch_sqlite_data.dashboard_data import get_manager_dashboard,get_hr_dashboard,_initials
from resume_agent.embed_agent import run_embedding_agent

# --- orchestrator replaces the direct resume_agent import ---
from orchestrator.orchestrator import run as orchestrate

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "data" / "employee_records.db"
COOKIE = "intellicrew_session_id"
HOURS = 8

UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="IntelliCrew")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))

SESSIONS: dict[str, dict] = {}

# The RAG model and index are loaded only after an authorized user opens chat.
_rag_service = None
_rag_service_lock = threading.Lock()
_chat_request_times = {}
_chat_request_lock = threading.Lock()


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        with _rag_service_lock:
            if _rag_service is None:
                from rag_service.service import RagService
                _rag_service = RagService()
    return _rag_service


def _session_key(sid: str) -> str:
    return hashlib.sha256(sid.encode("utf-8")).hexdigest()


def _allow_chat_request(session_key: str) -> bool:
    now = time.monotonic()
    with _chat_request_lock:
        requests = _chat_request_times.setdefault(session_key, deque())
        while requests and now - requests[0] >= 60:
            requests.popleft()
        if len(requests) >= 20:
            return False
        requests.append(now)
        return True


def _require_json(request: Request):
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Send a JSON request.")


def _chat_session(sid: str | None):
    session = require_session(sid)
    if session.get("role") not in ("HR", "MANAGER"):
        raise HTTPException(403, "Chat is available to HR and managers.")
    return session


# ---------- models ----------
class LoginRequest(BaseModel):
    login_type: Literal["manager", "hr"]
    user_id: str = Field(min_length=4, max_length=4)
    password: str = Field(min_length=1, max_length=128)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


# ---------- sessions ----------
def get_session(sid: str | None):
    s = SESSIONS.get(sid) if sid else None
    if s and s["expires"] > datetime.now(timezone.utc):
        return s
    SESSIONS.pop(sid, None) if sid else None
    return None


def require_session(sid: str | None):
    s = get_session(sid)
    if not s:
        raise HTTPException(401, "Please log in first.")
    return s


# ---------- pages ----------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    if get_session(sid):
        return RedirectResponse("/home", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={})


@app.get("/home", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    s = get_session(sid)
    if not s:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"userType": s["role"].lower(), "userId": s["user_id"], "userName": s["name"]},
    )


@app.get("/ask-intellicrew", response_class=HTMLResponse, include_in_schema=False)
def ask_intellicrew(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    session = get_session(sid)
    if not session:
        return RedirectResponse("/", status_code=303)
    if session.get("role") not in ("HR", "MANAGER"):
        raise HTTPException(403, "Chat is available to HR and managers.")

    examples = (
        ["Who is in my team?", "Available Python developers in my team",
         "What did Aarav Mehta build during his internship?",
         "Is Aarav Mehta available?"]
        if session["role"] == "MANAGER" else
        ["Who knows Python?", "What are Aarav Mehta's skills?",
         "What did Aarav Mehta build during his internship?", "Who is available?"]
    )
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "request": request,
            "role": session["role"],
            "user_name": session.get("name") or session.get("full_name") or session["role"],
            "examples": examples,
        },
    )


# ---------- auth API ----------
@app.post("/api/login")
def login(payload: LoginRequest, response: Response):
    uid = payload.user_id.strip().upper()

    if payload.login_type == "manager":
        prefix, role, table, id_column = "M", "MANAGER", "manager", "manager_id"
    else:
        prefix, role, table, id_column = "H", "HR", "hr", "hr_id"

    if not (uid.startswith(prefix) and uid[1:].isdigit()):
        raise HTTPException(400, f"ID must use the {prefix}001 format.")

    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        record = conn.execute(
            f"""
            SELECT {id_column} AS id, full_name AS name, password_hash
            FROM {table}
            WHERE {id_column} = ?
            """,
            (uid,),
        ).fetchone()

    if record is None or not verify_password(payload.password, record["password_hash"]):
        raise HTTPException(401, "Invalid login type, ID, or password.")

    sid = secrets.token_urlsafe(32)
    SESSIONS[sid] = {
        "user_id": record["id"],
        "role": role,
        "name": record["name"],
        "expires": datetime.now(timezone.utc) + timedelta(hours=HOURS),
    }
    response.set_cookie(COOKIE, sid, max_age=HOURS * 3600, httponly=True, samesite="lax", path="/")
    return {"message": "Login successful.", "role": role, "redirect_url": "/home"}


@app.post("/api/logout")
def logout(response: Response, sid: str | None = Cookie(None, alias=COOKIE)):
    SESSIONS.pop(sid, None) if sid else None
    response.delete_cookie(COOKIE, path="/")
    return {"message": "Logged out.", "redirect_url": "/"}


@app.get("/api/me")
def current_user(sid: str | None = Cookie(None, alias=COOKIE)):
    s = require_session(sid)
    return {"user_id": s["user_id"], "name": s["name"], "role": s["role"]}


@app.post("/api/chat")
async def chat(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    from starlette.concurrency import run_in_threadpool
    from rag_service.access import Principal

    session = _chat_session(sid)
    _require_json(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")
    if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
        raise HTTPException(400, "A message is required.")
    session_key = _session_key(sid)
    if not _allow_chat_request(session_key):
        raise HTTPException(429, "Too many chat requests. Please wait a minute.")

    principal = Principal(session["role"], session["user_id"])
    conversation_id = session_key
    try:
        result = await run_in_threadpool(
            lambda: _get_rag_service().chat(
                conversation_id, payload["message"], principal))
    except ValueError as error:
        return Response(
            content=json.dumps({"error": str(error)}),
            status_code=422, media_type="application/json")
    if result.route == "error":
        return Response(
            content=json.dumps({"error": result.answer}),
            status_code=503, media_type="application/json")
    return {"answer": result.answer, "route": result.route, "sources": result.sources}


@app.post("/api/chat/clear")
async def clear_chat(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    from starlette.concurrency import run_in_threadpool
    from rag_service.access import Principal

    session = _chat_session(sid)
    _require_json(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")
    principal = Principal(session["role"], session["user_id"])
    await run_in_threadpool(
        lambda: _get_rag_service().clear(_session_key(sid), principal))
    return {"ok": True}


@app.get("/api/chat/history")
def chat_history(sid: str | None = Cookie(None, alias=COOKIE)):
    from rag_service.access import Principal

    session = _chat_session(sid)
    principal = Principal(session["role"], session["user_id"])
    return _get_rag_service().history(_session_key(sid), principal)


# ---------- resume upload (HR only) — now goes through the orchestrator ----------
from typing import List, Optional


# these live in Emp_DataAgent.db (helpers shown in section 4)
# from Emp_DataAgent.db import log_resume
# from Emp_DataAgent.embed_agent import run_embedding_agent   # NEW separate agent


@app.post("/api/process-resumes")
async def process_resumes(
    files: List[UploadFile] = File(...),
    employee_ids: Optional[List[str]] = Form(None),   # re-submit: "filename:empid"
    sid: str | None = Cookie(None, alias=COOKIE),
):
    s = require_session(sid)
    if s["role"] != "HR":
        raise HTTPException(403, "Only HR can upload resumes.")

    # {filename: employee_id} from re-submit
    manual_ids = {}
    if employee_ids:
        for pair in employee_ids:
            if ":" in pair:
                fname, emp = pair.split(":", 1)
                manual_ids[fname.strip()] = emp.strip()

    results = []

    # ---------- loop + store each resume ----------
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        temp_path = os.path.join(UPLOAD_DIR, f"_tmp_{file.filename}")

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        manual = manual_ids.get(file.filename)

        state = {
            "file_path": temp_path,
            "raw_text": "",
            "extracted": {"employee_id": manual} if manual else {},
            "employee_id": None,
            "status": "started",
        }

        result = orchestrate(state, user_input="process resume", has_file=True)

        # case 1: no employee id -> drop temp file, ask HR
        if result["status"] == "need_employee_id":
            if os.path.exists(temp_path):
                os.remove(temp_path)
            results.append({
                "file": file.filename,
                "status": "need_employee_id",
                "message": "No Employee ID found. Please enter it manually.",
            })
            continue

        # case 2: success -> rename temp to <employee_id>.ext
        final_emp_id = result.get("employee_id")
        final_path = os.path.join(UPLOAD_DIR, f"{final_emp_id}{ext}")
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        results.append({
            "file": file.filename,
            "status": result["status"],
            "handled_by": result.get("handled_by"),
            "employee_id": final_emp_id,
            "full_name": result["extracted"].get("full_name"),
            "designation": result["extracted"].get("designation"),
            "skills_found": len(result.get("extracted", {}).get("skills", [])),
        })

    embed_summary = run_embedding_agent()
    # summary counts for the UI
    processed = sum(1 for r in results if r["status"] != "need_employee_id")
    need_id   = sum(1 for r in results if r["status"] == "need_employee_id")

    return {
        "status": "done",
        "total": len(files),
        "processed": processed,
        "need_employee_id": need_id,
        "results": results,
    }



# ---------- skill search (any logged-in user) — free text, orchestrator decides ----------
@app.post("/api/find-employees")
def find_employees(payload: QueryRequest, sid: str | None = Cookie(None, alias=COOKIE)):
    require_session(sid)

    state = {"query": payload.question, "skills": [], "matches": [], "status": "started"}

    # no file -> orchestrator routes this to skill_agent (keyword match)
    result = orchestrate(state, user_input=payload.question, has_file=False)

    return {
        "handled_by": result.get("handled_by"),
        "skills": result.get("skills", []),
        "status": result.get("status"),
        "matches": result.get("matches", []),
    }






# ---------- HR dashboard endpoint ----------
@app.get("/hr/dashboard",response_class=HTMLResponse,include_in_schema=False,)
def hr_dashboard(request: Request,sid: str | None = Cookie(None, alias=COOKIE),):
    # Get the logged-in user's session
    session = get_session(sid)

    # Redirect to login if the session does not exist
    if not session:
        return RedirectResponse("/", status_code=303)

    # Allow only HR users
    if session["role"] != "HR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only HR can view this dashboard.",
        )

    # Get the HR dashboard data
    data = get_hr_dashboard(session["user_id"])

    # Get HR name from the session if available
    hr_name = (
        session.get("full_name")
        or session.get("name")
        or "HR"
    )

    data["hr_name"] = hr_name
    data["hr_initials"] = _initials(hr_name)

    # Render the HR dashboard HTML
    return templates.TemplateResponse(
        request=request,
        name="hr_dashboard.html",
        context={
            "request": request,
            **data,
        },
    )
    
# ---------- manager dashboard (MANAGER only) ----------
@app.get("/manager/dashboard", response_class=HTMLResponse, include_in_schema=False)
def manager_dashboard(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    s = get_session(sid)
    if not s:
        return RedirectResponse("/", status_code=303)

    # managers only — block HR from opening the manager dashboard
    if s["role"] != "MANAGER":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only managers can view this dashboard.")

    data = get_manager_dashboard(s["user_id"])   # uses the logged-in manager's id
    return templates.TemplateResponse(
        request=request,
        name="manager_dashboard.html",
        context={"request": request, **data},
    )
    
    
    
import fetch_sqlite_data.requirement_data as requirement_data

@app.get("/api/projects")
def api_projects():
    # returns: {"projects": [{project_id, project_name, client, skills:[...]}, ...]}
    return {"projects": requirement_data.get_projects()}

#from skillgap_agent.matching_agent import run_matching_agent

class ProjectMatchRequest(BaseModel):
    project_id: int = Field(gt=0)


# ---------- project employee matching ----------
@app.post("/api/project-matches")
def api_project_matches(
    payload: ProjectMatchRequest,
    sid: str | None = Cookie(None, alias=COOKIE),
):
    """Run project matching through the central orchestrator."""

    session = require_session(sid)

    state = {
        "project_id": payload.project_id,
    }

    result = orchestrate(
        state,
        user_input="match employees for selected project",
        has_file=False,
    )

    response = result.get("response", {})

    if not response.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.get(
                "message",
                "Unable to generate employee matches.",
            ),
        )

    return {
        **response,
        "handled_by": result.get("handled_by"),
    }


# ---------- video summarizer (any logged-in user) — goes through orchestrator ----------
VIDEO_DIR = BASE_DIR / "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# ---------- video summarizer page (any logged-in user) ----------
@app.get("/summarizer", response_class=HTMLResponse, include_in_schema=False)
def summarizer_page(request: Request, sid: str | None = Cookie(None, alias=COOKIE)):
    s = get_session(sid)
    if not s:
        return RedirectResponse("/", status_code=303)   # not logged in -> login
    return templates.TemplateResponse(
        request=request,
        name="summarizer.html",
        context={"userType": s["role"].lower(), "userId": s["user_id"], "userName": s["name"]},
    )

@app.post("/api/summarize-video")
async def summarize_video(
    file: UploadFile = File(None),      # optional: a video file
    link: str = Form(None),            # optional: a YouTube / video URL
    sid: str | None = Cookie(None, alias=COOKIE),
):
    s = require_session(sid)   # any logged-in user can summarize

    # --- CASE A: a video file was uploaded -> save it in videos/ ---
    if file is not None and file.filename:
        save_path = os.path.join(VIDEO_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        source = save_path
        has_file = True

    # --- CASE B: a link was pasted -> pass the URL straight through ---
    elif link and link.strip():
        source = link.strip()
        has_file = False

    else:
        raise HTTPException(400, "Please upload a video file or paste a link.")

    # --- build the state the summarizer graph expects ---
    state = {
        "source": source,
        "is_url": False,
        "transcript": "",
        "summary": "",
        "status": "started",
        "session_id": s
    }

    # keyword makes the orchestrator pick summarizer_agent (not resume_agent)
    result = orchestrate(state, user_input="summarize video", has_file=has_file)

    return {
        "handled_by": result.get("handled_by"),
        "summary": result.get("summary"),
        "status": result.get("status"),
    }
    
    
from pydantic import BaseModel


class SelectedEmployee(BaseModel):
    employee_id: str
    rank: int | None = None
    matching_percentage: float | int | None = None
    description: str | None = None          # 👈 add this line


class SaveProjectAllocationRequest(BaseModel):
    project_id: int
    employees: list[SelectedEmployee]

    
from skill_analyze.allocation_helper import save_project_allocations  
# ---------- save selected employees to project_allocation (MANAGER only) ----------
# ---------- save selected employees to project_allocation (MANAGER only) ----------
@app.post("/api/project-allocations")
def api_project_allocations(
    payload: SaveProjectAllocationRequest,
    sid: str | None = Cookie(None, alias=COOKIE),
):
    """Save ONLY the manager-selected employees into project_allocation."""

    session = require_session(sid)

    try:
        result = save_project_allocations(
            project_id=payload.project_id,
            selected_employees=[emp.dict() for emp in payload.employees],
            manager_name=session["name"],
        )

        # Backend step: route email sending through the central orchestrator.
        mail_state = {
            "status": "started",
        }

        mail_summary = orchestrate(
            mail_state,
            user_input="send project allocation emails",
            has_file=False,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to save project allocations: {str(error)}",
        )        
        
from fastapi import Response



# ---------- centralized organization report (any logged-in user) ----------
@app.get("/api/centralized-report")
def api_centralized_report(sid: str | None = Cookie(None, alias=COOKIE)):
    """Generate the org-wide report through the orchestrator and return a PDF."""
    require_session(sid)

    # route through the orchestrator, just like the other agents
    state = {"status": "started"}
    result = orchestrate(state, user_input="generate centralized report", has_file=False)

    pdf_bytes = result.get("pdf_bytes")
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate the centralized report.",
        )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="intellicrew_report.pdf"'
        },
    )
