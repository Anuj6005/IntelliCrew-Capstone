# IntelliCrew

IntelliCrew is an AI-powered HR & workforce management platform that automates resume parsing, skill extraction, employee-to-project matching, video summarization, PDF reporting, and allocation emails — all coordinated through a central multi-agent orchestrator built with **LangGraph**.

It's built for two kinds of users:
- **HR** — uploads employee resumes, which are automatically parsed and stored.
- **Managers** — search employees by skill, match employees to project requirements, review AI-generated rankings, allocate teams, and trigger notification emails.

---

## ✨ Key Features

- **Login system** for two roles — Manager (`M001`, `M002`, …) and HR (`H001`, `H002`, …) — with scrypt-hashed passwords and cookie-based sessions.
- **Resume upload & parsing (HR)** — PDF/DOCX resumes are parsed with an LLM (Gemini) to extract name, department, designation, skills, and more, then stored in SQLite. If no employee ID is found on the resume, HR can manually assign one.
- **Resume embeddings** — parsed resume text is chunked and embedded into a local **ChromaDB** vector store (via HuggingFace `sentence-transformers/all-MiniLM-L6-v2`) for semantic search.
- **Skill search** — free-text queries ("find employees who know Kubernetes and Terraform") are routed to a matching agent that searches stored skills.
- **Project ↔ employee matching** — given a project's required skills, employees are ranked by skill-match percentage, with an LLM-generated description for each ranked candidate and an overall skill-gap report.
- **Project allocation** — managers select final employees for a project; selections are saved to the database.
- **Automated allocation emails** — a mail agent sends assignment emails (via Gmail SMTP) to newly allocated employees and marks them as notified.
- **Video summarization** — upload a video file or paste a YouTube link; audio is transcribed (OpenAI Whisper / YouTube Transcript API) and summarized (HuggingFace `distilbart-cnn-12-6`).
- **Centralized PDF report** — generates an organization-wide report (projects, employees, skills, allocations) with an LLM-written narrative, rendered to PDF with ReportLab.
- **HR & Manager dashboards** — role-specific dashboards built with Jinja2 templates.

---

## 🧠 Architecture

At the core of the app is a lightweight **orchestrator** that inspects the user's request (a natural-language string plus a flag for whether a file was attached) and routes it to the right LangGraph agent:

```
app.py (FastAPI)
   │
   ▼
orchestrator/orchestrator.py
   │  choose_agent(user_input, has_file) → agent_name
   ▼
orchestrator/registry.py  (keyword-based agent registry)
   │
   ├── resume_agent        → parse resume, extract fields, store skills, embed to Chroma
   ├── matching_agent       → skill-match employees against a project's requirements
   ├── report_agent         → build the org-wide PDF report
   ├── mail_agent           → send pending project-allocation emails
   └── summarizer_agent     → transcribe & summarize a video file or YouTube link
```

Each agent is its own small **LangGraph** state machine (nodes + edges), so every workflow (load → extract → validate → persist → …) is explicit and easy to extend.

### Tech stack

| Layer | Technology |
|---|---|
| Backend / API | FastAPI, Uvicorn |
| Agent orchestration | LangGraph, LangChain |
| LLM | Google Gemini (`langchain-google-genai`) |
| Embeddings & vector store | HuggingFace `sentence-transformers`, ChromaDB |
| Document parsing | `pypdf`, `docx2txt`, `langchain_community` loaders |
| Video transcription | OpenAI Whisper, `youtube-transcript-api` |
| Summarization | HuggingFace `transformers` (`distilbart-cnn-12-6`) |
| Database | SQLite (via `sqlite3` + SQLAlchemy) |
| PDF generation | ReportLab |
| Frontend | Jinja2 templates, HTML/CSS/JS (server-rendered) |
| Auth | Custom scrypt password hashing + cookie sessions |
| Email | Gmail SMTP (`smtplib`) |

---

## 📁 Project Structure

```
IntelliCrew-Capstone/
├── app.py                     # FastAPI app: routes, auth, dashboards, uploads
├── config.py                  # Settings loaded from environment variables
├── security.py                # Password hashing / verification (scrypt)
├── data_seed.py                # Creates & seeds the SQLite database
├── document_loader/
│   └── doc_loader.py          # PDF/DOCX text extraction + Chroma embedding
├── orchestrator/
│   ├── orchestrator.py        # Runs the chosen agent, tags the response
│   ├── registry.py            # Agent registry with routing keywords
│   ├── router.py               # Decides which agent handles a request
│   └── agent_log_helper.py
├── resume_agent/
│   ├── resume_agent.py        # LangGraph: parse & store a resume
│   ├── llm_extractor.py       # Gemini-based structured extraction
│   ├── embed_agent.py         # Batch embedding of processed resumes
│   └── db.py                   # Employee/skill DB helpers
├── skill_analyze/
│   ├── matching_agent.py       # LangGraph: employee ↔ project matching
│   ├── skill_match.py          # Skill-match scoring logic
│   ├── llm_extractor.py        # LLM descriptions & skill-gap reports
│   └── allocation_helper.py    # Saves manager-selected allocations
├── report_agent/
│   ├── report_graph.py         # LangGraph: build the centralized report
│   ├── report_data.py          # Aggregates data for the report
│   ├── llm_report.py           # LLM-written narrative
│   └── pdf_builder.py           # Renders the PDF with ReportLab
├── mailAgent/
│   ├── mailAgent.py             # LangGraph: send allocation emails
│   └── send_agent.py            # Pending-email lookup / mark-as-sent
├── video_summarization/
│   ├── summarize.py             # LangGraph: transcribe + summarize video
│   └── db.py                     # Logs video summaries
├── fetch_sqlite_data/
│   ├── dashboard_data.py         # HR / Manager dashboard queries
│   └── requirement_data.py       # Project listing queries
├── frontend/
│   ├── templates/                # login, home, dashboards, resume, summarizer
│   └── static/
├── videos/                        # Uploaded/sample videos
├── requirements.txt
└── .env.example
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API key
- A Gmail account with an [App Password](https://support.google.com/accounts/answer/185833) (for the mail agent)
- ~2–4 GB free disk for the Whisper/HuggingFace/transformer model downloads on first run

### 1. Clone the repository

```bash
git clone https://github.com/Anuj6005/IntelliCrew-Capstone.git
cd IntelliCrew-Capstone
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_gmail_app_password
GEMINI_API_KEY=your_gemini_api_key
```

Optional overrides (see `config.py`):

```env
GEMINI_MODEL=gemini-2.0-flash
DB_URL=sqlite:///data/employees.db
MAX_FIX_ATTEMPTS=3
```

### 4. Initialize the database

This creates `data/employee_records.db` and seeds sample managers, HR users, and projects:

```bash
python data_seed.py
```

### 5. Run the app

```bash
uvicorn app:app --reload
```

The app will be available at **http://127.0.0.1:8000**.

### 6. Log in

Use one of the seeded accounts (see `data_seed.py` for the full list and their passwords), for example:

| Role | ID format | Example |
|---|---|---|
| Manager | `M00X` | `M001` |
| HR | `H00X` | `H001` |

---

## 🔌 Key API Endpoints

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `POST` | `/api/login` | Authenticate and start a session | Public |
| `POST` | `/api/logout` | End the current session | Authenticated |
| `GET` | `/api/me` | Get current session info | Authenticated |
| `POST` | `/api/process-resumes` | Upload & parse resumes | HR only |
| `POST` | `/api/find-employees` | Free-text skill search | Authenticated |
| `GET` | `/api/projects` | List all projects | Authenticated |
| `POST` | `/api/project-matches` | Match employees to a project | Authenticated |
| `POST` | `/api/project-allocations` | Save selected employees & trigger emails | Manager only |
| `POST` | `/api/summarize-video` | Summarize an uploaded video or link | Authenticated |
| `GET` | `/api/centralized-report` | Download the org-wide PDF report | Authenticated |
| `GET` | `/hr/dashboard` | HR dashboard page | HR only |
| `GET` | `/manager/dashboard` | Manager dashboard page | Manager only |

---

## 🗺️ Roadmap Ideas

- Replace in-memory sessions with a persistent session store (e.g. Redis) for multi-instance deployments
- Add automated tests around the orchestrator's routing logic
- Support additional resume formats and richer skill taxonomies
- Add role-based fine-grained permissions beyond HR/Manager

---

## 📄 License

No license file is currently included in this repository — add one (e.g. MIT) if you plan to share or open-source this project.
