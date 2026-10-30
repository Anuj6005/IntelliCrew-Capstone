"""Section-aware resume chunking (pure Python, no Chroma / torch imports).

Splits a resume into sections (summary, experience, skills, ...), drops the
header block (name / contact / boilerplate; that data lives in SQLite), and
prefixes every chunk with "Name | Designation | Department | Section" so a chunk
is always attributable to a person when it is retrieved on its own.
"""
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# canonical section -> accepted heading texts (matched on a whole line, case-insensitive)
SECTION_ALIASES = {
    "summary": ["professional summary", "summary", "profile", "career objective",
                "objective", "about me"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "internship", "internships"],
    "education": ["education", "academic background", "academics", "qualifications"],
    "skills": ["skills", "technical skills", "key skills", "core competencies",
               "tools & technologies", "tools and technologies"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates", "licenses & certifications",
                       "licenses and certifications", "courses", "training"],
    "achievements": ["achievements", "awards", "honors", "accomplishments",
                     "publications"],
    "other": ["languages", "interests", "hobbies", "references"],
}
_HEADING_TO_SECTION = {h: sec for sec, hs in SECTION_ALIASES.items() for h in hs}

MAX_CHUNK_CHARS = 900
OVERLAP = 100
_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=OVERLAP)


def _heading_of(line: str):
    """Return the canonical section if the whole line is a known heading, else None."""
    key = re.sub(r"[:\-–—_\s]+$", "", line.strip().lower())
    return _HEADING_TO_SECTION.get(key)


def split_sections(text: str):
    """Return [(section, body)]. Text before the first heading is section 'header'."""
    sections, current, buf = [], "header", []
    for line in text.splitlines():
        sec = _heading_of(line)
        if sec:
            if "".join(buf).strip():
                sections.append((current, "\n".join(buf).strip()))
            current, buf = sec, []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current, "\n".join(buf).strip()))
    # merge neighbours with the same canonical section (e.g. PROFILE + PROFESSIONAL SUMMARY)
    merged = []
    for sec, body in sections:
        if merged and merged[-1][0] == sec:
            merged[-1] = (sec, merged[-1][1] + "\n" + body)
        else:
            merged.append((sec, body))
    return merged


def build_chunks(text: str, profile: dict):
    """Return [{'text': str, 'metadata': dict}] ready for Chroma.

    profile keys: employee_id, full_name, department, designation, manager_id,
                  source_file, source_hash   (missing values become "")
    """
    sections = split_sections(text)
    real = [s for s in sections if s[0] != "header"]
    if not real:                                   # no headings found: keep everything
        sections = [("general", text.strip())] if text.strip() else []
    else:                                          # drop the header / boilerplate block
        sections = real

    who = " | ".join(p for p in (profile.get("full_name"), profile.get("designation"),
                                 profile.get("department")) if p)
    chunks, idx = [], 0
    for section, body in sections:
        for piece in _splitter.split_text(body):
            prefix = f"{who} | {section.upper()}\n" if who else f"{section.upper()}\n"
            meta = {
                "employee_id": str(profile.get("employee_id") or ""),
                "name": profile.get("full_name") or "",
                "department": profile.get("department") or "",
                "designation": profile.get("designation") or "",
                "manager_id": profile.get("manager_id") or "",
                "section": section,
                "source_file": profile.get("source_file") or "",
                "source_hash": profile.get("source_hash") or "",
                "chunk_index": idx,
            }
            chunks.append({"text": prefix + piece, "metadata": meta})
            idx += 1
    return chunks
