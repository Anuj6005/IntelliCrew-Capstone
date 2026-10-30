# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings   # local embedding model

# CHROMA_DB = "chroma_db"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# # load the embedding model once (reused for every call)
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# def load_text(file_path: str) -> str:
#     """Load raw unstructured text from a PDF or DOCX."""
#     if file_path.lower().endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#     elif file_path.lower().endswith(".docx"):
#         loader = Docx2txtLoader(file_path)
#     else:
#         raise ValueError("Only PDF or DOCX supported")

#     docs = loader.load()
#     return "\n".join(d.page_content for d in docs)


# def embed_text(raw_text: str, employee_id: int, full_name: str):
#     """Split raw text and store chunks in ChromaDB using HuggingFace embeddings."""
#     splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#     chunks = splitter.split_text(raw_text)

#     # attach metadata so we can filter by employee later
#     metadatas = [{"employee_id": employee_id, "name": full_name} for _ in chunks]

#     # open existing store if present, else it gets created on first add
#     if Path(CHROMA_DB).exists():
#         print("Chroma directory already exists")
#         vector_db = Chroma(
#             collection_name="resumes",
#             persist_directory=CHROMA_DB,
#             embedding_function=embedding_model,
#         )
#     else:
#         vector_db = Chroma(
#             collection_name="resumes",
#             persist_directory=CHROMA_DB,
#             embedding_function=embedding_model,
#         )
#         print("Vector store created..")

#     # add the new chunks (works for both new and existing store)
#     vector_db.add_texts(texts=chunks, metadatas=metadatas)
#     return len(chunks)

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # silence Chroma telemetry noise

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings   # local embedding model

from document_loader.resume_chunker import build_chunks

CHROMA_DB = "chroma_db"
COLLECTION = "resumes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# load the embedding model once (reused for every call)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vector_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DB,
        embedding_function=embedding_model,
    )


def load_text(file_path: str) -> str:
    """Load raw unstructured text from a PDF or DOCX."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Only PDF or DOCX supported")

    docs = loader.load()
    return "\n".join(d.page_content for d in docs)


def file_sha1(file_path: str) -> str:
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# Every retrieval must use this filter so answers come from the CURRENT resume only.
# Combine it with other conditions via {"$and": [CURRENT_ONLY, {...}]}.
CURRENT_ONLY = {"is_current": True}


def _current_where(employee_id: str) -> dict:
    return {"$and": [{"employee_id": str(employee_id)}, CURRENT_ONLY]}


def resume_is_current(employee_id: str, source_hash: str) -> bool:
    """True if the CURRENT version in Chroma was built from this exact file."""
    got = get_vector_db().get(where=_current_where(employee_id), limit=1,
                              include=["metadatas"])
    metas = got.get("metadatas") or []
    return bool(metas) and metas[0].get("source_hash") == source_hash


def delete_employee_chunks(employee_id: str) -> None:
    """Remove EVERY version of one employee's chunks (use when the employee is deleted)."""
    get_vector_db().delete(where={"employee_id": str(employee_id)})


def embed_resume(raw_text: str, profile: dict) -> int:
    """Index a new resume version for an employee.

    The employee's previous chunks are NOT deleted: they are marked
    is_current=False (kept for history) and the new chunks get version+1.
    """
    employee_id = str(profile["employee_id"])
    chunks = build_chunks(raw_text, profile)
    if not chunks:
        return 0

    vector_db = get_vector_db()
    col = vector_db._collection
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    existing = col.get(where={"employee_id": employee_id}, include=["metadatas"])
    old_ids, old_metas = existing["ids"], existing["metadatas"] or []
    last_version = max((m.get("version", 0) for m in old_metas), default=0)

    # retire the previous current version (history stays searchable only on request)
    retire = [(i, m) for i, m in zip(old_ids, old_metas) if m.get("is_current", True)]
    if retire:
        col.update(
            ids=[i for i, _ in retire],
            metadatas=[{**m, "is_current": False, "superseded_at": now} for _, m in retire],
        )

    for c in chunks:
        c["metadata"].update({"is_current": True, "version": last_version + 1,
                              "indexed_at": now})
    vector_db.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def embed_text(raw_text: str, employee_id, full_name: str) -> int:
    """Backward-compatible wrapper for older callers."""
    return embed_resume(raw_text, {"employee_id": employee_id, "full_name": full_name})