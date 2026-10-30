"""Rebuild the whole Chroma index from the files in uploads/.

WARNING: this deletes ALL stored resume versions (the history) and indexes only
the files currently in uploads/. Use it when the chunking logic changes, not as a
routine step. Run from the project root with the app STOPPED:
    python reembed_all.py
"""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb

from document_loader.doc_loader import CHROMA_DB, COLLECTION
from resume_agent.embed_agent import run_embedding_agent

if input("This erases all resume version history. Type yes to continue: ").strip().lower() != "yes":
    raise SystemExit("cancelled")

client = chromadb.PersistentClient(path=CHROMA_DB)
if COLLECTION in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION)
    print(f"deleted old '{COLLECTION}' collection")

print("re-embedding...", run_embedding_agent())
