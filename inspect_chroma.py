# """Inspect IntelliCrew's Chroma index. Run from the project root:

#     python inspect_chroma.py

# Read-only: it never writes to chroma_db.
# """
# from collections import Counter

# import chromadb

# CHROMA_DB = "chroma_db"
# COLLECTION = "resumes"
# TEST_QUERIES = [
#     "Kubernetes and Terraform experience",
#     "machine learning projects",
#     "Python backend developer",
# ]

# client = chromadb.PersistentClient(path=CHROMA_DB)
# names = [c.name for c in client.list_collections()]
# print("collections:", names)
# if COLLECTION not in names:
#     raise SystemExit(f"no '{COLLECTION}' collection found")

# col = client.get_collection(COLLECTION)
# data = col.get(include=["metadatas", "documents"])
# ids, metas, docs = data["ids"], data["metadatas"], data["documents"]

# print(f"\nchunks: {len(ids)}")
# per_emp = Counter(m.get("employee_id") for m in metas)
# print(f"distinct employee_id values: {len(per_emp)}")
# print("chunks per employee (top 10):", per_emp.most_common(10))

# keys = Counter(k for m in metas for k in m)
# print("\nmetadata keys present:", dict(keys))
# print("id types:", Counter(type(m.get("employee_id")).__name__ for m in metas))
# print("unknown names:", sum(1 for m in metas if m.get("name") in (None, "unknown")))

# lens = [len(d) for d in docs]
# print(f"\nchunk length min/avg/max: {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

# dupes = len(docs) - len(set(docs))
# print(f"exact duplicate chunks: {dupes}  (non-zero means a resume was embedded twice)")

# print("\nsample chunk:\n", docs[0][:400].replace("\n", " "))

# # Retrieval smoke test with the same embedding model used at ingest time.
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DB, embedding_function=emb)

# for q in TEST_QUERIES:
#     print(f"\nQUERY: {q}")
#     for doc, dist in db.similarity_search_with_score(q, k=4):
#         m = doc.metadata
#         print(f"  dist={dist:.3f}  {m.get('employee_id')}  {m.get('name')}  | "
#               f"{doc.page_content[:90].replace(chr(10), ' ')}")


"""Inspect IntelliCrew's Chroma index. Run from the project root:

    python inspect_chroma.py

Read-only: it never writes to chroma_db.
"""
from collections import Counter

import chromadb

CHROMA_DB = "chroma_db"
COLLECTION = "resumes"
TEST_QUERIES = [
    "Kubernetes and Terraform experience",
    "machine learning projects",
    "Python backend developer",
]

client = chromadb.PersistentClient(path=CHROMA_DB)
names = [c.name for c in client.list_collections()]
print("collections:", names)
if COLLECTION not in names:
    raise SystemExit(f"no '{COLLECTION}' collection found")

col = client.get_collection(COLLECTION)
data = col.get(include=["metadatas", "documents"])
ids, metas, docs = data["ids"], data["metadatas"], data["documents"]

print(f"\nchunks: {len(ids)}")
per_emp = Counter(m.get("employee_id") for m in metas)
print(f"distinct employee_id values: {len(per_emp)}")
print("chunks per employee (top 10):", per_emp.most_common(10))

keys = Counter(k for m in metas for k in m)
print("\nmetadata keys present:", dict(keys))
if "is_current" in keys:
    cur = sum(1 for m in metas if m.get("is_current"))
    print(f"current chunks: {cur}   superseded (history): {len(metas) - cur}")
    vers = {}
    for m in metas:
        vers.setdefault(m.get("employee_id"), set()).add(m.get("version"))
    print("versions per employee:", {k: sorted(v) for k, v in vers.items()})
    print("sections (current):", dict(Counter(m.get("section") for m in metas if m.get("is_current"))))
print("id types:", Counter(type(m.get("employee_id")).__name__ for m in metas))
print("unknown names:", sum(1 for m in metas if m.get("name") in (None, "unknown")))

lens = [len(d) for d in docs]
print(f"\nchunk length min/avg/max: {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")

dupes = len(docs) - len(set(docs))
print(f"exact duplicate chunks: {dupes}  (non-zero means a resume was embedded twice)")

print("\nsample chunk:\n", docs[0][:400].replace("\n", " "))

# Retrieval smoke test with the same embedding model used at ingest time.
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DB, embedding_function=emb)

for q in TEST_QUERIES:
    print(f"\nQUERY: {q}")
    flt = {"is_current": True} if "is_current" in keys else None
    for doc, dist in db.similarity_search_with_score(q, k=4, filter=flt):
        m = doc.metadata
        print(f"  dist={dist:.3f}  {m.get('employee_id')}  {m.get('name')}  {m.get('section', '')}  | "
              f"{doc.page_content[:90].replace(chr(10), ' ')}")