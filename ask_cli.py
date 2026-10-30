"""Try the RAG pipeline in a terminal, with your real Chroma index, SQLite DB and Gemini key.

Run from the project root (app stopped, .venv active):
    python ask_cli.py --role HR --user H001
    python ask_cli.py --role MANAGER --user M001
Type a question; type 'exit' to quit. Sources and the SQL used are printed under each answer.
"""
import argparse
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
from rag_service.access import Principal
from rag_service.llm import gemini_llm
from rag_service.pipeline import answer_question

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True, choices=["HR", "MANAGER", "EMPLOYEE"])
ap.add_argument("--user", required=True, help="hr_id / manager_id, e.g. H001 or M001")
args = ap.parse_args()

principal, llm, history = Principal(args.role, args.user), gemini_llm(), []
print(f"Chatting as {principal.role} {principal.user_id}. Type 'exit' to quit.\n")
while True:
    q = input("you> ").strip()
    if q.lower() in ("exit", "quit", ""):
        break
    r = answer_question(q, principal, llm, history)
    print(f"\nbot> {r.answer}\n")
    print(f"     route={r.route}  standalone={r.standalone_question!r}")
    if r.sql:
        print(f"     sql={r.sql}" + (f"  ERROR={r.sql_error}" if r.sql_error else ""))
    for s in r.sources:
        if s["type"] == "resume":
            print(f"     - {s['employee_id']} {s['name']} · {s['section']} (distance {s['distance']})")
    print()
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]
