"""Run the RAG golden-question evaluation with local stubs or Gemini."""
import argparse
import json
import math
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from rag_service.access import Principal
from rag_service.eval.fixtures import EMPLOYEES, build_eval_fixture
from rag_service.llm import gemini_llm
from rag_service.pipeline import answer_question

GOLDEN_PATH = Path(__file__).with_name("golden_questions.json")
REPORT_DIR = Path(__file__).with_name("reports")


def load_golden_questions(path=GOLDEN_PATH):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _extract_question(prompt):
    marker = "LATEST MESSAGE:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\n\nReturn ONLY JSON:", 1)[0].strip()


def _history_from_prompt(prompt):
    marker = "CONVERSATION SO FAR:\n"
    if marker not in prompt or "\nLATEST MESSAGE:" not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("\nLATEST MESSAGE:", 1)[0]


def _rewrite_followup(question, history):
    if not history:
        return question
    lowered = question.lower()
    names = sorted((row[1] for row in EMPLOYEES), key=len, reverse=True)
    prior_name = next((name for name in names if name.lower() in history.lower()), None)
    if prior_name and re.search(r"\b(she|he|they|her|him|them)\b", lowered):
        return re.sub(r"\b(she|he|they|her|him|them)\b", prior_name, question,
                      flags=re.I)
    dept_match = re.search(r"\b(?:in|from)\s+([A-Z][A-Za-z ]+)", history)
    if dept_match and "that department" in lowered:
        return re.sub(r"that department", dept_match.group(1).strip(), question,
                      flags=re.I)
    return question


def _stub_route(question):
    q = (question or "").lower()
    if re.search(r"\b(hello|hi|hey|weather|capital|joke)\b", q):
        return "other"
    if re.search(r"\b(available|availability|free|unallocated)\b", q):
        return "sql"
    if re.search(r"\b(who\s+is|contact\s+of|email\s+of)\b", q) and not re.search(
            r"\b(who\s+is\s+in|who\s+is\s+manager\s+of)\b", q):
        return "person"
    structured = bool(re.search(
        r"\b(how many|count|number of|who knows|who has|who have|in my team|"
        r"team|department|employees in|employees from|developers?|engineers?|"
        r"analysts?)\b", q)) or any(
            employee[2].lower() in q for employee in EMPLOYEES)
    resume_detail = bool(re.search(
        r"\b(experience|built|build|project|projects|responsibilities|tools|"
        r"worked|work|resume|accomplishment|accomplishments)\b", q))
    if structured and resume_detail:
        return "both"
    if structured:
        return "sql"
    return "resumes"


class StubLLM:
    """Offline route and answer model for smoke-testing the harness."""
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if "You route questions" in prompt:
            question = _extract_question(prompt)
            history = _history_from_prompt(prompt)
            standalone = _rewrite_followup(question, history)
            return json.dumps({"standalone_question": standalone,
                               "route": _stub_route(standalone)})
        excerpt = re.search(
            r'<resume_excerpt employee="([^"]+)" name="([^"]+)" section="([^"]+)">',
            prompt)
        if excerpt:
            employee_id, name, section = excerpt.groups()
            return f"Resume evidence for {name} is available [{employee_id} · {section}]."
        if "DATABASE RESULT" in prompt:
            return "The database returned matching structured records [SQL]."
        return "I couldn't find supporting evidence in the supplied context."


def _employee_ids(result):
    found = set()
    for source in result.sources:
        if source.get("type") == "resume" and source.get("employee_id"):
            found.add(str(source["employee_id"]))
        found.update(str(value) for value in source.get("employee_ids", []))
    # Profile responses include a stable employee ID in their deterministic text.
    found.update(re.findall(r"\bE\d{3}\b", result.answer))
    return found


def _is_not_found(answer):
    text = (answer or "").lower()
    return "couldn't find" in text or "could not find" in text


def _p95(values):
    if not values:
        return 0.0
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _evaluate_cases(cases, llm, fixture):
    rows = []
    true_positive = false_positive = false_negative = 0
    route_correct = 0
    not_found_total = not_found_correct = 0
    scope_leaks = 0
    latencies = []

    for case in cases:
        principal = Principal(case["principal"]["role"], case["principal"]["user_id"])
        started = time.perf_counter()
        result = answer_question(
            case["question"], principal, llm, case.get("history"),
            vector_db=fixture["vector_db"], db_path=fixture["db_path"], rerank=False)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        expected_ids = set(case.get("expected_employee_ids", []))
        forbidden_ids = set(case.get("forbidden_employee_ids", []))
        actual_ids = _employee_ids(result)
        retrieved_ids = {str(source["employee_id"]) for source in result.sources
                         if source.get("type") == "resume" and source.get("employee_id")}
        if case["expected_route"] in ("resumes", "both"):
            true_positive += len(retrieved_ids & expected_ids)
            false_positive += len(retrieved_ids - expected_ids)
            false_negative += len(expected_ids - retrieved_ids)
        route_ok = result.route == case["expected_route"]
        route_correct += int(route_ok)
        not_found_actual = _is_not_found(result.answer)
        if case.get("expect_not_found", False):
            not_found_total += 1
            not_found_correct += int(not_found_actual)
        leak_ids = (actual_ids | set(re.findall(r"\bE\d{3}\b", result.answer))) & forbidden_ids
        scope_leaks += int(bool(leak_ids))
        rows.append({
            "id": case["id"], "question": case["question"],
            "route": result.route, "expected_route": case["expected_route"],
            "route_correct": route_ok,
            "expected_employee_ids": sorted(expected_ids),
            "employee_ids": sorted(actual_ids),
            "retrieved_resume_ids": sorted(retrieved_ids),
            "forbidden_employee_ids": sorted(forbidden_ids),
            "scope_leak_ids": sorted(leak_ids),
            "expect_not_found": bool(case.get("expect_not_found", False)),
            "not_found": not_found_actual,
            "not_found_correct": (not_found_actual == bool(case.get("expect_not_found", False))),
            "latency_ms": round(latency_ms, 3),
            "answer": result.answer,
        })

    precision = true_positive / float(true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / float(true_positive + false_negative) if true_positive + false_negative else 1.0
    not_found_accuracy = (not_found_correct / float(not_found_total)) if not_found_total else None
    metrics = {
        "question_count": len(rows),
        "route_accuracy": route_correct / float(len(rows)) if rows else 0.0,
        "retrieval_employee_id_precision": precision,
        "retrieval_employee_id_recall": recall,
        "not_found_accuracy": not_found_accuracy,
        "scope_leak_count": scope_leaks,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
    }
    return {"metrics": metrics, "results": rows}


def _write_reports(report, output_dir, llm_mode, real_embeddings):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report["configuration"] = {"llm": llm_mode, "real_embeddings": bool(real_embeddings)}
    json_path = output / f"rag-eval-{stamp}.json"
    markdown_path = output / f"rag-eval-{stamp}.md"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    metrics = report["metrics"]
    lines = [
        "# RAG Evaluation Report", "",
        f"- Questions: {metrics['question_count']}",
        f"- LLM: {llm_mode}",
        f"- Real embeddings: {bool(real_embeddings)}",
        f"- Route accuracy: {metrics['route_accuracy']:.1%}",
        f"- Retrieval employee ID precision: {metrics['retrieval_employee_id_precision']:.1%}",
        f"- Retrieval employee ID recall: {metrics['retrieval_employee_id_recall']:.1%}",
        (f"- Not-found accuracy: {metrics['not_found_accuracy']:.1%}"
         if metrics["not_found_accuracy"] is not None else "- Not-found accuracy: n/a"),
        f"- Scope leaks: {metrics['scope_leak_count']}",
        f"- Average latency: {metrics['average_latency_ms']:.2f} ms",
        f"- P95 latency: {metrics['p95_latency_ms']:.2f} ms", "",
        "| Case | Route (expected) | Expected IDs | Resume IDs | Not found | Leak IDs | Latency ms |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['id']} | {row['route']} ({row['expected_route']}) | "
            f"{', '.join(row['expected_employee_ids']) or '—'} | "
            f"{', '.join(row['retrieved_resume_ids']) or '—'} | {row['not_found']} | "
            f"{', '.join(row['scope_leak_ids']) or '—'} | {row['latency_ms']:.2f} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def run_evaluation(llm_mode="stub", real_embeddings=False, output_dir=REPORT_DIR,
                   golden_path=GOLDEN_PATH, workspace=None, llm=None):
    """Build temporary fixtures, evaluate them, and write JSON and Markdown reports."""
    cases = load_golden_questions(golden_path)
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="intelli-crew-rag-eval-")
        workspace = temporary.name
    try:
        fixture = build_eval_fixture(workspace, real_embeddings=real_embeddings)
        model = llm or (StubLLM() if llm_mode == "stub" else gemini_llm())
        report = _evaluate_cases(cases, model, fixture)
        json_path, markdown_path = _write_reports(report, output_dir, llm_mode, real_embeddings)
        report["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
        del fixture
        return report
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the standalone IntelliCrew RAG service.")
    parser.add_argument("--llm", choices=("stub", "gemini"), default="stub")
    parser.add_argument("--real-embeddings", action="store_true",
                        help="Use sentence-transformers/all-MiniLM-L6-v2 instead of deterministic local embeddings")
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    report = run_evaluation(args.llm, args.real_embeddings, args.output_dir)
    print(json.dumps({"metrics": report["metrics"], "report_paths": report["report_paths"]}, indent=2))
    return 1 if report["metrics"]["scope_leak_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
