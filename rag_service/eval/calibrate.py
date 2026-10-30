"""Measure resume distance separation and recommend a cutoff without changing settings."""
import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

from rag_service.access import Principal
from rag_service.eval.fixtures import build_eval_fixture
from rag_service.eval.run import GOLDEN_PATH, load_golden_questions
from rag_service.retriever import _build_filter, visible_ids
from rag_service.settings import SETTINGS


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _distribution(values):
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None,
                "p95": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _metrics(cases, threshold):
    true_positive = false_positive = false_negative = 0
    for case in cases:
        expected = set(case["expected_employee_ids"])
        found = {employee_id for employee_id, distance in case["employee_distances"].items()
                 if distance <= threshold}
        true_positive += len(expected & found)
        false_positive += len(found - expected)
        false_negative += len(expected - found)
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else 1.0)
    recall = (true_positive / (true_positive + false_negative)
              if true_positive + false_negative else 1.0)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative}


def calibrate_cases(cases, vector_db, db_path, *, top_k=SETTINGS.top_k,
                    current_cutoff=SETTINGS.max_distance, recommend_cutoff=True):
    """Compute per-query relevant/irrelevant hits using only scoped current documents."""
    observations = []
    relevant_distances, irrelevant_distances = [], []
    for case in cases:
        if case["expected_route"] not in ("resumes", "both"):
            continue
        principal = Principal(case["principal"]["role"], case["principal"]["user_id"])
        allowed = visible_ids(principal, db_path=db_path)
        if allowed is not None and not allowed:
            continue
        documents = vector_db.similarity_search_with_score(
            case["question"], k=top_k, filter=_build_filter(allowed))
        expected = set(case.get("expected_employee_ids", []))
        per_employee = {}
        for document, raw_distance in documents:
            employee_id = str(document.metadata.get("employee_id", ""))
            if not employee_id:
                continue
            distance = float(raw_distance)
            per_employee[employee_id] = min(per_employee.get(employee_id, distance), distance)
            if employee_id in expected:
                relevant_distances.append(distance)
            else:
                irrelevant_distances.append(distance)
        observations.append({"id": case["id"], "expected_employee_ids": sorted(expected),
                             "employee_distances": per_employee})

    if not observations:
        raise ValueError("No resume or both questions are available for calibration")
    recommended = recommendation_metrics = None
    if recommend_cutoff:
        candidates = sorted({current_cutoff} | {
            distance for case in observations for distance in case["employee_distances"].values()})
        scored = [(threshold, _metrics(observations, threshold)) for threshold in candidates]
        recommended, recommendation_metrics = max(
            scored, key=lambda item: (item[1]["f1"], item[1]["precision"], -item[0]))
    return {
        "question_count": len(observations),
        "current_max_distance": current_cutoff,
        "relevant_hit_distances": _distribution(relevant_distances),
        "irrelevant_hit_distances": _distribution(irrelevant_distances),
        "current_cutoff_metrics": _metrics(observations, current_cutoff),
        "recommended_max_distance": recommended,
        "recommended_metrics": recommendation_metrics,
        "cases": [{"id": case["id"],
                   "expected_employee_ids": case["expected_employee_ids"],
                   "employee_distances": case["employee_distances"]}
                  for case in observations],
    }


def run_calibration(*, real_embeddings=False, golden_path=GOLDEN_PATH, workspace=None,
                    top_k=SETTINGS.top_k, current_cutoff=SETTINGS.max_distance):
    """Build isolated synthetic data and return calibration metrics."""
    temporary = None
    if workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="intelli-crew-rag-calibration-")
        workspace = temporary.name
    try:
        fixture = build_eval_fixture(workspace, real_embeddings=real_embeddings)
        report = calibrate_cases(load_golden_questions(golden_path), fixture["vector_db"],
                                 fixture["db_path"], top_k=top_k,
                                 current_cutoff=current_cutoff,
                                 recommend_cutoff=real_embeddings)
        report["configuration"] = {
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2"
            if real_embeddings else "deterministic-keyword",
            "top_k": top_k,
            "production_data_used": False,
            "chroma_written": False,
            "cutoff_recommendation_actionable": bool(real_embeddings),
        }
        return report
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure resume distances and recommend, but do not apply, a cutoff.")
    parser.add_argument("--real-embeddings", action="store_true",
                        help="Use the local MiniLM model instead of deterministic test embeddings")
    args = parser.parse_args(argv)
    report = run_calibration(real_embeddings=args.real_embeddings)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
