"""Portable JSON and human-readable reports; unreviewed is never pass."""

from html import escape
import json
from pathlib import Path


def _block(value):
    return "<pre>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def render_report(result: dict) -> str:
    lines = ["# RAG evaluation", "",
        f"Dataset: **{result['dataset_kind']}**. Execution: **{result['execution_kind']}**.", "",
        "Synthetic/mocked runs do not measure real RAG quality. Human correctness, faithfulness, "
        "and citation support remain unreviewed until a person assesses them.", "",
        "## Retrieval", "", "Failure-inclusive macro averages; unanswerable questions excluded.", "",
        "| Method | K | Recall@K | MRR@K | Answerable | Errors (all queries) |",
        "|---|---:|---:|---:|---:|---:|"]
    def display(value):
        return "N/A" if value is None else f"{value:.4f}"
    for method, summary in result["summary"]["retrieval"].items():
        for k, metrics in summary["metrics"].items():
            lines.append(f"| {method} | {k} | {display(metrics['recall_failure_inclusive'])} | "
                f"{display(metrics['mrr_failure_inclusive'])} | {metrics['answerable_count']} | {summary['error_count']} |")
    lines += ["", "## Latency (seconds)", "",
              "p95 uses nearest rank and is shown only for at least 20 samples; still exploratory for small datasets.", "",
              "| Stage (successful requests) | Count | Mean | Median | p95 |",
              "|---|---:|---:|---:|---:|"]
    latencies = {}
    for method, summary in result["summary"]["retrieval"].items():
        latencies[method + " including query embedding"] = summary["successful_inclusive_latency"]
        latencies[method + " database only"] = summary["successful_database_latency"]
    latencies.update(result["summary"]["generation"]["latency"])
    for stage, summary in latencies.items():
        lines.append(f"| {stage} | {summary['count']} | {display(summary['mean_s'])} | "
                     f"{display(summary['median_s'])} | {display(summary['p95_s'])} |")
    lines += ["", "## Generation summary", "", _block(result["summary"]["generation"]),
              "", "## Reproducibility", "", _block({key: result[key] for key in
              ("created_utc", "dataset_sha256", "corpus", "settings")}), "", "## Question review", ""]
    for row in result["questions"]:
        question = row["question"]
        lines += [f"### {question['id']}", "", "Question, expected answer/facts, sources, and relevance labels:",
                  "", _block(question), "", "Retrieval rankings, scores, diagnostics, and errors:", "",
                  _block(row["retrieval"]), "", "Generated answer, evidence quotes, citations, context, and checks:",
                  "", _block(row["generation"]), "",
                  "Human review: correctness / faithfulness / citation support: **not_reviewed**. Notes: ______", ""]
    return "\n".join(lines) + "\n"


def write_reports(result: dict, directory: str | Path):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(render_report(result), encoding="utf-8")
    review = {"dataset_sha256": result["dataset_sha256"], "created_utc": result["created_utc"],
        "questions": [{"id": row["question"]["id"], "correctness": "not_reviewed",
            "faithfulness": "not_reviewed", "citation_support": "not_reviewed", "notes": ""}
            for row in result["questions"] if row["generation"] is not None]}
    (output / "human_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
