"""Dataset validation, reproducible run identities, and offline reports."""
from __future__ import annotations

import os

import hashlib
import json
from collections import Counter
from pathlib import Path

from vector_index import canonical_label, clean_text


def prediction_metrics(rows, prediction_key):
    labeled = [row for row in rows if clean_text(row.get("gold_label"))]
    pairs = [(canonical_label(row["gold_label"]),
              canonical_label(row.get(prediction_key))) for row in labeled]
    labels = sorted({value for pair in pairs for value in pair if value})
    per_class = {}
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        support = sum(gold == label for gold, _ in pairs)
        predicted = sum(pred == label for _, pred in pairs)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"support": support, "precision": precision,
                            "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0}
    failed = sum(not pred for _, pred in pairs)
    return {
        "evaluated_rows": len(pairs),
        "predicted_rows": len(pairs) - failed,
        "failed_rows": failed,
        "failure_rate": failed / len(pairs) if pairs else None,
        "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels) if labels else None,
        "macro_f1_label_policy": "union_of_gold_and_nonempty_predictions",
        "per_class": per_class,
    }


def grouped_metrics(rows, prediction_key):
    result = {}
    for key in ("domain", "inference_domain", "label_status"):
        result[key] = {
            value: prediction_metrics([row for row in rows if row.get(key) == value], prediction_key)
            for value in sorted({row.get(key, "") for row in rows} - {"", None})
        }
    return result


def candidate_counts(rows, key):
    counts = [len(row.get(key, [])) for row in rows]
    return {"min": min(counts, default=0), "max": max(counts, default=0),
            "mean": sum(counts) / len(counts) if counts else 0,
            "distribution": dict(sorted(Counter(counts).items()))}


def dataset_audit(examples, standards):
    categories = {canonical_label(item.category) for item in standards}
    missing = sorted({item.gold_label for item in examples
                      if canonical_label(item.gold_label) not in categories})
    if missing:
        raise ValueError(f"测试标签不在语料库中（规范化后）：{missing}")
    return {"rows": len(examples), "original_documents": len(standards),
            "unique_categories": len(categories),
            "test_categories": len({canonical_label(item.gold_label) for item in examples}),
            "domains": dict(Counter(item.domain for item in examples)),
            "label_status": dict(Counter(item.label_status for item in examples)),
            "input_mode": "field_name_only", "missing_labels": missing}


def run_manifest(args, input_path, corpus_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    excluded = {"output_dir", "cache_dir", "skip_rerank", "validate_only", "input_json",
                "input_xlsx", "corpus", "dataset_id", "abbreviation_cache_dir"}
    parameters = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key not in excluded}
    source_dir = Path(__file__).resolve().parent
    sources = {p.name: digest(p) for p in sorted(source_dir.glob("*.py"))
               if not p.name.startswith("test_")}
    corpus_paths = [corpus_path] if isinstance(corpus_path, Path) else list(corpus_path)
    identity = {"version": 2, "input_sha256": digest(input_path),
                "corpus_sha256": [digest(path) for path in corpus_paths], "input_mode": "field_name_only",
                "parameters": parameters, "source_sha256": sources}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"fingerprint": fingerprint, "identity": identity,
            "input_path": str(input_path), "corpus_paths": [str(path) for path in corpus_paths]}


def summarize_runs(run_directories):
    """Aggregate explicitly selected runs; never silently select stale results."""
    reports = {}
    for directory in run_directories:
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        dataset_id = summary.get("dataset_id", directory.parent.name)
        if dataset_id in reports:
            raise ValueError(f"同一数据集选择了多次实验：{dataset_id}")
        reports[dataset_id] = summary
    aggregate = {}
    for stage in ("retrieval", "rerank"):
        values = []
        for summary in reports.values():
            report = summary.get(stage)
            if stage == "retrieval" and report:
                report = report.get("classification_report")
            if report and report.get("evaluated_rows", 0):
                values.append(report)
        total = sum(value["evaluated_rows"] for value in values)
        aggregate[stage] = {
            "datasets_evaluated": len(values), "evaluated_rows": total,
            "sample_weighted_accuracy": sum(value["accuracy"] * value["evaluated_rows"] for value in values) / total if total else None,
            "mean_dataset_accuracy": sum(value["accuracy"] for value in values) / len(values) if values else None,
            "mean_dataset_macro_f1": sum(value["macro_f1"] for value in values) / len(values) if values else None,
            "failed_rows": sum(value["failed_rows"] for value in values),
        }
    return {"datasets": reports, "aggregate": aggregate,
            "note": "mean_dataset_macro_f1 为各数据集 Macro-F1 的等权平均，不是混合标签后的 Macro-F1"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="汇总明确指定的实验目录")
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize_runs(args.run_directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
