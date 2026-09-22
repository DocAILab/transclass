#!/usr/bin/env python3
"""LLM-rerank two persisted BGE-M3 finance recall experiments.

The module consumes saved retrieval candidates instead of rebuilding the
embedding index. Column I remains offline evaluation data and is never placed
in an LLM prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from retrieval import (
    DEFAULT_OUTPUT_DIR as DEFAULT_RECALL_DIR,
    expand_standards_with_fragments,
)
from llm_client import (
    JsonlRerankCache,
    OpenAICompatibleReranker,
    _provider_defaults,
    load_corpus,
)
from vector_index import canonical_label, clean_text, macro_f1


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "reranking"
)
DEFAULT_CONFIGS = (
    "bge_m3_fragments",
    "bge_m3_fragments_expanded",
)


def _prediction_metrics(
    rows: Sequence[dict[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    from dataset_utils import prediction_metrics
    return prediction_metrics(rows, prediction_key)


def _candidate_recall(
    rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[int] = (1, 5, 10, 20, 30),
) -> dict[str, Optional[float]]:
    labeled = [row for row in rows if clean_text(row.get("gold_label"))]
    available_k = max((len(row.get("candidates", [])) for row in rows), default=0)
    cutoffs = sorted({k for k in cutoffs if 0 < k <= available_k} | ({available_k} if available_k else set()))
    result: dict[str, Optional[float]] = {}
    for cutoff in cutoffs:
        hits = 0
        for row in labeled:
            gold = canonical_label(row["gold_label"])
            labels = {
                canonical_label(candidate["category"])
                for candidate in row["candidates"][:cutoff]
            }
            hits += int(gold in labels)
        result[str(cutoff)] = hits / len(labeled) if labeled else None
    return result


def _task_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        row["query"],
        tuple(candidate["standard_id"] for candidate in row["candidates"]),
    )


def prepare_rows(
    recall_rows: Sequence[dict[str, Any]],
    standards_by_id: dict[str, Any],
    candidate_source: str,
    candidate_k: int,
) -> list[dict[str, Any]]:
    candidate_key = f"{candidate_source}_candidates"
    rows: list[dict[str, Any]] = []
    missing_ids: set[str] = set()
    for source in recall_rows:
        candidates: list[dict[str, Any]] = []
        for candidate in source.get(candidate_key, [])[:candidate_k]:
            standard_id = clean_text(candidate.get("standard_id"))
            standard = standards_by_id.get(standard_id)
            if standard is None:
                missing_ids.add(standard_id)
                continue
            candidates.append(
                {
                    "standard_id": standard_id,
                    "category": clean_text(candidate.get("category")),
                    "scope": standard.scope,
                    "domain": standard.domain,
                }
            )
        query_views = [
            clean_text(value)
            for value in source.get("query_views", [])
            if clean_text(value)
        ]
        query = "\n".join(query_views)
        retrieval_prediction = (
            candidates[0]["category"] if candidates else ""
        )
        gold = clean_text(source.get("gold_label"))
        rows.append(
            {
                "source_row": source.get("source_row"),
                "domain": source.get("domain", ""),
                "inference_domain": source.get("inference_domain", ""),
                "label_status": source.get("label_status", ""),
                "gold_level": source.get("gold_level", ""),
                "source_id": source.get("source_id"),
                "field_name": clean_text(source.get("field_name")),
                "gold_label": gold or None,
                "query_views": query_views,
                "query": query,
                "retrieval_prediction": retrieval_prediction or None,
                "retrieval_correct": (
                    canonical_label(retrieval_prediction)
                    == canonical_label(gold)
                    if gold and retrieval_prediction
                    else None
                ),
                "candidates": candidates,
                "rerank_prediction": None,
                "rerank_standard_id": None,
                "rerank_reason": None,
                "rerank_method": None,
                "rerank_repair": None,
                "rerank_error": None,
                "rerank_correct": None,
            }
        )
    if missing_ids:
        examples = ", ".join(sorted(missing_ids)[:5])
        raise ValueError(
            f"召回结果中有 {len(missing_ids)} 个 standard_id 不在多片段知识库："
            f"{examples}"
        )
    return rows


def rerank_rows(
    rows: list[dict[str, Any]],
    reranker: OpenAICompatibleReranker,
    batch_size: int,
    failure_policy: str,
) -> int:
    unique: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for index, row in enumerate(rows):
        if row["candidates"]:
            unique.setdefault(_task_key(row), []).append(index)
    tasks = list(unique)
    failures = 0
    for start in range(0, len(tasks), batch_size):
        batch_keys = tasks[start : start + batch_size]
        representative_indexes = [unique[key][0] for key in batch_keys]
        requests = [
            (
                rows[index]["field_name"],
                rows[index]["query"],
                rows[index]["candidates"],
            )
            for index in representative_indexes
        ]
        try:
            decisions = reranker.rerank_batch(requests)
            for key, decision in zip(batch_keys, decisions):
                for index in unique[key]:
                    row = rows[index]
                    selected = next(
                        candidate
                        for candidate in row["candidates"]
                        if candidate["standard_id"] == decision.standard_id
                    )
                    prediction = selected["category"]
                    row["rerank_prediction"] = prediction
                    row["rerank_standard_id"] = decision.standard_id
                    row["rerank_reason"] = decision.reason or None
                    row["rerank_repair"] = decision.repair_note or None
                    row["rerank_method"] = (
                        "llm_cache" if decision.cached else "llm_batch_rerank"
                    )
                    if decision.repair_note:
                        row["rerank_method"] += "_repaired"
                    row["rerank_correct"] = (
                        canonical_label(prediction)
                        == canonical_label(row["gold_label"])
                        if row["gold_label"]
                        else None
                    )
        except Exception as exc:
            affected = sum(len(unique[key]) for key in batch_keys)
            failures += affected
            if failure_policy == "error":
                raise
            for key in batch_keys:
                for index in unique[key]:
                    row = rows[index]
                    row["rerank_error"] = str(exc)
                    row["rerank_prediction"] = row["retrieval_prediction"]
                    row["rerank_standard_id"] = row["candidates"][0][
                        "standard_id"
                    ]
                    row["rerank_method"] = (
                        "llm_error_fallback_retrieval_top1"
                    )
                    row["rerank_correct"] = row["retrieval_correct"]
        completed = min(start + len(batch_keys), len(tasks))
        print(
            f"LLM unique tasks {completed}/{len(tasks)}; "
            f"API requests={reranker.api_requests}; "
            f"cache hits={reranker.cache_hits}",
            flush=True,
        )
    return failures


def run_config(
    config: str,
    args: argparse.Namespace,
    standards_by_id: dict[str, Any],
    model: str,
    base_url: str,
    api_key_env: str,
) -> dict[str, Any]:
    recall_path = args.recall_dir / f"{config}.json"
    if not recall_path.exists():
        raise FileNotFoundError(f"找不到召回结果：{recall_path}")
    recall_rows = json.loads(recall_path.read_text(encoding="utf-8"))
    rows = prepare_rows(
        recall_rows,
        standards_by_id,
        args.candidate_source,
        args.candidate_k,
    )
    cache_path = args.output_dir / f"{config}.cache.jsonl"
    reranker = OpenAICompatibleReranker(
        provider=args.llm_provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        cache=JsonlRerankCache(cache_path),
        timeout=args.llm_timeout,
        max_retries=args.llm_max_retries,
        request_delay=args.llm_request_delay,
        rate_limit_delay=args.llm_rate_limit_delay,
        candidate_text_chars=args.llm_candidate_text_chars,
    )
    started = time.perf_counter()
    failures = rerank_rows(
        rows,
        reranker,
        args.llm_batch_size,
        args.llm_failure_policy,
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "config": config,
        "source_recall": str(recall_path.resolve()),
        "rows": len(rows),
        "unique_tasks": len({_task_key(row) for row in rows}),
        "candidate_source": args.candidate_source,
        "candidate_k": args.candidate_k,
        "candidate_recall_at": _candidate_recall(rows),
        "retrieval": _prediction_metrics(rows, "retrieval_prediction"),
        "rerank": _prediction_metrics(rows, "rerank_prediction"),
        "llm": {
            "provider": args.llm_provider,
            "model": model,
            "base_url": base_url,
            "batch_size": args.llm_batch_size,
            "api_requests": reranker.api_requests,
            "successful_batches": reranker.successful_batches,
            "cache_hits": reranker.cache_hits,
            "rate_limit_retries": reranker.rate_limit_retries,
            "failed_batches": reranker.failed_batches,
            "repaired_items": reranker.repaired_items,
            "fallback_rows": failures,
        },
        "inference_seconds": round(elapsed, 6),
        "seconds_per_row": round(elapsed / len(rows), 6) if rows else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{config}.json"
    metrics_path = args.output_dir / f"{config}.metrics.json"
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reranker.cache.close()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def summarize(output_dir: Path) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for config in DEFAULT_CONFIGS:
        path = output_dir / f"{config}.metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        summary.append(
            {
                "config": config,
                "candidate_recall@30": metrics["candidate_recall_at"]["30"],
                "retrieval_accuracy": metrics["retrieval"]["accuracy"],
                "retrieval_macro_f1": metrics["retrieval"]["macro_f1"],
                "rerank_accuracy": metrics["rerank"]["accuracy"],
                "rerank_macro_f1": metrics["rerank"]["macro_f1"],
                "api_requests": metrics["llm"]["api_requests"],
                "cache_hits": metrics["llm"]["cache_hits"],
                "fallback_rows": metrics["llm"]["fallback_rows"],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对两组 BGE-M3 多片段召回结果进行批量 LLM 重排"
    )
    parser.add_argument("--configs", nargs="+", choices=DEFAULT_CONFIGS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--recall-dir", type=Path, default=DEFAULT_RECALL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "finance" / "corpus.json",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("vector", "hybrid"),
        default="vector",
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument(
        "--llm-provider",
        choices=("modelscope", "openrouter", "custom"),
        default="modelscope",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--llm-candidate-text-chars", type=int, default=100)
    parser.add_argument("--llm-request-delay", type=float, default=3.0)
    parser.add_argument("--llm-rate-limit-delay", type=float, default=60.0)
    parser.add_argument(
        "--llm-failure-policy",
        choices=("error", "fallback"),
        default="error",
    )
    args = parser.parse_args(argv)
    if not args.summarize and not args.configs:
        args.configs = list(DEFAULT_CONFIGS)
    if args.candidate_k < 1 or args.candidate_k > 50:
        parser.error("--candidate-k 必须在 1 到 50 之间")
    if args.llm_batch_size < 1:
        parser.error("--llm-batch-size 必须大于 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    model, base_url, api_key_env = _provider_defaults(
        args.llm_provider,
        args.llm_model,
        args.llm_base_url,
        args.llm_api_key_env,
    )
    standards = expand_standards_with_fragments(
        load_corpus(args.corpus.expanduser().resolve())
    )
    standards_by_id = {item.standard_id: item for item in standards}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        metrics_path = args.output_dir / f"{config}.metrics.json"
        if metrics_path.exists():
            print(f"完整结果已存在，跳过：{config}", flush=True)
            continue
        print(f"运行 LLM 重排：{config}", flush=True)
        run_config(
            config,
            args,
            standards_by_id,
            model,
            base_url,
            api_key_env,
        )
    print(json.dumps(summarize(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if os.getenv("XRAG_DEBUG") == "1":
            raise
        raise SystemExit(1)
