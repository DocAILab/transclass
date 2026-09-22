#!/usr/bin/env python3
"""Dynamic abbreviation expansion + BGE-M3 fragments + LLM reranking."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

from abbreviation_cache import DEFAULT_ABBREVIATION_CACHE, open_shared_store

from abbreviation_generator import (
    AbbreviationGenerator,
    build_offline_dictionary,
    AbbreviationStore,
    FieldExpansion,
    compose_known_expansion,
    expansion_query_views,
    normalize_field,
    rule_parts,
    split_with_known,
)
from reranking import (
    _candidate_recall,
    _prediction_metrics,
    prepare_rows,
    rerank_rows,
)
from retrieval import (
    LOCAL_BGE_M3,
    MULTILINGUAL_EMBEDDING,
    _batched_vector_category_rankings,
    _compact_candidates,
    _metrics,
    _rrf_categories,
    expand_standards_with_fragments,
)
from llm_client import (
    JsonlRerankCache,
    MODELSCOPE_BASE_URL,
    MODELSCOPE_DEFAULT_MODEL,
    OpenAICompatibleReranker,
    load_corpus,
    read_finance_xlsx,
    read_test_json,
)
from vector_index import DEFAULT_CACHE, XRAGClassifier, canonical_label


from domain_config import DOMAINS, inference_domain, resolve_domain, validate_known_domain

from dataset_utils import (
    candidate_counts, dataset_audit, grouped_metrics, run_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "experiments"
)


def _expansion_task(
    field_name: str,
    store: AbbreviationStore,
) -> dict[str, Any]:
    parts = rule_parts(field_name)
    suggested = split_with_known(field_name, store.known_tokens())
    known: list[dict[str, Any]] = []
    for token in suggested:
        meanings = store.meanings_for_token(token)
        if meanings:
            known.append(
                {
                    "token": token,
                    "meanings": [
                        {
                            "english": value.english,
                            "chinese": value.chinese,
                            "confidence": value.confidence,
                        }
                        for value in meanings
                    ],
                }
            )
    return {
        "field_name": field_name,
        "normalized_field": normalize_field(field_name),
        "rule_parts": parts,
        "known_token_hints": known,
    }


def build_dynamic_dictionary(
    field_names: Sequence[str],
    store: AbbreviationStore,
    generator: AbbreviationGenerator,
    batch_size: int,
) -> tuple[dict[str, FieldExpansion], dict[str, Any]]:
    unique_fields = list(dict.fromkeys(field_names))
    expansions: dict[str, FieldExpansion] = {}
    source_counts: Counter[str] = Counter()
    full_cache_hits = 0
    composed_hits = 0
    llm_generated = 0
    for start in range(0, len(unique_fields), batch_size):
        window = unique_fields[start : start + batch_size]
        pending_fields: list[str] = []
        pending_tasks: list[dict[str, Any]] = []
        for field_name in window:
            cached = store.get_field(field_name, generator.model)
            if cached is not None:
                expansions[field_name] = cached
                source_counts[cached.source] += 1
                full_cache_hits += 1
                continue
            tokens = split_with_known(field_name, store.known_tokens())
            composed = compose_known_expansion(
                field_name,
                tokens,
                store,
                generator.model,
            )
            if composed is not None:
                store.save_field(composed)
                expansions[field_name] = composed
                source_counts[composed.source] += 1
                composed_hits += 1
                continue
            pending_fields.append(field_name)
            pending_tasks.append(_expansion_task(field_name, store))
        if pending_tasks:
            generated = generator.generate_batch(pending_tasks, on_success=store.save_field)
            for field_name, expansion in zip(pending_fields, generated):
                store.save_field(expansion)
                expansions[field_name] = expansion
                source_counts[expansion.source] += 1
                llm_generated += 1
        completed = min(start + len(window), len(unique_fields))
        print(
            f"dynamic dictionary {completed}/{len(unique_fields)}; "
            f"API requests={generator.api_requests}; "
            f"full cache={full_cache_hits}; composed={composed_hits}",
            flush=True,
        )
    return expansions, {
        "unique_fields": len(unique_fields),
        "source_counts": dict(source_counts),
        "full_cache_hits": full_cache_hits,
        "composed_without_api": composed_hits,
        "llm_generated_fields": llm_generated,
        "api_requests": generator.api_requests,
        "successful_batches": generator.successful_batches,
        "rate_limit_retries": generator.rate_limit_retries,
        "failed_batches": generator.failed_batches,
        "store": store.stats(),
    }


def run_retrieval(
    args: argparse.Namespace,
    examples: Sequence[Any],
    expansions: dict[str, FieldExpansion],
    standards: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from domain_config import require_local_model
    embedding_model = str(require_local_model(args.model_dir))
    init_started = time.perf_counter()
    classifier = XRAGClassifier(
        standards=standards,
        embedding_model=embedding_model,
        retriever_type="vector",
        top_k=len(standards),
        cache_dir=args.cache_dir / args.domain,
        device=args.device,
        recall_k=len(standards),
        rrf_k=args.rrf_k,
    )
    classifier.group_by_domain = args.domain == "generic"
    index_seconds = time.perf_counter() - init_started
    views_by_example = [
        expansion_query_views(expansions[example.field_name], args.domain)
        for example in examples
    ]
    unique_queries = list(
        dict.fromkeys(query for views in views_by_example for query in views)
    )
    retrieval_started = time.perf_counter()
    vector_by_query = _batched_vector_category_rankings(
        classifier,
        unique_queries,
    )
    ranking_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for index, (example, views) in enumerate(
        zip(examples, views_by_example),
        start=1,
    ):
        key = tuple(views)
        if key not in ranking_cache:
            routes = [
                (f"vector_q{view_index}", vector_by_query[query])
                for view_index, query in enumerate(views)
            ]
            ranking_cache[key] = _rrf_categories(routes, args.rrf_k)
        expansion = expansions[example.field_name]
        rows.append(
            {
                "source_row": example.source_row,
                "source_id": example.source_id,
                "field_name": example.field_name,
                "gold_label": example.gold_label,
                "gold_level": example.gold_level,
                "domain": example.domain,
                "inference_domain": args.domain,
                "label_status": example.label_status,
                "expansion": expansion.to_dict(),
                "query_views": views,
                "vector_candidates": _compact_candidates(
                    ranking_cache[key],
                    args.output_k,
                ),
            }
        )
        if index % 50 == 0 or index == len(examples):
            print(
                f"dynamic retrieval {index}/{len(examples)}; "
                f"unique fields={len(ranking_cache)}",
                flush=True,
            )
    retrieval_seconds = time.perf_counter() - retrieval_started
    metrics = {
        "method": "bge_m3_fragments_dynamic_abbreviation",
        "embedding": MULTILINGUAL_EMBEDDING,
        "embedding_source": embedding_model,
        "fragments": True,
        "abbreviation_generator": True,
        "rows": len(rows),
        "original_documents": args.original_documents,
        "requested_output_k": args.output_k,
        "actual_candidate_counts": candidate_counts(rows, "vector_candidates"),
        "indexed_vectors": len(standards),
        "unique_categories": len(
            {canonical_label(item.category) for item in standards}
        ),
        "unique_query_view_sets": len(ranking_cache),
        "unique_query_views": len(unique_queries),
        "index_seconds": round(index_seconds, 6),
        "retrieval_seconds": round(retrieval_seconds, 6),
        "seconds_per_row": round(retrieval_seconds / len(rows), 6),
        "vector": _metrics(rows, "vector_candidates"),
    }
    report_rows = [{**row, "prediction": row["vector_candidates"][0]["category"]
                    if row["vector_candidates"] else None} for row in rows]
    metrics["classification_report"] = _prediction_metrics(report_rows, "prediction")
    metrics["groups"] = grouped_metrics(report_rows, "prediction")
    return rows, metrics


def run_rerank(
    args: argparse.Namespace,
    retrieval_rows: Sequence[dict[str, Any]],
    standards_by_id: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = prepare_rows(
        retrieval_rows,
        standards_by_id,
        "vector",
        args.candidate_k,
    )
    reranker = OpenAICompatibleReranker(
        provider="modelscope",
        domain=args.domain,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        cache=JsonlRerankCache(args.output_dir / "rerank.cache.jsonl"),
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
        "error",
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "method": "bge_m3_fragments_dynamic_abbreviation_llm_rerank",
        "rows": len(rows),
        "unique_tasks": len(
            {
                (
                    row["query"],
                    tuple(
                        candidate["standard_id"]
                        for candidate in row["candidates"]
                    ),
                )
                for row in rows
            }
        ),
        "candidate_k": args.candidate_k,
        "actual_candidate_counts": candidate_counts(rows, "candidates"),
        "groups": grouped_metrics(rows, "rerank_prediction"),
        "candidate_recall_at": _candidate_recall(rows),
        "retrieval": _prediction_metrics(rows, "retrieval_prediction"),
        "rerank": _prediction_metrics(rows, "rerank_prediction"),
        "llm": {
            "provider": "modelscope",
            "model": args.llm_model,
            "base_url": args.llm_base_url,
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
        "seconds_per_row": round(elapsed / len(rows), 6),
    }
    reranker.cache.close()
    return rows, metrics


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def build_summary(
    dictionary_metrics: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    rerank_metrics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "dictionary": dictionary_metrics,
        "retrieval": {
            "top1_accuracy": retrieval_metrics["vector"]["top1_accuracy"],
            "macro_f1": retrieval_metrics["vector"]["macro_f1"],
            "classification_report": retrieval_metrics.get("classification_report"),
            "groups": retrieval_metrics.get("groups"),
            "actual_candidate_counts": retrieval_metrics.get("actual_candidate_counts"),
            "candidate_recall_at": retrieval_metrics["vector"][
                "candidate_recall_at"
            ],
        },
        "rerank": None,
    }
    if rerank_metrics is not None:
        result["rerank"] = rerank_metrics["rerank"]
        result["rerank_llm"] = rerank_metrics["llm"]
        result["rerank_groups"] = rerank_metrics["groups"]
        result["rerank_candidate_counts"] = rerank_metrics["actual_candidate_counts"]
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="动态缩写词典 + BGE-M3多片段 + LLM重排实验"
    )
    parser.add_argument("--use-domain", action=argparse.BooleanOptionalAction, default=True,
                        help="默认读取每条记录的 domain；--no-use-domain 禁用并使用通用流程")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    for name in ("finance", "education", "vehicle"):
        parser.add_argument(f"--{name}-corpus", type=Path, help="覆盖该领域的语料路径")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-json", type=Path)
    parser.add_argument("--dataset-id", help="仅用于结果目录隔离，不用于推理领域提示")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--build-index", action="store_true", help="只构建知识库索引，不调用线上模型")
    parser.add_argument("--model-dir", type=Path, default=LOCAL_BGE_M3)
    inputs.add_argument("--input-xlsx", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--abbreviation-cache-dir", type=Path, default=DEFAULT_ABBREVIATION_CACHE)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--header-row", type=int, default=2)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output-k", type=int, default=30)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--dictionary-batch-size", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=8)
    parser.add_argument("--llm-model", default=MODELSCOPE_DEFAULT_MODEL)
    parser.add_argument("--llm-base-url", default=MODELSCOPE_BASE_URL)
    parser.add_argument("--llm-api-key-env", default="MODELSCOPE_API_TOKEN")
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-request-delay", type=float, default=3.0)
    parser.add_argument("--llm-rate-limit-delay", type=float, default=60.0)
    parser.add_argument("--llm-candidate-text-chars", type=int, default=100)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
                        help="完全禁用 LLM：使用种子词与规则拆分，并跳过重排")
    args = parser.parse_args(argv)
    if args.no_llm:
        args.skip_rerank = True
    if args.input_json is None and args.input_xlsx is None:
        args.input_json = DOMAINS["finance"].data_paths(args.data_root)[0]
    if args.dictionary_batch_size < 1 or args.llm_batch_size < 1:
        parser.error("批次大小必须大于0")
    if args.candidate_k < 1 or args.output_k < 1 or args.rrf_k < 1:
        parser.error("候选数量和 RRF 参数必须大于0")
    if args.dataset_id and (Path(args.dataset_id).name != args.dataset_id or args.dataset_id in (".", "..")):
        parser.error("dataset-id 必须是单个目录名称")
    if args.output_k < args.candidate_k:
        parser.error("--output-k不能小于--candidate-k")
    return args


def run_route(args, examples, original_standards, audit, manifest):
    """Run one homogeneous inference route, with an isolated dictionary/cache."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "dataset.audit.json", audit)
    args.original_documents = len(original_standards)
    standards = expand_standards_with_fragments(original_standards)
    standards_by_id = {item.standard_id: item for item in standards}
    if args.no_llm:
        store = AbbreviationStore(args.output_dir / "abbreviations.sqlite3", domain=args.domain, model="local_rules")
        migration = None
    else:
        store, migration = open_shared_store(
            args.abbreviation_cache_dir, args.domain, args.llm_model, args.legacy_cache_roots,
        )
        write_json(args.output_dir / "abbreviation.cache.json", {
            "path": str(store.path.resolve()), "scope": {"domain": args.domain, "model": args.llm_model,
            "prompt_version": store.prompt_version}, "migration": migration,
        })
        print(f"共享词典：{store.path}；已有字段={store.stats()['fields']}；本次迁移={migration['imported']}", flush=True)
    try:
        if args.no_llm:
            expansions, dictionary_metrics = build_offline_dictionary(
                [example.field_name for example in examples], store,
            )
        else:
            generator = AbbreviationGenerator(
                diagnostics_path=args.output_dir / "abbreviation.responses.jsonl",
                domain=args.domain,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout=args.llm_timeout,
                max_retries=args.llm_max_retries,
                request_delay=args.llm_request_delay,
                rate_limit_delay=args.llm_rate_limit_delay,
            )
            expansions, dictionary_metrics = build_dynamic_dictionary(
                [example.field_name for example in examples],
                store,
                generator,
                args.dictionary_batch_size,
            )
        dictionary_metrics["cache_path"] = str(store.path.resolve())
        write_json(
            args.output_dir / "field_expansions.json",
            [expansions[field].to_dict() for field in expansions],
        )
        write_json(args.output_dir / "dictionary.metrics.json", dictionary_metrics)
    finally:
        store.close()

    retrieval_path = args.output_dir / "retrieval.json"
    retrieval_metrics_path = args.output_dir / "retrieval.metrics.json"
    if retrieval_path.exists() and retrieval_metrics_path.exists():
        print("纯召回结果已存在，跳过重复计算。", flush=True)
        retrieval_rows = json.loads(retrieval_path.read_text(encoding="utf-8"))
        retrieval_metrics = json.loads(
            retrieval_metrics_path.read_text(encoding="utf-8")
        )
    else:
        retrieval_rows, retrieval_metrics = run_retrieval(
            args,
            examples,
            expansions,
            standards,
        )
        write_json(retrieval_path, retrieval_rows)
        write_json(retrieval_metrics_path, retrieval_metrics)

    rerank_metrics: Optional[dict[str, Any]] = None
    if not args.skip_rerank:
        rerank_path = args.output_dir / "rerank.json"
        rerank_metrics_path = args.output_dir / "rerank.metrics.json"
        if rerank_path.exists() and rerank_metrics_path.exists():
            print("LLM重排结果已存在，跳过重复计算。", flush=True)
            rerank_metrics = json.loads(
                rerank_metrics_path.read_text(encoding="utf-8")
            )
        else:
            rerank_rows_value, rerank_metrics = run_rerank(
                args,
                retrieval_rows,
                standards_by_id,
            )
            write_json(rerank_path, rerank_rows_value)
            write_json(rerank_metrics_path, rerank_metrics)
    summary = build_summary(
        dictionary_metrics,
        retrieval_metrics,
        rerank_metrics,
    )
    summary["dataset_id"] = args.dataset_id
    summary["dataset"] = audit
    summary["manifest"] = manifest
    summary["output_dir"] = str(args.output_dir)
    write_json(args.output_dir / "summary.json", summary)
    return summary


def reproduce_results(argv):
    """Recalculate explicitly selected saved predictions without model calls."""
    from dataset_utils import prediction_metrics
    import math
    parser = argparse.ArgumentParser(description="复核指定实验目录的逐条预测")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    folder = args.result_dir.expanduser().resolve()
    baseline = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / "retrieval.json").read_text(encoding="utf-8"))
    result = {"mode": "saved_prediction_replay", "real_api_calls": 0,
              "source_directory": str(folder), "retrieval": _metrics(rows, "vector_candidates")}
    if len(rows) != baseline["dataset"]["rows"]:
        raise ValueError("逐条预测数量与汇总不一致")
    stages = [("retrieval", ("top1_accuracy", "macro_f1"))]
    if baseline.get("rerank") is not None:
        rerank = json.loads((folder / "rerank.json").read_text(encoding="utf-8"))
        if len(rerank) != len(rows):
            raise ValueError("重排预测数量不一致")
        result["rerank"] = prediction_metrics(rerank, "rerank_prediction")
        stages.append(("rerank", ("accuracy", "macro_f1")))
    for stage, keys in stages:
        for key in keys:
            if not math.isclose(result[stage][key], baseline[stage][key], abs_tol=1e-12):
                raise ValueError(f"{stage}.{key} 与保存的汇总不一致")
    if result["retrieval"]["candidate_recall_at"] != baseline["retrieval"]["candidate_recall_at"]:
        raise ValueError("Recall@K 与保存的汇总不一致")
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--reproduce":
        return reproduce_results(argv[1:])
    args = parse_args(argv)
    input_path = (args.input_json or args.input_xlsx).expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"测试集不存在：{input_path}；先执行 bash run_rag.sh --download-data")
    examples = (read_test_json(input_path, use_domain=args.use_domain) if args.input_json
                else read_finance_xlsx(input_path, sheet_name=args.sheet, header_row=args.header_row,
                                       input_column="D", label_column="I", level_column="J"))
    if not examples:
        raise ValueError("测试集为空")
    groups = {}
    for example in examples:
        route = inference_domain(example.domain, args.use_domain)
        groups.setdefault(route, []).append(example)
    required_domains = set(DOMAINS) if "generic" in groups else set(groups)
    corpora = {}
    paths = []
    for domain in sorted(required_domains):
        config = DOMAINS[domain]
        option = {"finance": "finance_corpus", "dcg_education": "education_corpus",
                  "dcg_vehicle": "vehicle_corpus"}[domain]
        path = (getattr(args, option) or config.data_paths(args.data_root)[1]).expanduser().resolve()
        validate_known_domain([], json.loads(path.read_text(encoding="utf-8")), domain)
        # Namespace source IDs before combining corpora; identical IDs in different domains are legal.
        corpora[domain] = [replace(item, standard_id=f"{domain}:{item.standard_id}")
                           for item in load_corpus(path, domain=domain)]
        paths.append(path)
    all_standards = [record for domain in sorted(corpora) for record in corpora[domain]]
    audit = dataset_audit(examples, all_standards)
    audit["domain_policy"] = "record_domain_or_generic" if args.use_domain else "generic_only"
    audit["use_domain"] = args.use_domain
    audit["llm_enabled"] = not args.no_llm
    audit["query_expansion"] = "offline_seed_and_rules" if args.no_llm else "dynamic_llm"
    route_audits = {}
    for route, items in groups.items():
        standards = all_standards if route == "generic" else corpora[route]
        report = dataset_audit(items, standards)
        category_count = len({(item.domain, canonical_label(item.category)) for item in standards})
        report.update(inference_domain=route, domain_name=resolve_domain(route).chinese,
                      candidate_categories=category_count,
                      effective_candidate_k=min(args.candidate_k, category_count),
                      candidate_set_can_cover_all_categories=args.candidate_k >= category_count,
                      corpus_domains=sorted({item.domain for item in standards}))
        route_audits[route] = report
    audit["routes"] = route_audits
    if not args.validate_only:
        from domain_config import require_local_model
        require_local_model(args.model_dir)
    if args.build_index and not args.validate_only:
        for route in groups:
            standards = all_standards if route == "generic" else corpora[route]
            classifier = XRAGClassifier(
                standards=expand_standards_with_fragments(standards),
                embedding_model=str(args.model_dir.expanduser().resolve()),
                retriever_type="vector", top_k=len(standards),
                cache_dir=args.cache_dir / route, device=args.device,
                recall_k=len(standards), rrf_k=args.rrf_k,
            )
            print(f"索引已就绪：{route}；目录：{args.cache_dir / route}；线上模型调用 0 次")
            del classifier
        return 0
    manifest = run_manifest(args, input_path, paths)
    args.legacy_cache_roots = []  # 旧词典仅通过显式迁移命令导入。
    args.dataset_id = args.dataset_id or input_path.stem
    output_dir = args.output_dir / args.dataset_id / manifest["fingerprint"]
    if args.validate_only:
        print(json.dumps({"dataset": audit, "manifest": manifest, "output_dir": str(output_dir)},
                         ensure_ascii=False, indent=2))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "dataset.audit.json", audit)
    route_summaries = {}
    for route, items in groups.items():
        route_args = argparse.Namespace(**vars(args))
        route_args.domain = route
        route_args.output_dir = output_dir / "routes" / route
        route_summaries[route] = run_route(
            route_args, items, all_standards if route == "generic" else corpora[route],
            route_audits[route], manifest,
        )
    retrieval_rows = []
    rerank_rows_value = []
    for route in groups:
        route_dir = output_dir / "routes" / route
        retrieval_rows.extend(json.loads((route_dir / "retrieval.json").read_text(encoding="utf-8")))
        if not args.skip_rerank:
            rerank_rows_value.extend(json.loads((route_dir / "rerank.json").read_text(encoding="utf-8")))
    retrieval_rows.sort(key=lambda row: row["source_row"])
    rerank_rows_value.sort(key=lambda row: row["source_row"])
    write_json(output_dir / "retrieval.json", retrieval_rows)
    report_rows = [{**row, "prediction": row["vector_candidates"][0]["category"]
                    if row["vector_candidates"] else None} for row in retrieval_rows]
    summary = {
        "dataset_id": args.dataset_id, "dataset": audit, "manifest": manifest,
        "output_dir": str(output_dir), "routes": route_summaries,
        "retrieval": {**_metrics(retrieval_rows, "vector_candidates"),
                      "classification_report": _prediction_metrics(report_rows, "prediction"),
                      "groups": grouped_metrics(report_rows, "prediction"),
                      "actual_candidate_counts": candidate_counts(retrieval_rows, "vector_candidates")},
        "rerank": None,
    }
    if not args.skip_rerank:
        write_json(output_dir / "rerank.json", rerank_rows_value)
        summary["rerank"] = _prediction_metrics(rerank_rows_value, "rerank_prediction")
        summary["rerank_groups"] = grouped_metrics(rerank_rows_value, "rerank_prediction")
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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
