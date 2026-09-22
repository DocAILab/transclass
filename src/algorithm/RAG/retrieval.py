#!/usr/bin/env python3
"""Pure-retrieval ablation for the finance corpus.

The test workbook contributes only column D to retrieval. Column I is read
after retrieval as the gold fourth-level category for offline evaluation.
No LLM API is used by this module.
"""

from __future__ import annotations

import os

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from llm_client import (
    FinanceExample,
    build_finance_query,
    load_corpus,
    read_finance_xlsx,
)
from vector_index import (
    DEFAULT_CACHE,
    StandardRecord,
    XRAGClassifier,
    canonical_label,
    clean_text,
    macro_f1,
    stable_id,
)


PROJECT_ROOT = Path(os.environ.get("RAG_WORKSPACE_ROOT", "/Users/andiandian/Desktop/trandatacls")).expanduser().resolve() / "transclass_repo"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "retrieval"
)
CURRENT_EMBEDDING = "BAAI/bge-small-zh-v1.5"
MULTILINGUAL_EMBEDDING = "BAAI/bge-m3"
LOCAL_BGE_M3 = PROJECT_ROOT / "src/algorithm/RAG/models/BAAI_bge-m3"
CONFIGS = {
    "bge_zh_single": {
        "embedding": CURRENT_EMBEDDING,
        "fragments": False,
        "query_expansion": False,
    },
    "bge_m3_single": {
        "embedding": MULTILINGUAL_EMBEDDING,
        "fragments": False,
        "query_expansion": False,
    },
    "bge_zh_fragments": {
        "embedding": CURRENT_EMBEDDING,
        "fragments": True,
        "query_expansion": False,
    },
    "bge_m3_fragments": {
        "embedding": MULTILINGUAL_EMBEDDING,
        "fragments": True,
        "query_expansion": False,
    },
    "bge_m3_fragments_expanded": {
        "embedding": MULTILINGUAL_EMBEDDING,
        "fragments": True,
        "query_expansion": True,
    },
}


# Longest-match dictionary for common database-field abbreviations. Unknown
# spans are preserved, so expansion cannot delete information from column D.
ABBREVIATIONS: dict[str, tuple[str, str]] = {
    "ACCOUNT": ("account", "账户"),
    "ACCT": ("account", "账户"),
    "ACC": ("account", "账户"),
    "AVAILABLE": ("available", "可用"),
    "AVAIL": ("available", "可用"),
    "BALANCE": ("balance", "余额"),
    "BAL": ("balance", "余额"),
    "BANK": ("bank", "银行"),
    "BNK": ("bank", "银行"),
    "BRANCH": ("branch", "分支机构"),
    "CERTIFICATE": ("certificate", "证件"),
    "CERT": ("certificate", "证件"),
    "CHANNEL": ("channel", "渠道"),
    "CHNL": ("channel", "渠道"),
    "CODE": ("code", "代码"),
    "CONTRACT": ("contract", "合同"),
    "CURRENCY": ("currency", "币种"),
    "CURR": ("currency", "币种"),
    "CUSTOMER": ("customer", "客户"),
    "CUST": ("customer", "客户"),
    "DATE": ("date", "日期"),
    "DEPARTMENT": ("department", "部门"),
    "DEPT": ("department", "部门"),
    "DESCRIPTION": ("description", "描述"),
    "DESC": ("description", "描述"),
    "END": ("end", "结束"),
    "EXPIRE": ("expire", "到期"),
    "EXP": ("expire", "到期"),
    "FILE": ("file", "文件"),
    "FILENAME": ("file name", "文件名"),
    "FIRM": ("institution", "机构"),
    "FLAG": ("flag", "标志"),
    "FLOW": ("flow", "流程"),
    "FREEZE": ("freeze", "冻结"),
    "IDENTIFIER": ("identifier", "标识"),
    "INSTITUTION": ("institution", "机构"),
    "INST": ("institution", "机构"),
    "LOAN": ("loan", "贷款"),
    "MOBILE": ("mobile phone", "手机"),
    "MOB": ("mobile phone", "手机"),
    "MONEY": ("money", "资金"),
    "NAME": ("name", "名称"),
    "NUMBER": ("number", "编号"),
    "NUM": ("number", "编号"),
    "ORGANIZATION": ("organization", "组织机构"),
    "ORG": ("organization", "组织机构"),
    "PATH": ("path", "路径"),
    "PAYMENT": ("payment", "支付"),
    "PAY": ("payment", "支付"),
    "PHONE": ("phone", "电话"),
    "PRODUCT": ("product", "产品"),
    "PROD": ("product", "产品"),
    "RATE": ("rate", "利率"),
    "RECEIVE": ("receive", "接收"),
    "RECEIVER": ("receiver", "接收方"),
    "RECV": ("receiver", "接收方"),
    "RCV": ("receiver", "接收方"),
    "SEND": ("sender", "发送方"),
    "SENDER": ("sender", "发送方"),
    "SERIAL": ("serial number", "流水号"),
    "SIGN": ("sign", "签署"),
    "START": ("start", "开始"),
    "STATE": ("state", "状态"),
    "STATUS": ("status", "状态"),
    "STAT": ("status", "状态"),
    "TELEPHONE": ("telephone", "电话"),
    "TIME": ("time", "时间"),
    "TRANSACTION": ("transaction", "交易"),
    "TRANS": ("transaction", "交易"),
    "TYPE": ("type", "类型"),
    "USEABLE": ("usable", "可用"),
    "USER": ("user", "用户"),
    "AMT": ("amount", "金额"),
    "ADDR": ("address", "地址"),
    "AUTO": ("automatic", "自动"),
    "CRT": ("create", "创建"),
    "CREATED": ("created", "创建"),
    "DT": ("date", "日期"),
    "ID": ("identifier", "标识"),
    "IP": ("IP address", "IP地址"),
    "NO": ("number", "编号"),
    "SEQ": ("sequence", "序列号"),
    "SQ": ("sequence", "序列号"),
    "TEL": ("telephone", "电话"),
    "TM": ("time", "时间"),
    "TP": ("type", "类型"),
    "TXN": ("transaction", "交易"),
    "CD": ("code", "代码"),
    "NM": ("name", "名称"),
}


def _split_identifier_parts(field_name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", field_name)
    return [
        value.upper()
        for value in re.split(r"[^A-Za-z0-9]+", spaced)
        if value
    ]


def _longest_abbreviation_split(value: str) -> list[str]:
    """Greedily preserve unknown spans and split known abbreviations."""
    if not value or not value.isascii() or not value.isalnum():
        return [value] if value else []
    if value.isdigit():
        return [value]
    keys = sorted(ABBREVIATIONS, key=len, reverse=True)
    result: list[str] = []
    unknown = ""
    index = 0
    while index < len(value):
        matched = next(
            (key for key in keys if value.startswith(key, index)),
            None,
        )
        if matched:
            if unknown:
                result.append(unknown)
                unknown = ""
            result.append(matched)
            index += len(matched)
        else:
            unknown += value[index]
            index += 1
    if unknown:
        result.append(unknown)
    return result


def expand_finance_field(field_name: str) -> dict[str, str]:
    tokens: list[str] = []
    for part in _split_identifier_parts(clean_text(field_name)):
        tokens.extend(_longest_abbreviation_split(part))
    english = [ABBREVIATIONS.get(token, (token.lower(), token))[0] for token in tokens]
    chinese = [ABBREVIATIONS.get(token, (token, token))[1] for token in tokens]
    return {
        "tokens": " ".join(tokens),
        "english": " ".join(english),
        "chinese": " ".join(chinese),
    }


def finance_query_views(
    field_name: str,
    expanded: bool,
) -> list[str]:
    base = build_finance_query(field_name)
    if not expanded:
        return [base]
    expansion = expand_finance_field(field_name)
    candidates = [
        base,
        (
            f"financial database field {field_name}; "
            f"meaning: {expansion['english']}"
        ),
        (
            f"金融数据库字段 {field_name}；"
            f"中文含义：{expansion['chinese']}"
        ),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = clean_text(candidate)
        if normalized and normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    return result


def semantic_fragments(text: str, limit: int = 3) -> list[str]:
    """Extract short semantic facets while retaining the full definition."""
    source = clean_text(text)
    if not source:
        return []
    candidates: list[str] = []
    sentences = re.split(r"[。；;]", source)
    for sentence in sentences:
        sentence = clean_text(sentence)
        if not sentence:
            continue
        enumeration = re.split(
            r"(?:例如|比如|如|包括但不限于|包括|包含|主要有|主要包括)",
            sentence,
            maxsplit=1,
        )
        tail = enumeration[-1]
        if len(enumeration) > 1:
            candidates.extend(re.split(r"[、，,]", tail))
        else:
            candidates.append(sentence)

    result: list[str] = []
    seen: set[str] = {canonical_label(source.strip("。；; "))}
    for candidate in candidates:
        value = clean_text(candidate).strip("（）()[]【】等以及和及：: ")
        key = canonical_label(value)
        if len(value) < 2 or key in seen:
            continue
        if len(value) > 100:
            continue
        result.append(value)
        seen.add(key)
        if len(result) >= limit:
            break
    return result


def expand_standards_with_fragments(
    standards: Sequence[StandardRecord],
) -> list[StandardRecord]:
    result = list(standards)
    for standard in standards:
        for fragment_index, fragment in enumerate(
            semantic_fragments(standard.scope),
            start=1,
        ):
            source_file = f"{standard.source_file}#{standard.standard_id}#facet-{fragment_index}"
            result.append(
                StandardRecord(
                    standard_id=stable_id(
                        standard.domain,
                        source_file,
                        fragment,
                        standard.category,
                        standard.level,
                    ),
                    domain=standard.domain,
                    source_file=source_file,
                    scope=fragment,
                    category=standard.category,
                    level=standard.level,
                    reference=standard.reference,
                    category_path=standard.category_path,
                )
            )
    return result


def _category_ranking(
    classifier: XRAGClassifier,
    retriever: Any,
    query: str,
    skip_nonpositive: bool = False,
) -> list[dict[str, Any]]:
    """Collapse multiple document vectors to each category's best rank."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document_rank, (standard_id, score) in enumerate(
        classifier._retrieve_one(retriever, query),
        start=1,
    ):
        if skip_nonpositive and score <= 0:
            continue
        standard = classifier.standards[standard_id]
        key = canonical_label(standard.category)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "category_key": key,
                "category": standard.category,
                "standard_id": standard_id,
                "document_rank": document_rank,
                "raw_score": score,
            }
        )
    return result


def _batched_vector_category_rankings(
    classifier: XRAGClassifier,
    queries: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Embed all unique queries in batches and rank the persisted vectors."""
    vector_data = classifier.index.vector_store._data
    node_ids: list[str] = []
    standard_ids: list[str] = []
    embeddings: list[list[float]] = []
    for node_id, embedding in vector_data.embedding_dict.items():
        standard_id = vector_data.text_id_to_ref_doc_id.get(node_id, "")
        if standard_id not in classifier.standards:
            continue
        node_ids.append(node_id)
        standard_ids.append(standard_id)
        embeddings.append(embedding)
    del node_ids
    document_matrix = np.asarray(embeddings, dtype=np.float32)
    document_norms = np.linalg.norm(document_matrix, axis=1, keepdims=True)
    document_matrix = document_matrix / np.maximum(document_norms, 1e-12)

    embedder = classifier.vector._embed_model
    query_matrix = np.asarray(
        embedder._embed(list(queries), prompt_name="query"),
        dtype=np.float32,
    )
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    query_matrix = query_matrix / np.maximum(query_norms, 1e-12)
    similarities = query_matrix @ document_matrix.T

    result: dict[str, list[dict[str, Any]]] = {}
    for query_index, query in enumerate(queries):
        ranking: list[dict[str, Any]] = []
        seen: set[str] = set()
        order = np.argsort(-similarities[query_index])
        for document_rank, matrix_index in enumerate(order, start=1):
            standard_id = standard_ids[int(matrix_index)]
            standard = classifier.standards[standard_id]
            key = canonical_label(standard.category)
            if getattr(classifier, "group_by_domain", False):
                key = f"{standard.domain}::{key}"
            if not key or key in seen:
                continue
            seen.add(key)
            ranking.append(
                {
                    "category_key": key,
                    "category": standard.category,
                    "domain": standard.domain,
                    "standard_id": standard_id,
                    "document_rank": document_rank,
                    "raw_score": float(
                        similarities[query_index, int(matrix_index)]
                    ),
                }
            )
        result[query] = ranking
    return result


def _rrf_categories(
    rankings: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = Counter()
    representatives: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for route_name, ranking in rankings:
        for category_rank, item in enumerate(ranking, start=1):
            key = item["category_key"]
            scores[key] += 1.0 / (rrf_k + category_rank)
            representatives.setdefault(key, dict(item))
            evidence.setdefault(key, {})[route_name] = {
                "category_rank": category_rank,
                "document_rank": item["document_rank"],
                "raw_score": item["raw_score"],
            }
    ordered = sorted(
        scores,
        key=lambda key: (-scores[key], representatives[key]["category"]),
    )
    return [
        {
            **representatives[key],
            "score": scores[key],
            "routes": evidence[key],
        }
        for key in ordered
    ]


def retrieve_categories_from_cache(
    query_views: Sequence[str],
    vector_by_query: dict[str, list[dict[str, Any]]],
    bm25_by_query: dict[str, list[dict[str, Any]]],
    rrf_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vector_rankings: list[tuple[str, Sequence[dict[str, Any]]]] = []
    hybrid_rankings: list[tuple[str, Sequence[dict[str, Any]]]] = []
    for view_index, query in enumerate(query_views):
        vector = vector_by_query[query]
        bm25 = bm25_by_query[query]
        vector_name = f"vector_q{view_index}"
        bm25_name = f"bm25_q{view_index}"
        vector_rankings.append((vector_name, vector))
        hybrid_rankings.extend(((vector_name, vector), (bm25_name, bm25)))
    return (
        _rrf_categories(vector_rankings, rrf_k),
        _rrf_categories(hybrid_rankings, rrf_k),
    )


def _metrics(
    rows: Sequence[dict[str, Any]],
    candidate_key: str,
    cutoffs: Sequence[int] = (1, 5, 10, 20, 30, 50),
) -> dict[str, Any]:
    from dataset_utils import prediction_metrics

    labeled = [row for row in rows if clean_text(row.get("gold_label"))]
    gold = [canonical_label(row["gold_label"]) for row in labeled]
    predicted = [
        canonical_label(row[candidate_key][0]["category"])
        if row.get(candidate_key)
        else ""
        for row in labeled
    ]
    available_k = max((len(row.get(candidate_key, [])) for row in rows), default=0)
    cutoffs = sorted({k for k in cutoffs if 0 < k <= available_k} | ({available_k} if available_k else set()))
    recall_at: dict[str, Optional[float]] = {}
    for cutoff in cutoffs:
        hits = 0
        for row in labeled:
            target = canonical_label(row["gold_label"])
            candidates = {
                canonical_label(item["category"])
                for item in row.get(candidate_key, [])[:cutoff]
            }
            hits += int(target in candidates)
        recall_at[str(cutoff)] = hits / len(labeled) if labeled else None
    return {
        "evaluated_rows": len(labeled),
        "top1_accuracy": (
            sum(left == right for left, right in zip(gold, predicted))
            / len(labeled)
            if labeled
            else None
        ),
        "macro_f1": prediction_metrics(
            [{"gold_label": g, "prediction": p} for g, p in zip(gold, predicted)],
            "prediction",
        )["macro_f1"],
        "candidate_recall_at": recall_at,
    }


def _compact_candidates(
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {
            "category": item["category"],
            "category_key": item["category_key"],
            "domain": item.get("domain", ""),
            "standard_id": item["standard_id"],
            "score": item["score"],
        }
        for item in candidates[:limit]
    ]


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    config = CONFIGS[args.config]
    embedding_source = str(config["embedding"])
    embedding_model = (
        str(LOCAL_BGE_M3)
        if embedding_source == MULTILINGUAL_EMBEDDING
        and (LOCAL_BGE_M3 / "config.json").exists()
        else embedding_source
    )
    examples: list[FinanceExample] = read_finance_xlsx(
        args.input_xlsx.expanduser().resolve(),
        sheet_name=args.sheet,
        header_row=args.header_row,
        input_column="D",
        label_column="I",
        level_column="J",
    )
    if args.limit:
        examples = examples[: args.limit]
    original_standards = load_corpus(args.corpus.expanduser().resolve())
    standards = (
        expand_standards_with_fragments(original_standards)
        if config["fragments"]
        else original_standards
    )

    init_started = time.perf_counter()
    classifier = XRAGClassifier(
        standards=standards,
        embedding_model=embedding_model,
        retriever_type="hybrid",
        top_k=len(standards),
        cache_dir=args.cache_dir,
        device=args.device,
        recall_k=len(standards),
        rrf_k=args.rrf_k,
    )
    index_seconds = time.perf_counter() - init_started

    views_by_example = [
        finance_query_views(
            example.field_name,
            expanded=bool(config["query_expansion"]),
        )
        for example in examples
    ]
    unique_queries = list(
        dict.fromkeys(query for views in views_by_example for query in views)
    )
    rows: list[dict[str, Any]] = []
    retrieval_started = time.perf_counter()
    vector_by_query = _batched_vector_category_rankings(
        classifier,
        unique_queries,
    )
    bm25_by_query: dict[str, list[dict[str, Any]]] = {}
    for index, query in enumerate(unique_queries, start=1):
        bm25_by_query[query] = _category_ranking(
            classifier,
            classifier.bm25,
            query,
            skip_nonpositive=True,
        )
        if index % 250 == 0 or index == len(unique_queries):
            print(
                f"[{args.config}] BM25 {index}/{len(unique_queries)}",
                flush=True,
            )

    retrieval_cache: dict[
        tuple[str, ...],
        tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    for index, (example, views) in enumerate(
        zip(examples, views_by_example),
        start=1,
    ):
        cache_key = tuple(views)
        if cache_key not in retrieval_cache:
            retrieval_cache[cache_key] = retrieve_categories_from_cache(
                views,
                vector_by_query,
                bm25_by_query,
                args.rrf_k,
            )
        vector, hybrid = retrieval_cache[cache_key]
        rows.append(
            {
                "source_row": example.source_row,
                "source_id": example.source_id,
                "field_name": example.field_name,
                "gold_label": example.gold_label,
                "query_views": views,
                "vector_candidates": _compact_candidates(
                    vector,
                    args.output_k,
                ),
                "hybrid_candidates": _compact_candidates(
                    hybrid,
                    args.output_k,
                ),
            }
        )
        if index % 50 == 0 or index == len(examples):
            print(
                f"[{args.config}] retrieval {index}/{len(examples)}; "
                f"unique field queries={len(retrieval_cache)}",
                flush=True,
            )
    retrieval_seconds = time.perf_counter() - retrieval_started

    metrics = {
        "config": args.config,
        **config,
        "embedding_source": embedding_model,
        "input": str(args.input_xlsx.expanduser().resolve()),
        "corpus": str(args.corpus.expanduser().resolve()),
        "rows": len(rows),
        "original_documents": len(original_standards),
        "indexed_vectors": len(standards),
        "unique_categories": len(
            {canonical_label(item.category) for item in standards}
        ),
        "unique_query_view_sets": len(retrieval_cache),
        "unique_query_views": len(unique_queries),
        "index_seconds": round(index_seconds, 6),
        "retrieval_seconds": round(retrieval_seconds, 6),
        "seconds_per_row": round(retrieval_seconds / len(rows), 6),
        "vector": _metrics(rows, "vector_candidates"),
        "hybrid": _metrics(rows, "hybrid_candidates"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.config}.json"
    metrics_path = args.output_dir / f"{args.config}.metrics.json"
    predictions_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def summarize(output_dir: Path) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for config_name in CONFIGS:
        path = output_dir / f"{config_name}.metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        summary.append(
            {
                "config": config_name,
                "embedding": metrics["embedding"],
                "fragments": metrics["fragments"],
                "query_expansion": metrics["query_expansion"],
                "indexed_vectors": metrics["indexed_vectors"],
                "vector_top1": metrics["vector"]["top1_accuracy"],
                "vector_macro_f1": metrics["vector"]["macro_f1"],
                "vector_recall@5": metrics["vector"]["candidate_recall_at"]["5"],
                "vector_recall@10": metrics["vector"]["candidate_recall_at"]["10"],
                "vector_recall@30": metrics["vector"]["candidate_recall_at"]["30"],
                "vector_recall@50": metrics["vector"]["candidate_recall_at"]["50"],
                "hybrid_top1": metrics["hybrid"]["top1_accuracy"],
                "hybrid_macro_f1": metrics["hybrid"]["macro_f1"],
                "hybrid_recall@5": metrics["hybrid"]["candidate_recall_at"]["5"],
                "hybrid_recall@10": metrics["hybrid"]["candidate_recall_at"]["10"],
                "hybrid_recall@30": metrics["hybrid"]["candidate_recall_at"]["30"],
                "hybrid_recall@50": metrics["hybrid"]["candidate_recall_at"]["50"],
                "index_seconds": metrics["index_seconds"],
                "retrieval_seconds": metrics["retrieval_seconds"],
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="金融知识库纯召回消融实验（不调用 LLM）"
    )
    parser.add_argument("--config", choices=tuple(CONFIGS))
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "finance" / "corpus.json",
    )
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=PROJECT_ROOT.parent / "vector_index" / "finance_test.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--header-row", type=int, default=2)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.summarize and not args.config:
        parser.error("请提供 --config，或使用 --summarize 汇总已有结果")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(args.output_dir), ensure_ascii=False, indent=2))
        return 0
    run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
