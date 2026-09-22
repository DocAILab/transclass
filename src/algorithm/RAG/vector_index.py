#!/usr/bin/env python3
"""XRAG-based retrieval-augmented field classification for TransClass data.

The normalized TransClass metadata becomes the retrieval query. Classification
catalogues under data/knowledge/standards_map become atomic retrieval documents.
An OpenAI-compatible LLM can optionally explain abbreviated field names and select
one result from the retrieved Top-K candidates.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
STANDARDS_DIR = PROJECT_ROOT / "data" / "knowledge" / "standards_map"
DEFAULT_CACHE = PROJECT_ROOT / "cache/indexes"

PROFILE_STANDARD_FILES: dict[str, tuple[str, ...]] = {
    "shougang": ("guanji_dict.json",),
    "finance": ("financial_standards_dict.json",),
    "iov": ("iov_white_papers_dict.json",),
    "education": ("education_dict.json",),
    "personal": (
        "general_personal_info_dict.json",
        "sensitive_personal_info_dict.json",
    ),
    "personal-general": ("general_personal_info_dict.json",),
    "personal-sensitive": ("sensitive_personal_info_dict.json",),
    "beijing-fta": ("beijing_fta_dict.json",),
    "shanghai-fta": ("shanghai_fta_dict.json",),
}

DATASET_DEFAULT_PROFILE = {
    "shougang": "shougang",
    "finance": "finance",
    "pers_info": "personal",
    "iov": "iov",
    "education": "education",
}

LABEL_CODE_SUFFIX = re.compile(
    r"\s*[\(\[（【]\s*[A-Za-z]+\d*(?:-\d+)*\s*[\)\]）】]\s*$"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def canonical_label(value: Any) -> str:
    """Normalize label formatting while preserving the semantic category."""
    text = clean_text(value)
    previous = None
    while text and text != previous:
        previous = text
        text = LABEL_CODE_SUFFIX.sub("", text).strip()
    text = re.sub(r"[_/／>|｜—–]+", "-", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "", text)
    return text.strip("-")


def canonical_level(value: Any) -> str:
    text = clean_text(value).upper().replace("LEVEL", "L").replace("级", "")
    aliases = {"1": "L1", "2": "L2", "3": "L3", "4": "L4"}
    return aliases.get(text, text)


def stable_id(
    domain: str,
    source_file: str,
    scope: str,
    category: str,
    level: str,
) -> str:
    raw = "\x1f".join(
        (domain, source_file, scope, category, level)
    ).encode("utf-8")
    return f"{domain}_{hashlib.sha1(raw).hexdigest()[:12]}"


@dataclass(frozen=True)
class StandardRecord:
    standard_id: str
    domain: str
    source_file: str
    scope: str
    category: str
    level: str
    reference: str
    category_path: tuple[str, ...] = ()

    @property
    def document_text(self) -> str:
        parts = [
            f"分类名称：{self.category}",
            f"定义与范围：{self.scope}",
        ]
        if self.level:
            parts.append(f"数据级别：{self.level}")
        if self.reference:
            parts.append(f"标准来源：{self.reference}")
        return "。".join(parts)


def load_standards(
    paths: Path | Sequence[Path],
    domain: str,
) -> list[StandardRecord]:
    if isinstance(paths, Path):
        paths = [paths]
    records: list[StandardRecord] = []
    for path in paths:
        raw = load_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"标准文件必须是 JSON 对象：{path}")
        for scope, metadata in raw.items():
            if not isinstance(metadata, dict) or not metadata.get("category"):
                raise ValueError(
                    f"标准条目缺少 category：{path.name}: {scope!r}。"
                    "请使用标准 *_dict.json，不要使用 *_expanded.json。"
                )
            category = clean_text(metadata["category"])
            level = clean_text(metadata.get("class"))
            reference = clean_text(metadata.get("ref"))
            scope_text = clean_text(scope)
            records.append(
                StandardRecord(
                    standard_id=stable_id(
                        domain,
                        path.name,
                        scope_text,
                        category,
                        level,
                    ),
                    domain=domain,
                    source_file=path.name,
                    scope=scope_text,
                    category=category,
                    level=level,
                    reference=reference,
                )
            )
    ids = [item.standard_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("合并后的标准库存在重复 standard_id。")
    return records


def nested_get(row: dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = nested_get(row, key) if "." in key else row.get(key)
        text = clean_text(value)
        if text:
            return text
    return ""


QUERY_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("数据库名", ("metadata.database_name", "数据库名")),
    ("数据库描述", ("metadata.database_description", "数据库描述")),
    ("表名", ("metadata.table_name", "表名", "表名称")),
    ("表描述", ("metadata.table_description", "表描述")),
    ("字段名", (
        "metadata.field_name",
        "字段名",
        "列名称",
        "input",
        "ori_input",
    )),
    ("字段描述", ("metadata.field_description", "字段描述")),
    ("字段类型", (
        "metadata.field_type",
        "数据类别",
        "字段属性",
    )),
    ("业务系统", ("系统", "sys_name", "资产的业务系统名称")),
    ("数据资源说明", ("数据资源说明（内容）", "数据资源说明")),
)


def field_name(row: dict[str, Any]) -> str:
    return first_present(
        row,
        ("metadata.field_name", "字段名", "列名称", "input", "ori_input"),
    )


def build_query(
    row: dict[str, Any],
    translated_meaning: str = "",
) -> str:
    """Build a query exclusively from non-label metadata."""
    parts = ["任务：依据数据分类分级标准判断数据库字段所属类别"]
    for label, aliases in QUERY_FIELDS:
        value = first_present(row, aliases)
        if value:
            parts.append(f"{label}：{value}")
    if translated_meaning:
        parts.append(f"模型推断的字段含义：{clean_text(translated_meaning)}")
    return "；".join(parts)


def gold_candidates(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    classification = row.get("classification")
    if isinstance(classification, dict):
        levels = [
            clean_text(classification.get(f"level_{index}"))
            for index in range(1, 5)
        ]
        hierarchy = "-".join(value for value in levels if value)
        if hierarchy:
            candidates.append(hierarchy)
        if levels[-1]:
            candidates.append(levels[-1])
    legacy = first_present(
        row,
        ("四级分类", "label", "gold_category", "gold_category_id"),
    )
    if legacy:
        candidates.append(legacy)

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = canonical_label(candidate)
        if normalized and normalized not in seen:
            result.append(candidate)
            seen.add(normalized)
    return result


class GoldResolver:
    """Resolve normalized or legacy gold labels to stable standard IDs."""

    def __init__(self, standards: Sequence[StandardRecord]) -> None:
        self.standards = list(standards)
        self.by_id = {item.standard_id: item for item in standards}
        self.by_scope: dict[str, list[str]] = {}
        self.by_category: dict[str, list[str]] = {}
        self.by_leaf: dict[str, list[str]] = {}
        for item in standards:
            self.by_scope.setdefault(
                canonical_label(item.scope), []
            ).append(item.standard_id)
            category = canonical_label(item.category)
            self.by_category.setdefault(category, []).append(item.standard_id)
            leaf = category.rsplit("-", 1)[-1]
            self.by_leaf.setdefault(leaf, []).append(item.standard_id)

    def _filter_level(
        self,
        standard_ids: Iterable[str],
        row: dict[str, Any],
    ) -> list[str]:
        values = list(standard_ids)
        gold_level = canonical_level(row.get("data_level"))
        if not gold_level or len(values) <= 1:
            return values
        matching = [
            standard_id
            for standard_id in values
            if canonical_level(self.by_id[standard_id].level) == gold_level
        ]
        return matching or values

    def _unique(
        self,
        standard_ids: Iterable[str],
        row: dict[str, Any],
    ) -> Optional[str]:
        values = self._filter_level(standard_ids, row)
        return values[0] if len(values) == 1 else None

    def resolve(
        self,
        row: dict[str, Any],
    ) -> tuple[Optional[str], str]:
        direct_id = first_present(
            row,
            ("gold_standard_id", "classification.standard_id"),
        )
        if direct_id in self.by_id:
            return direct_id, "standard_id"

        candidates = gold_candidates(row)
        if not candidates:
            return None, "missing"

        ambiguous = False
        for candidate in candidates:
            normalized = canonical_label(candidate)
            for method, index in (
                ("scope_exact", self.by_scope),
                ("category_exact", self.by_category),
                ("category_leaf", self.by_leaf),
            ):
                matches = index.get(normalized, [])
                selected = self._unique(matches, row)
                if selected:
                    return selected, method
                ambiguous = ambiguous or bool(matches)

        category_values = sorted(self.by_category)
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            normalized = canonical_label(candidate)
            for category in category_values:
                scored.append(
                    (
                        difflib.SequenceMatcher(
                            None, normalized, category
                        ).ratio(),
                        category,
                    )
                )
        scored.sort()
        if scored:
            best_score, best_category = scored[-1]
            second_score = scored[-2][0] if len(scored) > 1 else 0.0
            if best_score >= 0.985 and best_score - second_score >= 0.02:
                selected = self._unique(
                    self.by_category[best_category],
                    row,
                )
                if selected:
                    return selected, f"category_fuzzy:{best_score:.4f}"

        return None, "ambiguous" if ambiguous else "unmapped"


class OptionalLLM:
    """OpenAI-compatible helper, disabled unless explicitly requested."""

    def __init__(self) -> None:
        api_key = os.getenv("XRAG_API_KEY", "").strip()
        model = os.getenv("XRAG_LLM_MODEL", "").strip()
        if not api_key or not model:
            raise RuntimeError(
                "使用 --translate 或 --llm-select 时必须设置 "
                "XRAG_API_KEY 和 XRAG_LLM_MODEL；"
                "可选设置 XRAG_API_BASE。"
            )
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai/httpx 依赖，请先执行安装脚本。"
            ) from exc
        arguments: dict[str, Any] = {
            "api_key": api_key,
            "http_client": httpx.Client(follow_redirects=True),
        }
        api_base = os.getenv("XRAG_API_BASE", "").strip()
        if api_base:
            arguments["base_url"] = api_base
        self.client = OpenAI(**arguments)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def translate(self, row: dict[str, Any]) -> str:
        return self.complete(
            "你负责推断数据库字段的中文业务含义。"
            "只输出一个中文词语或短语，不解释。",
            build_query(row),
        ).splitlines()[0].strip()

    def choose(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> Optional[str]:
        compact = [
            {
                "standard_id": item["standard_id"],
                "category": item["category"],
                "level": item["level"],
                "scope": item["scope"],
            }
            for item in candidates
        ]
        raw = self.complete(
            "你负责数据分类。只能从候选中选择一个 standard_id。"
            "只输出 JSON，例如 {\"standard_id\":\"xrag_xxx\"}，"
            "不要创造新类别。",
            f"字段信息：\n{query}\n\n候选标准：\n"
            f"{json.dumps(compact, ensure_ascii=False)}",
        )
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            selected = str(
                json.loads(match.group(0)).get("standard_id", "")
            )
        except json.JSONDecodeError:
            return None
        allowed = {item["standard_id"] for item in candidates}
        return selected if selected in allowed else None


class XRAGClassifier:
    """Atomic-standard index using XRAG embedding/retriever components."""

    def __init__(
        self,
        standards: Sequence[StandardRecord],
        embedding_model: str,
        retriever_type: str,
        top_k: int,
        cache_dir: Path,
        device: str,
        recall_k: Optional[int] = None,
        rrf_k: int = 60,
    ) -> None:
        try:
            from llama_index.core import (
                Document,
                Settings,
                StorageContext,
                VectorStoreIndex,
                load_index_from_storage,
            )
            from xrag.embs.embedding import get_embedding
            from xrag.retrievers.retriever import (
                bm25_retriever,
                vector_retriever,
            )
        except ImportError as exc:
            raise RuntimeError(
                "XRAG 运行依赖尚未安装。请执行："
                "按照交付目录 readme.md 配置运行环境"
            ) from exc

        self.standards = {
            item.standard_id: item for item in standards
        }
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.recall_k = max(top_k, recall_k or top_k)
        self.rrf_k = rrf_k

        # XRAG 0.1.4 exposes only the model name and configures batch size
        # internally.
        embedding = get_embedding(embedding_model)
        if device != "auto":
            model = getattr(embedding, "_model", None)
            if model is not None and hasattr(model, "to"):
                model.to(device)
            if hasattr(embedding, "_device"):
                embedding._device = device
        Settings.embed_model = embedding
        Settings.llm = None

        from domain_config import model_identity
        fingerprint_payload = "\n".join(
            [
                model_identity(embedding_model),
                *sorted(
                    f"{item.domain}\t{item.standard_id}\t{item.document_text}\t{item.category_path}"
                    for item in standards
                ),
            ]
        )
        fingerprint = hashlib.sha1(
            fingerprint_payload.encode("utf-8")
        ).hexdigest()[:12]
        persist_dir = cache_dir / fingerprint
        persist_dir.parent.mkdir(parents=True, exist_ok=True)

        if (persist_dir / "index_store.json").exists():
            context = StorageContext.from_defaults(
                persist_dir=str(persist_dir)
            )
            self.index = load_index_from_storage(context)
        else:
            documents = [
                Document(
                    text=item.document_text,
                    doc_id=item.standard_id,
                    metadata={
                        "id": item.standard_id,
                        "standard_id": item.standard_id,
                        "domain": item.domain,
                        "source_file": item.source_file,
                        "category": item.category,
                        "level": item.level,
                        "reference": item.reference,
                        "scope": item.scope,
                        "category_path": list(item.category_path),
                    },
                )
                for item in standards
            ]
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )
            self.index.storage_context.persist(
                persist_dir=str(persist_dir)
            )

        # XRAG 0.1.4's factories create retrievers with Top-K=3. Override the
        # public property afterwards so the CLI's --top-k remains effective.
        self.vector = vector_retriever(self.index)
        self.bm25 = bm25_retriever(self.index)
        self.vector.similarity_top_k = self.recall_k
        self.bm25.similarity_top_k = self.recall_k

    @staticmethod
    def _node_id(node_with_score: Any) -> str:
        metadata = getattr(
            node_with_score.node,
            "metadata",
            {},
        ) or {}
        return str(
            metadata.get("standard_id")
            or metadata.get("id")
            or node_with_score.node.node_id
        )

    def _retrieve_one(
        self,
        retriever: Any,
        query: str,
    ) -> list[tuple[str, float]]:
        result: list[tuple[str, float]] = []
        for node in retriever.retrieve(query):
            standard_id = self._node_id(node)
            if standard_id in self.standards:
                result.append(
                    (standard_id, float(node.score or 0.0))
                )
        return result

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        result_limit = limit or self.top_k
        route_rankings: dict[str, list[tuple[str, float]]] = {}
        if self.retriever_type == "vector":
            route_rankings["vector"] = self._retrieve_one(
                self.vector,
                query,
            )
        elif self.retriever_type == "bm25":
            route_rankings["bm25"] = self._retrieve_one(
                self.bm25,
                query,
            )
        else:
            route_rankings["vector"] = self._retrieve_one(
                self.vector,
                query,
            )
            route_rankings["bm25"] = self._retrieve_one(
                self.bm25,
                query,
            )

        route_evidence: dict[str, dict[str, dict[str, float | int]]] = {}
        for route_name, ranking in route_rankings.items():
            for rank, (standard_id, raw_score) in enumerate(
                ranking,
                start=1,
            ):
                route_evidence.setdefault(standard_id, {})[route_name] = {
                    "rank": rank,
                    "score": raw_score,
                }

        if len(route_rankings) == 1:
            ranked = next(iter(route_rankings.values()))
        else:
            scores: dict[str, float] = {}
            for ranking in route_rankings.values():
                for rank, (standard_id, _) in enumerate(
                    ranking,
                    start=1,
                ):
                    scores[standard_id] = (
                        scores.get(standard_id, 0.0)
                        + 1.0 / (self.rrf_k + rank)
                    )
            ranked = sorted(
                scores.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )

        result: list[dict[str, Any]] = []
        for standard_id, score in ranked[:result_limit]:
            item = self.standards[standard_id]
            result.append(
                {
                    **asdict(item),
                    "score": score,
                    "routes": route_evidence.get(standard_id, {}),
                }
            )
        return result


def macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> float:
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return 0.0
    values: list[float] = []
    for label in labels:
        true_positive = sum(
            gold_item == label and predicted_item == label
            for gold_item, predicted_item in zip(gold, predicted)
        )
        false_positive = sum(
            gold_item != label and predicted_item == label
            for gold_item, predicted_item in zip(gold, predicted)
        )
        false_negative = sum(
            gold_item == label and predicted_item != label
            for gold_item, predicted_item in zip(gold, predicted)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(values) / len(values)


def resolve_standard_paths(args: argparse.Namespace) -> tuple[list[Path], str]:
    if args.standards:
        profile = args.profile or args.dataset or "custom"
        return [path.resolve() for path in args.standards], profile

    profile = args.profile or DATASET_DEFAULT_PROFILE.get(args.dataset)
    if not profile:
        raise ValueError(
            f"数据集 {args.dataset!r} 没有默认标准库。"
            "请传入 --profile 或一个/多个 --standards。"
        )
    filenames = PROFILE_STANDARD_FILES[profile]
    return [STANDARDS_DIR / filename for filename in filenames], profile


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return args.input.resolve()
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / args.dataset
        / f"{args.split}.json"
    )


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.resolve()
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / args.dataset
        / f"xrag_{args.split}_predictions.json"
    )


def validate_rows(rows: Any, input_path: Path) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError(f"输入必须是 JSON 对象数组：{input_path}")
    if not rows:
        raise ValueError(f"输入数据为空：{input_path}")
    missing = [
        index
        for index, row in enumerate(rows)
        if not field_name(row)
    ]
    if missing:
        examples = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"{len(missing)} 条记录缺少 metadata.field_name，"
            f"索引示例：{examples}"
        )
    return rows


def run(args: argparse.Namespace) -> int:
    standards_paths, profile = resolve_standard_paths(args)
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(args)
    domain = args.domain or profile
    standards = load_standards(standards_paths, domain)
    rows = validate_rows(load_json(input_path), input_path)
    if args.limit:
        rows = rows[: args.limit]

    resolver = GoldResolver(standards)
    resolutions = [resolver.resolve(row) for row in rows]
    labeled = sum(method != "missing" for _, method in resolutions)
    mapped = sum(standard_id is not None for standard_id, _ in resolutions)
    methods = Counter(
        method.split(":", 1)[0] for _, method in resolutions
    )

    if args.validate_only:
        report = {
            "dataset": args.dataset,
            "input": str(input_path),
            "rows": len(rows),
            "standards": len(standards),
            "standards_files": [
                str(path.relative_to(PROJECT_ROOT))
                if path.is_relative_to(PROJECT_ROOT)
                else str(path)
                for path in standards_paths
            ],
            "labeled_rows": labeled,
            "mapped_gold": mapped,
            "unmapped_or_ambiguous_gold": labeled - mapped,
            "methods": methods,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if mapped == labeled else 2

    llm = OptionalLLM() if args.translate or args.llm_select else None
    index_started = time.perf_counter()
    classifier = XRAGClassifier(
        standards=standards,
        embedding_model=args.embedding,
        retriever_type=args.retriever,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
        device=args.device,
        recall_k=getattr(args, "recall_k", 0) or args.top_k,
        rrf_k=getattr(args, "rrf_k", 60),
    )
    index_init_seconds = time.perf_counter() - index_started

    predictions: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    for index, (row, resolution) in enumerate(zip(rows, resolutions)):
        translated = (
            llm.translate(row)
            if args.translate and llm
            else ""
        )
        query = build_query(row, translated)
        candidates = classifier.retrieve(query)
        selected_id = (
            candidates[0]["standard_id"] if candidates else None
        )
        selection_method = "retrieval_top1"
        if args.llm_select and llm and candidates:
            llm_choice = llm.choose(query, candidates)
            if llm_choice:
                selected_id = llm_choice
                selection_method = "llm_candidate_selection"
            else:
                selection_method = "llm_invalid_fallback_top1"

        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["standard_id"] == selected_id
            ),
            None,
        )
        gold_id, gold_resolution = resolution
        gold_standard = (
            asdict(resolver.by_id[gold_id])
            if gold_id
            else None
        )
        predictions.append(
            {
                "row_index": index,
                "source_id": clean_text(row.get("id")) or None,
                "field_name": field_name(row),
                "query": query,
                "translated_meaning": translated or None,
                "prediction": selected,
                "selection_method": selection_method,
                "candidates": candidates,
                "gold_standard_id": gold_id,
                "gold_standard": gold_standard,
                "gold_resolution": gold_resolution,
                "correct": (
                    selected_id == gold_id if gold_id else None
                ),
                "top_k_hit": (
                    gold_id in {
                        candidate["standard_id"]
                        for candidate in candidates
                    }
                    if gold_id
                    else None
                ),
            }
        )
    inference_seconds = time.perf_counter() - inference_started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    mapped_predictions = [
        item for item in predictions if item["gold_standard_id"]
    ]
    gold_ids = [
        str(item["gold_standard_id"])
        for item in mapped_predictions
    ]
    predicted_ids = [
        str(
            (item["prediction"] or {}).get(
                "standard_id",
                "__NONE__",
            )
        )
        for item in mapped_predictions
    ]
    denominator = len(mapped_predictions)
    metrics = {
        "dataset": args.dataset,
        "profile": profile,
        "input": str(input_path),
        "rows": len(predictions),
        "standards": len(standards),
        "labeled_rows": labeled,
        "mapped_gold": denominator,
        "unmapped_or_ambiguous_gold": labeled - denominator,
        "accuracy": (
            sum(item["correct"] is True for item in mapped_predictions)
            / denominator
            if denominator
            else None
        ),
        f"recall_at_{args.top_k}": (
            sum(item["top_k_hit"] is True for item in mapped_predictions)
            / denominator
            if denominator
            else None
        ),
        "macro_f1": (
            macro_f1(gold_ids, predicted_ids)
            if denominator
            else None
        ),
        "retriever": args.retriever,
        "recall_k_per_route": getattr(args, "recall_k", 0)
        or args.top_k,
        "rrf_k": getattr(args, "rrf_k", 60),
        "embedding": args.embedding,
        "device": args.device,
        "index_init_seconds": round(index_init_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "seconds_per_row": (
            round(inference_seconds / len(predictions), 6)
            if predictions
            else None
        ),
        "llm_translation": args.translate,
        "llm_selection": args.llm_select,
        "output": str(output_path),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TransClass XRAG 检索增强字段分类"
    )
    parser.add_argument(
        "--dataset",
        default="shougang",
        help="data/processed 下的数据集目录名",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="test",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="覆盖 data/processed/<dataset>/<split>.json",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_STANDARD_FILES),
        help="选择内置标准库组合",
    )
    parser.add_argument(
        "--standards",
        action="append",
        type=Path,
        help="自定义标准 JSON；可重复传入以合并多个标准库",
    )
    parser.add_argument("--domain", help="稳定 standard_id 的领域前缀")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
    )
    parser.add_argument(
        "--embedding",
        default="BAAI/bge-small-zh-v1.5",
    )
    parser.add_argument(
        "--retriever",
        choices=("vector", "bm25", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--recall-k",
        type=int,
        default=0,
        help="每一路召回的候选数；0 表示与 --top-k 相同",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 融合常数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 条；0 表示全部",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="用兼容 OpenAI 的 LLM 推断字段含义",
    )
    parser.add_argument(
        "--llm-select",
        action="store_true",
        help="让 LLM 只在 Top-K 候选中复选",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验新数据结构和标签映射，不加载 XRAG",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    if args.recall_k < 0:
        parser.error("--recall-k 不能小于 0")
    if args.rrf_k < 1:
        parser.error("--rrf-k 必须大于 0")
    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    return args


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if os.getenv("XRAG_DEBUG") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
