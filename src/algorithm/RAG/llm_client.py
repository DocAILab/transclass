#!/usr/bin/env python3
"""Finance-field experiment: multi-route recall, RRF fusion, and LLM reranking.

The experiment intentionally uses only column D (field name) from the input
workbook to build retrieval queries. Column I is read only as the gold
fourth-level category for offline evaluation.
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence
from xml.etree import ElementTree

from domain_config import resolve_domain

from vector_index import (
    DEFAULT_CACHE,
    StandardRecord,
    XRAGClassifier,
    canonical_label,
    clean_text,
    load_json,
    macro_f1,
    stable_id,
)


PROJECT_ROOT = Path(os.environ.get("RAG_WORKSPACE_ROOT", "/Users/andiandian/Desktop/trandatacls")).expanduser().resolve() / "transclass_repo"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finance_xrag_rerank"
    / "predictions.json"
)
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
MODELSCOPE_DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"

SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
CELL_REFERENCE = re.compile(r"([A-Z]+)(\d+)")


@dataclass(frozen=True)
class FinanceExample:
    source_row: int
    source_id: str
    field_name: str
    gold_label: str
    gold_level: str
    domain: str = ""
    label_status: str = ""


def read_test_json(path: Path, use_domain: bool = True) -> list[FinanceExample]:
    """Read labels for evaluation; field_name is the only business input."""
    raw = load_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("测试集必须是非空 JSON 对象数组")
    examples = []
    seen = set()
    for index, row in enumerate(raw, 1):
        if not isinstance(row, dict):
            raise ValueError(f"测试集第 {index} 条不是对象")
        metadata = row.get("metadata")
        classification = row.get("classification")
        grading = row.get("grading", {})
        if not all(isinstance(value, dict) for value in (metadata, classification, grading)):
            raise ValueError(f"测试集第 {index} 条结构不合法")
        identifier = clean_text(row.get("id"))
        name = clean_text(metadata.get("field_name"))
        label = clean_text(classification.get("category_leaf_level"))
        if not identifier or identifier in seen or not name or not label:
            raise ValueError(f"测试集第 {index} 条 ID 重复或 ID/字段名/叶子标签为空")
        seen.add(identifier)
        examples.append(FinanceExample(
            source_row=index, source_id=identifier, field_name=name,
            gold_label=label, gold_level=clean_text(grading.get("sensitivity_level")),
            domain=(row.get("domain") if isinstance(row.get("domain"), str) else "")
            if use_domain else "",
            label_status=clean_text(row.get("label_status")),
        ))
    return examples


@dataclass(frozen=True)
class RerankDecision:
    standard_id: str
    ranked_standard_ids: list[str]
    reason: str
    raw_response: str
    cached: bool
    repair_note: str = ""


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    return [
        "".join(
            node.text or ""
            for node in item.iter(f"{{{SHEET_NS}}}t")
        )
        for item in root.findall(f"{{{SHEET_NS}}}si")
    ]


def _worksheet_path(
    archive: zipfile.ZipFile,
    sheet_name: str,
) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = ""
    for sheet in workbook.findall(
        f".//{{{SHEET_NS}}}sheet"
    ):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(
                f"{{{REL_NS}}}id",
                "",
            )
            break
    if not relationship_id:
        available = [
            sheet.attrib.get("name", "")
            for sheet in workbook.findall(
                f".//{{{SHEET_NS}}}sheet"
            )
        ]
        raise ValueError(
            f"工作簿中没有工作表 {sheet_name!r}；可用工作表：{available}"
        )

    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relationship in relationships.findall(
        f"{{{PACKAGE_REL_NS}}}Relationship"
    ):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            if target.startswith("xl/"):
                return target
            return str(PurePosixPath("xl") / target)
    raise ValueError(f"无法解析工作表 {sheet_name!r} 的 XML 路径。")


def _cell_value(
    cell: ElementTree.Element,
    shared_strings: Sequence[str],
) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iter(f"{{{SHEET_NS}}}t")
        )
    value_node = cell.find(f"{{{SHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"非法 shared string 索引：{raw}") from exc
    return raw


def read_finance_xlsx(
    path: Path,
    sheet_name: str = "Sheet1",
    header_row: int = 2,
    input_column: str = "D",
    label_column: str = "I",
    level_column: str = "J",
    id_column: str = "A",
) -> list[FinanceExample]:
    """Read the experiment columns without requiring a heavyweight XLSX library."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        worksheet = ElementTree.fromstring(
            archive.read(_worksheet_path(archive, sheet_name))
        )

    rows: dict[int, dict[str, str]] = {}
    for row_node in worksheet.findall(f".//{{{SHEET_NS}}}row"):
        for cell in row_node.findall(f"{{{SHEET_NS}}}c"):
            match = CELL_REFERENCE.fullmatch(cell.attrib.get("r", ""))
            if not match:
                continue
            column, row_text = match.groups()
            rows.setdefault(int(row_text), {})[column] = _cell_value(
                cell,
                shared_strings,
            )

    header = rows.get(header_row, {})
    input_header = clean_text(header.get(input_column))
    label_header = clean_text(header.get(label_column))
    if input_header != "字段名称" or label_header != "四级分类":
        raise ValueError(
            "Excel 列定义与实验约定不一致："
            f"{input_column}{header_row}={input_header!r}，"
            f"{label_column}{header_row}={label_header!r}；"
            "预期分别为“字段名称”和“四级分类”。"
        )

    examples: list[FinanceExample] = []
    for row_number in sorted(rows):
        if row_number <= header_row:
            continue
        row = rows[row_number]
        field_name = clean_text(row.get(input_column))
        if not field_name:
            continue
        examples.append(
            FinanceExample(
                source_row=row_number,
                source_id=clean_text(row.get(id_column))
                or str(row_number),
                field_name=field_name,
                gold_label=clean_text(row.get(label_column)),
                gold_level=clean_text(row.get(level_column)),
            )
        )
    if not examples:
        raise ValueError(f"Excel 中没有可用的 {input_column} 列输入：{path}")
    return examples


def _first_text(mapping: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = clean_text(mapping.get(key))
        if value:
            return value
    return ""


CATEGORY_KEYS = (
    "category_leaf_level",
    "category",
    "label",
    "target",
    "title",
    "name",
    "四级分类",
    "分类名称",
)
TEXT_KEYS = (
    "data_description_and_example",
    "scope",
    "text",
    "content",
    "document",
    "page_content",
    "description",
    "definition",
    "定义与范围",
)
LEVEL_KEYS = ("sensitivity_level", "class", "level", "data_level", "数据级别")
REFERENCE_KEYS = ("reference_standard", "ref", "reference", "source", "标准来源")
CONTAINER_KEYS = ("corpus", "documents", "records", "items", "data")


def _hierarchical_category(item: dict[str, Any]) -> str:
    english = [
        clean_text(item.get(f"level_{index}"))
        for index in range(1, 5)
    ]
    chinese = [
        clean_text(item.get(key))
        for key in ("一级子类", "二级子类", "三级子类", "四级分类")
    ]
    values = english if any(english) else chinese
    return "-".join(value for value in values if value)


def _unwrap_corpus(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    for key in CONTAINER_KEYS:
        value = raw.get(key)
        if isinstance(value, (list, dict)):
            return value
    return raw


def _record_from_mapping(
    item: dict[str, Any],
    fallback_scope: str,
    domain: str,
    source_file: str,
) -> StandardRecord:
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    combined = {**metadata, **item}
    category = _first_text(combined, CATEGORY_KEYS)
    if not category:
        category = _hierarchical_category(combined)
    scope = _first_text(combined, TEXT_KEYS) or clean_text(fallback_scope)
    level = _first_text(combined, LEVEL_KEYS)
    reference = _first_text(combined, REFERENCE_KEYS)
    if not category:
        raise ValueError(
            "corpus 条目缺少类别字段。支持的字段包括："
            f"{', '.join(CATEGORY_KEYS)}；条目：{item!r}"
        )
    if not scope:
        scope = category
    return StandardRecord(
        standard_id=clean_text(combined.get("id")) or stable_id(
            domain,
            source_file,
            scope,
            category,
            level,
        ),
        domain=domain,
        source_file=source_file,
        scope=scope,
        category=category,
        level=level,
        reference=reference,
        category_path=tuple(clean_text(combined.get(key)) for key in (
            "category_root_level", "category_branch_level",
            "category_subbranch_level", "category_leaf_level"
        ) if clean_text(combined.get(key))),
    )


def load_corpus(path: Path, domain: str = "finance") -> list[StandardRecord]:
    """Load common mapping/list/LangChain-style JSON corpus layouts."""
    raw = _unwrap_corpus(load_json(path))
    records: list[StandardRecord] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                records.append(
                    _record_from_mapping(
                        value,
                        fallback_scope=clean_text(key),
                        domain=domain,
                        source_file=path.name,
                    )
                )
            elif isinstance(value, str):
                records.append(
                    _record_from_mapping(
                        {"category": key, "text": value},
                        fallback_scope=key,
                        domain=domain,
                        source_file=path.name,
                    )
                )
            else:
                raise ValueError(
                    f"不支持的 corpus 映射值类型：{key!r} -> "
                    f"{type(value).__name__}"
                )
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                records.append(
                    _record_from_mapping(
                        item,
                        fallback_scope=str(index),
                        domain=domain,
                        source_file=path.name,
                    )
                )
            else:
                raise ValueError(
                    "corpus 数组中的每个条目必须是 JSON 对象；"
                    f"索引 {index} 是 {type(item).__name__}。"
                )
    else:
        raise ValueError("corpus.json 顶层必须是 JSON 对象或对象数组。")

    if not records:
        raise ValueError(f"知识库为空：{path}")
    identifiers = [record.standard_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("corpus 中存在生成相同 standard_id 的重复条目。")
    return records


def expand_field_name(field_name: str) -> str:
    raw = clean_text(field_name)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    spaced = re.sub(r"[_\-.]+", " ", spaced)
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", spaced)
    spaced = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", spaced)
    return re.sub(r"\s+", " ", spaced).strip()


def build_finance_query(field_name: str, mode: str = "expanded") -> str:
    raw = clean_text(field_name)
    if mode == "raw":
        return raw
    expanded = expand_field_name(raw)
    return (
        "任务：判断金融数据库字段的四级分类；"
        f"字段名称：{raw}；字段拆分：{expanded}"
    )


def collapse_candidates_by_category(
    document_candidates: Sequence[dict[str, Any]],
    limit: int,
    max_definitions: int = 3,
) -> list[dict[str, Any]]:
    """Collapse atomic corpus documents into unique label candidates.

    This operation uses only labels from corpus.json. Test-set gold labels are
    deliberately not consulted.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for document in document_candidates:
        category = clean_text(document.get("category"))
        key = canonical_label(category)
        if not key:
            continue
        if key not in grouped:
            representative = dict(document)
            representative["target_label"] = category
            representative["supporting_documents"] = []
            grouped[key] = representative
            order.append(key)
        supporting = grouped[key]["supporting_documents"]
        if len(supporting) < max_definitions:
            supporting.append(
                {
                    "standard_id": document["standard_id"],
                    "scope": document["scope"],
                    "score": document["score"],
                    "routes": document.get("routes", {}),
                }
            )

    result: list[dict[str, Any]] = []
    for key in order[:limit]:
        candidate = grouped[key]
        definitions = list(
            dict.fromkeys(
                clean_text(item["scope"])
                for item in candidate["supporting_documents"]
                if clean_text(item["scope"])
            )
        )
        candidate["scope"] = "；".join(definitions)
        result.append(candidate)
    return result


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        stripped = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM 响应中没有 JSON 对象。")
        stripped = match.group(0)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("LLM 响应 JSON 必须是对象。")
    return value


class JsonlRerankCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")
        self._lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise RuntimeError(
                "已有另一个重排进程正在使用缓存："
                f"{self.path}。请勿同时启动两个实验。"
            ) from exc
        self.values: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"重排缓存第 {line_number} 行不是合法 JSON：{path}"
                        ) from exc
                    if isinstance(item, dict) and item.get("cache_key"):
                        self.values[str(item["cache_key"])] = item

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self.values.get(key)

    def close(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is None or handle.closed:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    def append(self, key: str, value: dict[str, Any]) -> None:
        if key in self.values:
            return
        payload = {"cache_key": key, **value}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
        self.values[key] = payload


class OpenAICompatibleReranker:
    domain = "finance"
    PROMPT_VERSION = "routed_or_generic_leaf_rerank_v3"

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str,
        cache: JsonlRerankCache,
        timeout: float,
        max_retries: int,
        request_delay: float,
        rate_limit_delay: float,
        candidate_text_chars: int,
        domain: str = "finance",
    ) -> None:
        self.domain = resolve_domain(domain).key
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"启用大模型重排前请设置环境变量 {api_key_env}。"
                "密钥不要写入命令行参数、代码或 Git。"
            )
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai/httpx 依赖，请先执行安装脚本。"
            ) from exc
        # openai 1.47's bundled default transport still passes the removed
        # `proxies` argument when the environment has httpx >= 0.28. Supplying
        # a plain client keeps the experiment compatible with both versions.
        http_client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.cache = cache
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.rate_limit_delay = rate_limit_delay
        self.candidate_text_chars = candidate_text_chars
        self.api_requests = 0
        self.successful_batches = 0
        self.cache_hits = 0
        self.rate_limit_retries = 0
        self.failed_batches = 0
        self.repaired_items = 0

    def _cache_key(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> str:
        payload = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "query": query,
            "domain": self.domain,
            "prompt_version": self.PROMPT_VERSION,
            "candidates": self._compact_candidates(candidates),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _decision_from_cache(
        cached: dict[str, Any],
    ) -> RerankDecision:
        return RerankDecision(
            standard_id=str(cached["standard_id"]),
            ranked_standard_ids=[
                str(value)
                for value in cached.get("ranked_standard_ids", [])
            ],
            reason=clean_text(cached.get("reason")),
            raw_response=clean_text(cached.get("raw_response")),
            cached=True,
            repair_note=clean_text(cached.get("repair_note")),
        )

    def _compact_candidates(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "standard_id": candidate["standard_id"],
                "category": candidate["category"],
                "domain": candidate.get("domain", ""),
                "definition": clean_text(candidate.get("scope"))[
                    : self.candidate_text_chars
                ],
            }
            for candidate in candidates
        ]

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        status_code = getattr(exc, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    def _retry_seconds(
        self,
        exc: Exception,
        attempt: int,
    ) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        if self._status_code(exc) == 429:
            return min(
                self.rate_limit_delay * (2 ** attempt),
                300.0,
            )
        return min(float(2 ** attempt), 30.0)

    def rerank_batch(
        self,
        requests: Sequence[
            tuple[str, str, Sequence[dict[str, Any]]]
        ],
    ) -> list[RerankDecision]:
        """Rerank several independent fields with one API request.

        Cache keys remain per field, so caches created by the previous
        single-item implementation are reused without conversion.
        """
        decisions: list[Optional[RerankDecision]] = [
            None for _ in requests
        ]
        pending: list[
            tuple[
                int,
                str,
                str,
                str,
                Sequence[dict[str, Any]],
            ]
        ] = []
        for index, (field_name, query, candidates) in enumerate(requests):
            cache_key = self._cache_key(query, candidates)
            cached = self.cache.get(cache_key)
            if cached:
                decisions[index] = self._decision_from_cache(cached)
                self.cache_hits += 1
                continue
            pending.append(
                (index, cache_key, field_name, query, candidates)
            )

        if not pending:
            return [
                decision
                for decision in decisions
                if decision is not None
            ]

        payload_items: list[dict[str, Any]] = []
        allowed_by_item: dict[str, set[str]] = {}
        for batch_index, (
            _,
            _,
            field_name,
            query,
            candidates,
        ) in enumerate(pending):
            item_id = f"item_{batch_index}"
            allowed_by_item[item_id] = {
                str(candidate["standard_id"])
                for candidate in candidates
            }
            payload_items.append(
                {
                    "item_id": item_id,
                    "field_name": field_name,
                    "query": query,
                    "candidates": self._compact_candidates(candidates),
                }
            )

        domain = resolve_domain(self.domain)
        introduction = (
            "你是通用数据库字段分类专家。所属领域未知，不得预设字段属于某个领域。"
            "候选来自多个领域，请根据字段名含义与候选定义进行跨领域比较。"
            if self.domain == "generic" else
            f"你是{domain.chinese}数据分类专家，已知所属领域为{domain.chinese}。"
        )
        system = (
            introduction + "需要独立处理多条字段的叶子类别分类任务。"
            "每条任务只能从该任务的 candidates 中选择已有 standard_id，"
            "不得跨任务选择或创造类别。字段名是唯一业务输入。"
            "必须为每个 item_id 返回且只返回一条结果。"
            "只输出 JSON 对象，格式为："
            '{"results":[{"item_id":"item_0",'
            '"standard_id":"候选ID",'
            '"ranked_standard_ids":["候选ID1","候选ID2"],'
            '"reason":"不超过60字的中文理由"}]}。'
        )
        user = json.dumps(
            {"tasks": payload_items},
            ensure_ascii=False,
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self.api_requests += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=max(500, min(2000, 180 * len(pending))),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw = (
                    response.choices[0].message.content or ""
                ).strip()
                parsed = _json_object(raw)
                parsed_results = parsed.get("results")
                if not isinstance(parsed_results, list):
                    raise ValueError("LLM 批量响应缺少 results 数组。")
                result_by_item: dict[str, dict[str, Any]] = {}
                for result in parsed_results:
                    if not isinstance(result, dict):
                        raise ValueError("results 中存在非对象元素。")
                    item_id = clean_text(result.get("item_id"))
                    if (
                        item_id not in allowed_by_item
                        or item_id in result_by_item
                    ):
                        raise ValueError(
                            f"LLM 返回了未知或重复 item_id：{item_id!r}"
                        )
                    result_by_item[item_id] = result
                if set(result_by_item) != set(allowed_by_item):
                    missing = sorted(
                        set(allowed_by_item) - set(result_by_item)
                    )
                    raise ValueError(
                        f"LLM 批量响应缺少任务：{missing}"
                    )

                validated: list[
                    tuple[
                        int,
                        str,
                        dict[str, Any],
                        RerankDecision,
                    ]
                ] = []
                for batch_index, (
                    original_index,
                    cache_key,
                    _,
                    _,
                    candidates,
                ) in enumerate(pending):
                    item_id = f"item_{batch_index}"
                    result = result_by_item[item_id]
                    allowed = allowed_by_item[item_id]
                    model_selected = clean_text(
                        result.get("standard_id")
                    )
                    parsed_ranking = result.get(
                        "ranked_standard_ids",
                        [],
                    )
                    if not isinstance(parsed_ranking, list):
                        parsed_ranking = []
                    valid_ranking = [
                        clean_text(value)
                        for value in parsed_ranking
                        if clean_text(value) in allowed
                    ]
                    repair_note = ""
                    if model_selected in allowed:
                        selected = model_selected
                    elif valid_ranking:
                        selected = valid_ranking[0]
                        repair_note = (
                            "模型首选 ID 越界，采用该任务排序列表中"
                            "排名最高的合法候选"
                        )
                    else:
                        selected = str(candidates[0]["standard_id"])
                        repair_note = (
                            "模型首选 ID 越界且无合法排序候选，"
                            "回退到融合检索第一名"
                        )
                    ranking = list(
                        dict.fromkeys([selected, *valid_ranking])
                    )
                    reason = clean_text(result.get("reason"))[:200]
                    item_raw = json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    stored = {
                        "standard_id": selected,
                        "ranked_standard_ids": ranking,
                        "reason": reason,
                        "raw_response": item_raw,
                        "repair_note": repair_note,
                    }
                    validated.append(
                        (
                            original_index,
                            cache_key,
                            stored,
                            RerankDecision(
                                standard_id=selected,
                                ranked_standard_ids=ranking,
                                reason=reason,
                                raw_response=item_raw,
                                cached=False,
                                repair_note=repair_note,
                            ),
                        )
                    )
                for (
                    original_index,
                    cache_key,
                    stored,
                    decision,
                ) in validated:
                    self.cache.append(cache_key, stored)
                    decisions[original_index] = decision
                    self.repaired_items += bool(decision.repair_note)
                self.successful_batches += 1
                if self.request_delay:
                    time.sleep(self.request_delay)
                assert all(
                    decision is not None for decision in decisions
                )
                return [
                    decision
                    for decision in decisions
                    if decision is not None
                ]
            except Exception as exc:  # Provider-specific API exceptions.
                last_error = exc
                status_code = self._status_code(exc)
                if status_code == 429:
                    self.rate_limit_retries += 1
                if attempt >= self.max_retries:
                    self.failed_batches += 1
                    break
                delay = self._retry_seconds(exc, attempt)
                print(
                    "LLM 批次请求失败"
                    f"（HTTP {status_code or 'unknown'}），"
                    f"{delay:.1f} 秒后重试 "
                    f"{attempt + 2}/{self.max_retries + 1}。",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert last_error is not None
        raise RuntimeError(
            "LLM 批量重排在 "
            f"{self.max_retries + 1} 次尝试后失败：{last_error}"
        ) from last_error

    def rerank(
        self,
        field_name: str,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> RerankDecision:
        return self.rerank_batch(
            [(field_name, query, candidates)]
        )[0]


def _provider_defaults(
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
) -> tuple[str, str, str]:
    if provider == "modelscope":
        return (
            model or MODELSCOPE_DEFAULT_MODEL,
            base_url or MODELSCOPE_BASE_URL,
            api_key_env or "MODELSCOPE_API_TOKEN",
        )
    if provider == "openrouter":
        return (
            model or OPENROUTER_DEFAULT_MODEL,
            base_url or OPENROUTER_BASE_URL,
            api_key_env or "OPENROUTER_API_KEY",
        )
    if not model or not base_url or not api_key_env:
        raise ValueError(
            "使用 custom LLM 时必须同时传入 --llm-model、"
            "--llm-base-url 和 --llm-api-key-env。"
        )
    return model, base_url, api_key_env


def _prediction_metrics(
    rows: Sequence[dict[str, Any]],
    prediction_key: str,
) -> dict[str, Any]:
    labeled = [
        row
        for row in rows
        if clean_text(row.get("gold_label"))
        and clean_text(row.get(prediction_key))
    ]
    gold = [clean_text(row["gold_label"]) for row in labeled]
    predicted = [clean_text(row[prediction_key]) for row in labeled]
    return {
        "evaluated_rows": len(labeled),
        "accuracy": (
            sum(
                canonical_label(left) == canonical_label(right)
                for left, right in zip(gold, predicted)
            )
            / len(labeled)
            if labeled
            else None
        ),
        "macro_f1": macro_f1(
            [canonical_label(value) for value in gold],
            [canonical_label(value) for value in predicted],
        )
        if labeled
        else None,
    }


def _candidate_recall_at(
    rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[int] = (1, 5, 10, 20, 30, 50, 100),
) -> dict[str, Optional[float]]:
    labeled = [
        row for row in rows if clean_text(row.get("gold_label"))
    ]
    result: dict[str, Optional[float]] = {}
    for cutoff in cutoffs:
        hits = 0
        for row in labeled:
            gold = canonical_label(row["gold_label"])
            candidate_labels = [
                canonical_label(candidate["target_label"])
                for candidate in row.get("candidates", [])[:cutoff]
            ]
            hits += int(gold in candidate_labels)
        result[str(cutoff)] = (
            hits / len(labeled) if labeled else None
        )
    return result


def _gold_label_audit(
    gold_labels: Sequence[str],
    standards: Sequence[StandardRecord],
) -> dict[str, Any]:
    corpus_labels = sorted(
        {
            clean_text(standard.category)
            for standard in standards
            if clean_text(standard.category)
        }
    )
    normalized_corpus = {
        canonical_label(label): label for label in corpus_labels
    }
    unmatched = []
    for gold_label in gold_labels:
        normalized = canonical_label(gold_label)
        if normalized in normalized_corpus:
            continue
        closest = difflib.get_close_matches(
            clean_text(gold_label),
            corpus_labels,
            n=1,
            cutoff=0.5,
        )
        unmatched.append(
            {
                "gold_label": gold_label,
                "closest_corpus_label": closest[0] if closest else None,
            }
        )
    return {
        "exact_match_count": len(gold_labels) - len(unmatched),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
    }


def _compact_output_row(row: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value for key, value in row.items() if key != "candidates"
    }
    candidates = row.get("candidates", [])
    compact["candidate_labels"] = [
        candidate["target_label"] for candidate in candidates
    ]
    compact["candidate_standard_ids"] = [
        candidate["standard_id"] for candidate in candidates
    ]
    return compact


def run(args: argparse.Namespace) -> int:
    corpus_path = args.corpus.expanduser().resolve()
    input_path = args.input_xlsx.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    examples = read_finance_xlsx(
        input_path,
        sheet_name=args.sheet,
        header_row=args.header_row,
        input_column=args.input_column,
        label_column=args.label_column,
        level_column=args.level_column,
    )
    if args.limit:
        examples = examples[: args.limit]
    standards = load_corpus(corpus_path, domain=args.domain)
    gold_labels = sorted(
        {
            example.gold_label
            for example in examples
            if example.gold_label
        }
    )
    validation = {
        "input": str(input_path),
        "corpus": str(corpus_path),
        "rows": len(examples),
        "labeled_rows": sum(bool(item.gold_label) for item in examples),
        "unique_gold_labels": len(gold_labels),
        "standards": len(standards),
        "unique_corpus_labels": len(
            {canonical_label(item.category) for item in standards}
        ),
        "input_column": args.input_column,
        "label_column": args.label_column,
        "gold_distribution": Counter(
            item.gold_label for item in examples if item.gold_label
        ),
        "gold_label_audit": _gold_label_audit(
            gold_labels,
            standards,
        ),
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    init_started = time.perf_counter()
    classifier = XRAGClassifier(
        standards=standards,
        embedding_model=args.embedding,
        retriever_type=args.retriever,
        top_k=args.document_pool_k,
        cache_dir=args.cache_dir,
        device=args.device,
        recall_k=args.recall_k,
        rrf_k=args.rrf_k,
    )
    index_init_seconds = time.perf_counter() - init_started

    reranker: Optional[OpenAICompatibleReranker] = None
    llm_model = None
    llm_base_url = None
    llm_api_key_env = None
    if args.rerank:
        llm_model, llm_base_url, llm_api_key_env = _provider_defaults(
            args.llm_provider,
            args.llm_model,
            args.llm_base_url,
            args.llm_api_key_env,
        )
        rerank_cache = (
            args.rerank_cache.expanduser().resolve()
            if args.rerank_cache
            else output_path.with_suffix(".rerank-cache.jsonl")
        )
        reranker = OpenAICompatibleReranker(
            provider=args.llm_provider,
            model=llm_model,
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            cache=JsonlRerankCache(rerank_cache),
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            request_delay=args.llm_request_delay,
            rate_limit_delay=args.llm_rate_limit_delay,
            candidate_text_chars=args.llm_candidate_text_chars,
        )

    rows: list[dict[str, Any]] = []
    inference_started = time.perf_counter()
    llm_failures = 0
    for example in examples:
        query = build_finance_query(example.field_name, args.query_mode)
        document_candidates = classifier.retrieve(
            query,
            limit=args.document_pool_k,
        )
        candidates = collapse_candidates_by_category(
            document_candidates,
            limit=args.fusion_k,
        )
        retrieval_prediction = (
            clean_text(candidates[0]["target_label"])
            if candidates
            else ""
        )
        candidate_hit = (
            any(
                canonical_label(candidate["target_label"])
                == canonical_label(example.gold_label)
                for candidate in candidates
            )
            if example.gold_label
            else None
        )
        rerank_prediction = ""
        rerank_standard_id = None
        rerank_reason = None
        rerank_method = None
        rerank_error = None
        rerank_repair = None

        rows.append(
            {
                "source_row": example.source_row,
                "source_id": example.source_id,
                "field_name": example.field_name,
                "query": query,
                "gold_label": example.gold_label or None,
                "gold_level": example.gold_level or None,
                "retrieval_prediction": retrieval_prediction or None,
                "retrieval_correct": (
                    canonical_label(retrieval_prediction)
                    == canonical_label(example.gold_label)
                    if example.gold_label and retrieval_prediction
                    else None
                ),
                "candidate_hit": candidate_hit,
                "candidates": candidates,
                "rerank_prediction": rerank_prediction or None,
                "rerank_standard_id": rerank_standard_id,
                "rerank_reason": rerank_reason,
                "rerank_method": rerank_method,
                "rerank_error": rerank_error,
                "rerank_repair": rerank_repair,
                "rerank_correct": (
                    canonical_label(rerank_prediction)
                    == canonical_label(example.gold_label)
                    if example.gold_label and rerank_prediction
                    else None
                ),
            }
        )

    if reranker:
        rerank_indexes = [
            index
            for index, row in enumerate(rows)
            if row["candidates"]
        ]
        for start in range(
            0,
            len(rerank_indexes),
            args.llm_batch_size,
        ):
            batch_indexes = rerank_indexes[
                start : start + args.llm_batch_size
            ]
            requests = [
                (
                    rows[index]["field_name"],
                    rows[index]["query"],
                    rows[index]["candidates"],
                )
                for index in batch_indexes
            ]
            try:
                decisions = reranker.rerank_batch(requests)
                for index, decision in zip(
                    batch_indexes,
                    decisions,
                ):
                    row = rows[index]
                    selected = next(
                        candidate
                        for candidate in row["candidates"]
                        if candidate["standard_id"]
                        == decision.standard_id
                    )
                    prediction = clean_text(
                        selected["target_label"]
                    )
                    row["rerank_prediction"] = prediction or None
                    row["rerank_standard_id"] = decision.standard_id
                    row["rerank_reason"] = decision.reason
                    row["rerank_repair"] = (
                        decision.repair_note or None
                    )
                    if decision.cached:
                        row["rerank_method"] = (
                            "llm_cache_repaired"
                            if decision.repair_note
                            else "llm_cache"
                        )
                    else:
                        row["rerank_method"] = (
                            "llm_batch_rerank_repaired"
                            if decision.repair_note
                            else "llm_batch_rerank"
                        )
                    row["rerank_correct"] = (
                        canonical_label(prediction)
                        == canonical_label(row["gold_label"])
                        if row["gold_label"] and prediction
                        else None
                    )
            except Exception as exc:
                llm_failures += len(batch_indexes)
                if args.llm_failure_policy == "error":
                    raise
                for index in batch_indexes:
                    row = rows[index]
                    row["rerank_error"] = str(exc)
                    row["rerank_prediction"] = row[
                        "retrieval_prediction"
                    ]
                    row["rerank_standard_id"] = row["candidates"][
                        0
                    ]["standard_id"]
                    row["rerank_method"] = (
                        "llm_error_fallback_retrieval_top1"
                    )
                    row["rerank_correct"] = row["retrieval_correct"]
    inference_seconds = time.perf_counter() - inference_started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = (
        rows
        if args.include_candidate_details
        else [_compact_output_row(row) for row in rows]
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    labeled_rows = [
        row for row in rows if clean_text(row.get("gold_label"))
    ]
    metrics = {
        **validation,
        "retriever": args.retriever,
        "embedding": args.embedding,
        "device": args.device,
        "query_mode": args.query_mode,
        "recall_k_per_route": args.recall_k,
        "document_pool_k": args.document_pool_k,
        "fusion_k": args.fusion_k,
        "rrf_k": args.rrf_k,
        "candidate_recall": (
            sum(row["candidate_hit"] is True for row in labeled_rows)
            / len(labeled_rows)
            if labeled_rows
            else None
        ),
        "candidate_recall_at": _candidate_recall_at(rows),
        "retrieval": _prediction_metrics(
            rows,
            "retrieval_prediction",
        ),
        "rerank_enabled": args.rerank,
        "rerank": (
            _prediction_metrics(rows, "rerank_prediction")
            if args.rerank
            else None
        ),
        "llm": {
            "provider": args.llm_provider if args.rerank else None,
            "model": llm_model,
            "base_url": llm_base_url,
            "api_key_env": llm_api_key_env,
            "batch_size": args.llm_batch_size if args.rerank else None,
            "api_requests": (
                reranker.api_requests if reranker else 0
            ),
            "successful_batches": (
                reranker.successful_batches if reranker else 0
            ),
            "cache_hits": (
                reranker.cache_hits if reranker else 0
            ),
            "rate_limit_retries": (
                reranker.rate_limit_retries if reranker else 0
            ),
            "failed_batches": (
                reranker.failed_batches if reranker else 0
            ),
            "repaired_items": (
                reranker.repaired_items if reranker else 0
            ),
            "repaired_rows": sum(
                bool(row.get("rerank_repair")) for row in rows
            ),
            "failures": llm_failures,
        },
        "index_init_seconds": round(index_init_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "seconds_per_row": (
            round(inference_seconds / len(rows), 6) if rows else None
        ),
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
        description="金融字段多路召回、RRF 融合与免费 LLM API 重排实验"
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--input-xlsx", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--header-row", type=int, default=2)
    parser.add_argument("--input-column", default="D")
    parser.add_argument("--label-column", default="I")
    parser.add_argument("--level-column", default="J")
    parser.add_argument("--domain", default="finance")
    parser.add_argument(
        "--query-mode",
        choices=("raw", "expanded"),
        default="expanded",
        help="expanded 只对 D 列字段名做切词，不引入其他 Excel 列",
    )
    parser.add_argument(
        "--retriever",
        choices=("vector", "bm25", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--embedding",
        default="BAAI/bge-small-zh-v1.5",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=30,
        help="向量和 BM25 每一路分别召回的数量",
    )
    parser.add_argument(
        "--fusion-k",
        type=int,
        default=30,
        help="按四级类别去重后交给 LLM 的候选数量",
    )
    parser.add_argument(
        "--document-pool-k",
        type=int,
        default=0,
        help="RRF 后、类别去重前保留的文档数；0 表示两路召回总量",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="调用 OpenAI 兼容的免费 LLM API 对融合候选重排",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("modelscope", "openrouter", "custom"),
        default="modelscope",
    )
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument(
        "--llm-api-key-env",
        help="保存 API Key 的环境变量名；不接受明文 Key 参数",
    )
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=0,
        help="免费额度优先：默认失败即停，重新运行时通过缓存续跑",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=8,
        help="一次 API 请求合并处理的字段数量",
    )
    parser.add_argument(
        "--llm-candidate-text-chars",
        type=int,
        default=100,
        help="批量 Prompt 中每个候选定义保留的最大字符数",
    )
    parser.add_argument(
        "--llm-request-delay",
        type=float,
        default=3.0,
        help="成功批次之间的等待秒数",
    )
    parser.add_argument(
        "--llm-rate-limit-delay",
        type=float,
        default=60.0,
        help="429 未返回 Retry-After 时的首次等待秒数",
    )
    parser.add_argument(
        "--llm-failure-policy",
        choices=("fallback", "error"),
        default="error",
        help="默认遇到失败即停止，保留缓存后续续跑",
    )
    parser.add_argument("--rerank-cache", type=Path)
    parser.add_argument(
        "--include-candidate-details",
        action="store_true",
        help="在预测 JSON 中保存候选定义和逐路分数；默认仅保存候选标签和 ID",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证 corpus 和 Excel 的 D/I 列，不加载 XRAG",
    )
    args = parser.parse_args(argv)
    args.input_column = args.input_column.upper()
    args.label_column = args.label_column.upper()
    args.level_column = args.level_column.upper()
    if args.header_row < 1:
        parser.error("--header-row 必须大于 0")
    if args.recall_k < 1:
        parser.error("--recall-k 必须大于 0")
    if args.document_pool_k < 0:
        parser.error("--document-pool-k 不能小于 0")
    if args.document_pool_k == 0:
        args.document_pool_k = args.recall_k * (
            2 if args.retriever == "hybrid" else 1
        )
    if args.fusion_k < 1:
        parser.error("--fusion-k 必须大于 0")
    if args.fusion_k > args.document_pool_k:
        parser.error("--fusion-k 不应大于 --document-pool-k")
    if args.rrf_k < 1:
        parser.error("--rrf-k 必须大于 0")
    if args.limit < 0:
        parser.error("--limit 不能小于 0")
    if args.llm_max_retries < 0:
        parser.error("--llm-max-retries 不能小于 0")
    if args.llm_batch_size < 1:
        parser.error("--llm-batch-size 必须大于 0")
    if args.llm_candidate_text_chars < 20:
        parser.error("--llm-candidate-text-chars 不能小于 20")
    if args.llm_request_delay < 0:
        parser.error("--llm-request-delay 不能小于 0")
    if args.llm_rate_limit_delay < 0:
        parser.error("--llm-rate-limit-delay 不能小于 0")
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
