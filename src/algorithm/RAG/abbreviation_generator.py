#!/usr/bin/env python3
"""Dynamic abbreviation expansion backed by a two-level SQLite cache."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from abbreviation_response import parse_abbreviation_response
from domain_config import resolve_domain
from vector_index import clean_text


PROMPT_VERSION = "routed_or_generic_dynamic_abbreviation_v3"
SEED_MEANINGS: dict[str, tuple[str, str]] = {
    "ID": ("identifier", "标识"),
    "NO": ("number", "编号"),
    "NAME": ("name", "名称"),
    "DATE": ("date", "日期"),
    "TIME": ("time", "时间"),
    "CODE": ("code", "代码"),
}


def normalize_field(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", clean_text(value)).upper()


def rule_parts(value: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", clean_text(value))
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", spaced)
    spaced = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", spaced)
    return [
        part.upper()
        for part in re.split(r"[^A-Za-z0-9]+", spaced)
        if part
    ]


@dataclass(frozen=True)
class Meaning:
    english: str
    chinese: str
    confidence: float


@dataclass(frozen=True)
class Segment:
    token: str
    meanings: tuple[Meaning, ...]


@dataclass(frozen=True)
class FieldExpansion:
    field_name: str
    normalized_field: str
    segments: tuple[Segment, ...]
    meanings: tuple[Meaning, ...]
    source: str
    model: str
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AbbreviationStore:
    def __init__(self, path: Path, domain: str = "finance",
                 model: Optional[str] = None, prompt_version: str = PROMPT_VERSION) -> None:
        self.path = path
        self.domain = resolve_domain(domain).key
        self.model = model
        self.prompt_version = prompt_version
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._insert_seeds()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS abbreviation_meanings (
                domain TEXT NOT NULL,
                token TEXT NOT NULL,
                english TEXT NOT NULL,
                chinese TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                context_field TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    domain, token, english, chinese, source,
                    model, prompt_version
                )
            );
            CREATE TABLE IF NOT EXISTS field_expansions (
                domain TEXT NOT NULL,
                normalized_field TEXT NOT NULL,
                original_field TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                meanings_json TEXT NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    domain, normalized_field, model, prompt_version
                )
            );
            CREATE INDEX IF NOT EXISTS idx_abbreviation_token
                ON abbreviation_meanings(domain, token, confidence DESC);
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _insert_seeds(self) -> None:
        now = self._now()
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO abbreviation_meanings (
                domain, token, english, chinese, confidence, source,
                model, prompt_version, context_field, created_at
            ) VALUES (?, ?, ?, ?, ?, 'seed', 'local', ?, '', ?)
            """,
            [
                (
                    self.domain,
                    token,
                    english,
                    chinese,
                    1.0,
                    self.prompt_version,
                    now,
                )
                for token, (english, chinese) in SEED_MEANINGS.items()
            ],
        )
        self.connection.commit()

    def meanings_for_token(
        self,
        token: str,
        limit: int = 2,
    ) -> tuple[Meaning, ...]:
        rows = self.connection.execute(
            """
            SELECT english, chinese, MAX(confidence) AS best_confidence
            FROM abbreviation_meanings
            WHERE domain = ? AND token = ?
              AND (source = 'seed' OR (prompt_version = ? AND (? IS NULL OR model = ?)))
            GROUP BY english, chinese
            ORDER BY best_confidence DESC, english, chinese
            LIMIT ?
            """,
            (self.domain, normalize_field(token), self.prompt_version, self.model, self.model, limit),
        ).fetchall()
        return tuple(
            Meaning(clean_text(row[0]), clean_text(row[1]), float(row[2]))
            for row in rows
        )

    def known_tokens(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT token FROM abbreviation_meanings WHERE domain = ? "
                "AND (source = 'seed' OR (prompt_version = ? AND (? IS NULL OR model = ?)))",
                (self.domain, self.prompt_version, self.model, self.model),
            )
        }

    def get_field(
        self,
        field_name: str,
        model: str,
        prompt_version: Optional[str] = None,
    ) -> Optional[FieldExpansion]:
        prompt_version = prompt_version or self.prompt_version
        normalized = normalize_field(field_name)
        row = self.connection.execute(
            """
            SELECT original_field, segments_json, meanings_json, source
            FROM field_expansions
            WHERE domain = ? AND normalized_field = ?
              AND model = ? AND prompt_version = ?
            """,
            (self.domain, normalized, model, prompt_version),
        ).fetchone()
        if row is None:
            return None
        segment_values = json.loads(row[1])
        meaning_values = json.loads(row[2])
        return FieldExpansion(
            field_name=clean_text(field_name),
            normalized_field=normalized,
            segments=tuple(
                Segment(
                    token=clean_text(item["token"]).upper(),
                    meanings=tuple(
                        Meaning(
                            clean_text(value["english"]),
                            clean_text(value["chinese"]),
                            float(value["confidence"]),
                        )
                        for value in item.get("meanings", [])
                    ),
                )
                for item in segment_values
            ),
            meanings=tuple(
                Meaning(
                    clean_text(item["english"]),
                    clean_text(item["chinese"]),
                    float(item["confidence"]),
                )
                for item in meaning_values
            ),
            source=clean_text(row[3]),
            model=model,
            prompt_version=prompt_version,
        )

    def save_field(self, expansion: FieldExpansion) -> None:
        if expansion.prompt_version != self.prompt_version or (self.model is not None and expansion.model != self.model):
            raise ValueError("字段缓存的模型或提示词版本与当前词典不匹配")
        now = self._now()
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO field_expansions (
                    domain, normalized_field, original_field, segments_json,
                    meanings_json, source, model, prompt_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.domain,
                    expansion.normalized_field,
                    expansion.field_name,
                    json.dumps(
                        [asdict(value) for value in expansion.segments],
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        [asdict(value) for value in expansion.meanings],
                        ensure_ascii=False,
                    ),
                    expansion.source,
                    expansion.model,
                    expansion.prompt_version,
                    now,
                ),
            )
            if expansion.source != "llm":
                return
            for segment in expansion.segments:
                for meaning in segment.meanings:
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO abbreviation_meanings (
                            domain, token, english, chinese, confidence,
                            source, model, prompt_version, context_field,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, 'llm', ?, ?, ?, ?)
                        """,
                        (
                            self.domain,
                            segment.token,
                            meaning.english,
                            meaning.chinese,
                            meaning.confidence,
                            expansion.model,
                            expansion.prompt_version,
                            expansion.normalized_field,
                            now,
                        ),
                    )

    def stats(self) -> dict[str, int]:
        token_count = self.connection.execute(
            "SELECT COUNT(DISTINCT token) FROM abbreviation_meanings WHERE domain = ?",
            (self.domain,),
        ).fetchone()[0]
        meaning_count = self.connection.execute(
            "SELECT COUNT(*) FROM abbreviation_meanings WHERE domain = ?",
            (self.domain,),
        ).fetchone()[0]
        field_count = self.connection.execute(
            "SELECT COUNT(*) FROM field_expansions WHERE domain = ?",
            (self.domain,),
        ).fetchone()[0]
        llm_fields = self.connection.execute(
            """
            SELECT COUNT(*) FROM field_expansions
            WHERE domain = ? AND source = 'llm'
            """,
            (self.domain,),
        ).fetchone()[0]
        return {
            "tokens": int(token_count),
            "meanings": int(meaning_count),
            "fields": int(field_count),
            "llm_fields": int(llm_fields),
        }

    def close(self) -> None:
        self.connection.close()


def split_with_known(
    field_name: str,
    known_tokens: set[str],
) -> list[str]:
    keys = sorted(
        (key for key in known_tokens if key),
        key=lambda value: (-len(value), value),
    )
    result: list[str] = []
    for part in rule_parts(field_name):
        if part.isdigit() or part in known_tokens:
            result.append(part)
            continue
        unknown = ""
        index = 0
        while index < len(part):
            matched = next(
                (key for key in keys if part.startswith(key, index)),
                None,
            )
            if matched:
                if unknown:
                    result.append(unknown)
                    unknown = ""
                result.append(matched)
                index += len(matched)
            else:
                unknown += part[index]
                index += 1
        if unknown:
            result.append(unknown)
    return result


def compose_known_expansion(
    field_name: str,
    tokens: Sequence[str],
    store: AbbreviationStore,
    model: str,
) -> Optional[FieldExpansion]:
    segment_meanings = [store.meanings_for_token(token) for token in tokens]
    if not tokens or any(not meanings for meanings in segment_meanings):
        return None
    segments = tuple(
        Segment(token, meanings)
        for token, meanings in zip(tokens, segment_meanings)
    )
    primary = Meaning(
        " ".join(values[0].english for values in segment_meanings),
        "".join(values[0].chinese for values in segment_meanings),
        min(values[0].confidence for values in segment_meanings),
    )
    meanings = [primary]
    for token_index, values in enumerate(segment_meanings):
        if len(values) < 2:
            continue
        english = [item[0].english for item in segment_meanings]
        chinese = [item[0].chinese for item in segment_meanings]
        english[token_index] = values[1].english
        chinese[token_index] = values[1].chinese
        meanings.append(
            Meaning(
                " ".join(english),
                "".join(chinese),
                min(
                    values[1].confidence,
                    *(items[0].confidence for items in segment_meanings),
                ),
            )
        )
        break
    return FieldExpansion(
        field_name=clean_text(field_name),
        normalized_field=normalize_field(field_name),
        segments=segments,
        meanings=tuple(meanings[:2]),
        source="cache_compose",
        model=model,
    )


def build_offline_dictionary(field_names, store):
    """Seed meanings and deterministic splitting only; never construct an API client."""
    expansions = {}
    source_counts = {}
    for field_name in dict.fromkeys(field_names):
        tokens = split_with_known(field_name, store.known_tokens())
        expansion = compose_known_expansion(field_name, tokens, store, "local_rules")
        if expansion is None:
            expansion = FieldExpansion(
                field_name=field_name, normalized_field=normalize_field(field_name),
                segments=tuple(Segment(token, store.meanings_for_token(token)) for token in tokens),
                meanings=(), source="rule_only", model="local_rules",
            )
        expansions[field_name] = expansion
        source_counts[expansion.source] = source_counts.get(expansion.source, 0) + 1
    return expansions, {
        "mode": "offline_seed_and_rules", "unique_fields": len(expansions),
        "source_counts": source_counts, "api_requests": 0, "llm_generated_fields": 0,
        "store": store.stats(),
    }


class AbbreviationGenerator:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout: float = 90.0,
        max_retries: int = 2,
        request_delay: float = 3.0,
        rate_limit_delay: float = 60.0,
        domain: str = "finance",
        diagnostics_path: Optional[Path] = None,
    ) -> None:
        self.diagnostics_path = diagnostics_path
        self.domain = resolve_domain(domain).key
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"请先设置环境变量 {api_key_env}。")
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai/httpx 依赖。") from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(timeout=timeout, follow_redirects=True),
        )
        self.model = model
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.rate_limit_delay = rate_limit_delay
        self.api_requests = 0
        self.successful_batches = 0
        self.rate_limit_retries = 0
        self.failed_batches = 0

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
        value = getattr(exc, "status_code", None)
        return value if isinstance(value, int) else None

    def _write_diagnostic(self, value: dict[str, Any]) -> None:
        path = getattr(self, "diagnostics_path", None)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), **value}
            # Append each event before parsing or retrying, preserving raw content exactly.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            # Diagnostics must not trigger another paid API request or mask its error.
            print(f"响应日志写入失败（{type(exc).__name__}）：{path}", file=sys.stderr, flush=True)

    @staticmethod
    def _usage_dict(response: Any) -> Optional[dict[str, Any]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump(mode="json")
        return {name: getattr(usage, name, None)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")}

    @staticmethod
    def _output_budget(tasks: Sequence[dict[str, Any]]) -> int:
        # Allow both segment meanings and up to two full-field meanings. This is
        # an output ceiling, not a request to pad otherwise concise responses.
        estimate = sum(
            max(1200, 500 + 240 * len(task.get("rule_parts") or rule_parts(task["field_name"])))
            for task in tasks
        )
        return max(1600, min(16000, estimate))

    @classmethod
    def _json_mode_unsupported(cls, exc: Exception) -> bool:
        if cls._status_code(exc) not in (400, 422):
            return False
        message = str(exc).lower()
        return (
            ("response_format" in message or "json_object" in message)
            and any(term in message for term in (
                "not support", "unsupported", "unknown", "unrecognized", "invalid parameter",
            ))
        )

    def generate_batch(
        self,
        tasks: Sequence[dict[str, Any]],
        on_success: Optional[Callable[[FieldExpansion], None]] = None,
    ) -> list[FieldExpansion]:
        """Generate expansions, persisting validated items before retrying failures.

        Malformed JSON is never patched or accepted. Only the failed portion is
        requested again, with smaller batches down to individually bounded retries.
        Callback errors intentionally propagate outside the API retry handler.
        """
        return self._generate_with_recovery(list(tasks), on_success)

    def _generate_with_recovery(
        self,
        tasks: list[dict[str, Any]],
        on_success: Optional[Callable[[FieldExpansion], None]],
        parent_batch_id: Optional[str] = None,
        correction_hint: str = "",
    ) -> list[FieldExpansion]:
        if not tasks:
            return []
        accepted, batch_id, next_hint = self._request_expansions(
            tasks, parent_batch_id, correction_hint,
        )
        for index in sorted(accepted):
            if on_success is not None:
                on_success(accepted[index])
        # Persist first so interruption during pacing cannot lose valid results.
        if self.request_delay:
            time.sleep(self.request_delay)
        missing = [index for index in range(len(tasks)) if index not in accepted]
        if not missing:
            return [accepted[index] for index in range(len(tasks))]
        # A bad single response is exhausted inside _request_expansions. Each
        # recursive recovery therefore either shrinks the task set or halves it.
        groups = [missing]
        if len(missing) == len(tasks):
            midpoint = len(missing) // 2
            groups = [missing[:midpoint], missing[midpoint:]]
        self._write_diagnostic({
            "event": "recovery", "batch_id": batch_id, "domain": self.domain,
            "strategy": "retry_failed_items" if accepted else "split_batch",
            "accepted_count": len(accepted), "pending_count": len(missing),
            "child_batch_sizes": [len(group) for group in groups],
        })
        for group in groups:
            recovered = self._generate_with_recovery(
                [tasks[index] for index in group], on_success, parent_batch_id=batch_id,
                correction_hint=next_hint,
            )
            accepted.update(zip(group, recovered))
        return [accepted[index] for index in range(len(tasks))]

    def _validated_items(
        self,
        tasks: Sequence[dict[str, Any]],
        parsed: dict[str, Any],
        issues: Optional[list[str]] = None,
    ) -> dict[int, FieldExpansion]:
        issues = issues if issues is not None else []
        values = parsed.get("results")
        if not isinstance(values, list):
            raise ValueError("缩写响应缺少results数组。")
        by_id: dict[str, list[dict[str, Any]]] = {}
        for value in values:
            if isinstance(value, dict):
                by_id.setdefault(clean_text(value.get("item_id")), []).append(value)
        accepted = {}
        for index, task in enumerate(tasks):
            matching = by_id.get(f"item_{index}", [])
            if len(matching) != 1:
                issues.append(f"item_{index}缺失" if not matching else f"item_{index}重复")
                continue  # Missing/duplicate IDs cannot be assigned safely.
            value = matching[0]
            normalized = normalize_field(task["field_name"])
            if any(
                key in value and normalize_field(value[key]) != normalized
                for key in ("field_name", "normalized_field")
            ):
                issues.append(f"item_{index}字段名不匹配")
                continue
            if ("field_meanings" not in value and _valid_meanings(parsed.get("field_meanings"))):
                issues.append(
                    f"item_{index}缺少任务内的field_meanings；根层完整含义无法确定归属，需单独重试"
                )
                continue
            try:
                accepted[index] = validate_generated_expansion(
                    task["field_name"], value, self.model,
                )
            except (ValueError, TypeError, AttributeError) as exc:
                issues.append(f"item_{index}含义或拆分结构无效：{exc}")
                continue
        return accepted

    def _request_expansions(
        self,
        tasks: Sequence[dict[str, Any]],
        parent_batch_id: Optional[str],
        correction_hint: str = "",
    ) -> tuple[dict[int, FieldExpansion], str, str]:
        domain = resolve_domain(self.domain)
        introduction = (
            "你是通用数据库元数据缩写分析专家。所属领域未知，不得预设领域。"
            "只根据字段字符串和通用数据库"
            if self.domain == "generic" else
            f"你是{domain.chinese}数据库元数据缩写分析专家。已知所属领域为{domain.chinese}。"
            "只根据字段字符串、已知领域和通用数据库"
        )
        system = (
            introduction +
            "命名常识拆分字段并生成英文、中文含义；不要推断或输出数据分类标签。"
            "连续大写字段可以联合拆分。segments 的 token 按顺序拼接后必须与"
            "normalized_field 完全一致，不得删除或增加字符。每个 segment 和完整"
            "字段最多给2个合理含义；不确定时保留原token并降低confidence。"
            "每个item_id必须返回且只返回一次。只输出完整合法的JSON对象，"
            "使用紧凑格式，不输出Markdown、注释或解释；含义用简短词组表达，"
            "字符串中的双引号必须转义，所有数组和对象必须闭合。以下ACC_NO仅为格式示例，"
            "实际只输出当前tasks中的字段："
            '{"results":[{"item_id":"item_0","field_name":"ACC_NO",'
            '"normalized_field":"ACCNO",'
            '"field_meanings":[{"english":"account number",'
            '"chinese":"账户编号","confidence":0.9}],'
            '"segments":[{"token":"ACC","meanings":[{"english":"account",'
            '"chinese":"账户","confidence":0.9}]},'
            '{"token":"NO","meanings":[{"english":"number",'
            '"chinese":"编号","confidence":0.9}]}]}]}。'
        )
        expected_ids = [f"item_{index}" for index in range(len(tasks))]
        system += (
            f"本批有{len(tasks)}条任务，results数组必须有{len(tasks)}个独立任务对象，"
            f"item_id按顺序为{','.join(expected_ids)}。"
            "根对象只允许出现一次results键；禁止重复results键或在根对象直接放item_id、"
            "segments、field_meanings。每条任务的segments和field_meanings必须放在该任务对象内；"
            "所有任务共用一个results数组，不要为每条任务重复生成根对象。"
            "先写每条任务的完整field_meanings，再写segments；token和meanings只允许放在"
            "segments数组的元素内，不能直接放在任务对象或results数组中。"
            "严格填写用户消息的response_template；空字符串是待填写占位符，必须替换为"
            "实际内容。模板中的segments可以按实际拆分扩展，每个片段一个对象；"
            "field_meanings必须包含完整字段的英文和中文含义，不能省略或移到根节点。"
        )
        # Concrete identities and a complete container for every requested item
        # reduce omissions. Empty placeholders cannot pass meaning validation.
        template_meaning = {"english": "", "chinese": "", "confidence": 0.0}
        payload = {"task_count": len(tasks), "expected_item_ids": expected_ids, "tasks": [
            {**task, "item_id": f"item_{index}"}
            for index, task in enumerate(tasks)
        ], "response_template": {"results": [
            {"item_id": f"item_{index}", "field_name": task["field_name"],
             "normalized_field": normalize_field(task["field_name"]),
             "field_meanings": [dict(template_meaning)],
             "segments": [{"token": "", "meanings": [dict(template_meaning)]}]}
            for index, task in enumerate(tasks)
        ]}}
        batch_id = uuid.uuid4().hex
        max_tokens = self._output_budget(tasks)
        attempt = 0
        retry_hint = correction_hint
        while attempt <= self.max_retries:
            response = None
            finish_reason = None
            stage = "request"
            accepted: dict[int, FieldExpansion] = {}
            validation_issues: list[str] = []
            json_mode = getattr(self, "_json_mode_enabled", True)
            diagnostic = {
                "batch_id": batch_id, "parent_batch_id": parent_batch_id,
                "attempt": attempt + 1, "max_attempts": self.max_retries + 1,
                "model": self.model, "domain": self.domain,
                "field_names": [task["field_name"] for task in tasks],
                "batch_size": len(tasks), "max_tokens": max_tokens,
                "json_mode": json_mode,
                "correction_mode": bool(retry_hint),
            }
            try:
                self.api_requests += 1
                response = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system + retry_hint},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                )
                stage = "response"
                choice = response.choices[0] if response.choices else None
                original_content = choice.message.content if choice is not None else None
                finish_reason = getattr(choice, "finish_reason", None)
                self._write_diagnostic({
                    **diagnostic, "event": "response",
                    "response_id": getattr(response, "id", None),
                    "request_id": getattr(response, "_request_id", None),
                    "finish_reason": finish_reason, "usage": self._usage_dict(response),
                    "raw_content": original_content,
                })
                stage = "parse"
                if not isinstance(original_content, str):
                    raise ValueError("缩写响应正文不是文本。")
                # Do not normalize whitespace inside JSON strings before decoding.
                parsed, repairs = parse_abbreviation_response(original_content, tasks)
                if repairs:
                    self._write_diagnostic({
                        **diagnostic, "event": "structure_recovery", "repairs": repairs,
                    })
                stage = "validation"
                accepted = self._validated_items(tasks, parsed, validation_issues)
                if len(accepted) != len(tasks):
                    raise ValueError(
                        "；".join(validation_issues) or "缩写响应任务不完整"
                    )
            except Exception as exc:
                status = self._status_code(exc)
                if response is None:
                    error_kind = "http_error" if status is not None else "request_error"
                    detail = f"HTTP {status}" if status is not None else f"请求错误：{type(exc).__name__}"
                elif finish_reason == "length":
                    error_kind = "output_truncated"
                    detail = f"响应截断：finish_reason=length，max_tokens={max_tokens}"
                elif stage == "parse":
                    error_kind = "json_parse_error"
                    detail = f"JSON解析失败：{exc}"
                else:
                    error_kind = "response_validation_error"
                    detail = f"响应结构校验失败：{exc}"
                error = {
                    **diagnostic, "event": "failure", "error_kind": error_kind,
                    "error_type": type(exc).__name__, "http_status": status,
                    "finish_reason": finish_reason, "accepted_count": len(accepted),
                    "validation_issues": validation_issues,
                }
                if isinstance(exc, (ValueError, TypeError)) and response is not None:
                    error["error_message"] = str(exc)
                if isinstance(exc, json.JSONDecodeError):
                    error["parse_position"] = {"line": exc.lineno, "column": exc.colno, "offset": exc.pos}
                    error["parse_position_basis"] = "JSON body after trimming whitespace and optional Markdown fence, not raw_content"
                self._write_diagnostic(error)
                path = getattr(self, "diagnostics_path", None)
                log_note = f"；响应日志：{path}" if path is not None else ""
                if response is None and json_mode and self._json_mode_unsupported(exc):
                    self._json_mode_enabled = False
                    print(f"接口不支持JSON输出约束，改用提示词约束并继续。{log_note}",
                          file=sys.stderr, flush=True)
                    continue  # One compatibility retry; disabled for subsequent requests.
                if response is not None and stage in ("parse", "validation") and len(tasks) > 1:
                    print(f"缩写生成响应失败（{detail}），已验证{len(accepted)}条，"
                          f"缩小批次重试剩余任务。{log_note}", file=sys.stderr, flush=True)
                    return accepted, batch_id, self._correction_hint(stage)
                if status == 429:
                    self.rate_limit_retries += 1
                # Retrying authentication or other permanent client errors only wastes time.
                permanent_http = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
                if attempt >= self.max_retries or permanent_http:
                    self.failed_batches += 1
                    print(f"缩写生成批次最终失败（{detail}）{log_note}", file=sys.stderr, flush=True)
                    raise RuntimeError(f"缩写生成批次失败：{detail}") from exc
                if response is not None:
                    if finish_reason == "length" or stage == "parse":
                        max_tokens = min(16000, max_tokens * 2)
                    retry_hint = self._correction_hint(stage)
                delay = (min(self.rate_limit_delay * (2**attempt), 300.0)
                         if status == 429 else min(float(2**attempt), 30.0))
                print(f"缩写生成批次失败（{detail}），{delay:.1f}秒后重试"
                      f"{attempt + 2}/{self.max_retries + 1}。{log_note}", file=sys.stderr, flush=True)
                time.sleep(delay)
                attempt += 1
                continue
            self.successful_batches += 1
            return accepted, batch_id, ""
        raise AssertionError("unreachable abbreviation retry state")

    @staticmethod
    def _correction_hint(stage: str) -> str:
        issue = "JSON语法未通过校验" if stage == "parse" else "有任务遗漏、含义缺失或层级错误"
        # Child batches renumber IDs. Do not carry parent item IDs or its orphan
        # meanings into a different task's repair request.
        return (
            f"\n上次响应{issue}。本次为纠错请求，只处理当前tasks，"
            "以当前response_template中的item_id和field_name为准。"
            "逐项填写完整字段field_meanings的english和chinese，再填写segments。"
            "检查每条任务都返回且只返回一次，field_meanings位于该任务对象内；"
            "每个token及其meanings位于segments的同一个元素内。"
            "不能将任何任务或片段提前闭合到外层。返回前检查完整JSON的括号和逗号。"
        )

def _valid_meanings(value: Any) -> tuple[Meaning, ...]:
    if not isinstance(value, list):
        return ()
    result: list[Meaning] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        english = clean_text(item.get("english"))[:160]
        chinese = clean_text(item.get("chinese"))[:160]
        if not english or not chinese:
            continue
        key = (english.casefold(), chinese)
        if key in seen:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        result.append(
            Meaning(english, chinese, max(0.0, min(confidence, 1.0)))
        )
        seen.add(key)
        if len(result) >= 2:
            break
    return tuple(result)


def validate_generated_expansion(
    field_name: str,
    value: dict[str, Any],
    model: str,
) -> FieldExpansion:
    normalized = normalize_field(field_name)
    segments: list[Segment] = []
    raw_segments = value.get("segments")
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            token = normalize_field(item.get("token", ""))
            meanings = _valid_meanings(item.get("meanings"))
            if token and meanings:
                segments.append(Segment(token, meanings))
    field_meanings = _valid_meanings(value.get("field_meanings"))
    if not field_meanings:
        raise ValueError(f"字段{field_name!r}没有合法的完整中英文含义。")
    split_valid = bool(segments) and (
        "".join(item.token for item in segments) == normalized
    )
    if not split_valid:
        # Keep the full-field meaning but do not let a bad split poison the
        # reusable token dictionary with invented or missing characters.
        segments = [Segment(normalized, field_meanings)]
    return FieldExpansion(
        field_name=clean_text(field_name),
        normalized_field=normalized,
        segments=tuple(segments),
        meanings=field_meanings,
        source="llm" if split_valid else "llm_unsplit",
        model=model,
    )


def expansion_query_views(expansion: FieldExpansion, domain: str = "finance") -> list[str]:
    config = resolve_domain(domain)
    task = ("任务：判断数据库字段的叶子类别；所属领域未知；"
            if config.key == "generic" else
            f"任务：判断{config.chinese}数据库字段的叶子类别；")
    base = (
        task +
        f"字段名称：{expansion.field_name}；"
        f"字段拆分：{' '.join(item.token for item in expansion.segments)}"
    )
    candidates = [base]
    for meaning in expansion.meanings[:2]:
        candidates.extend(
            [
                (
                    f"{config.english} database field {expansion.field_name}; "
                    f"meaning: {meaning.english}"
                ),
                (
                    f"{config.chinese}数据库字段 {expansion.field_name}；"
                    f"中文含义：{meaning.chinese}"
                ),
            ]
        )
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = clean_text(candidate)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result
