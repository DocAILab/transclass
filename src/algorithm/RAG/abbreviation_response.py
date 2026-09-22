"""Decode abbreviation JSON without losing records or guessing missing content.

Repairs only relocate already generated values with an unambiguous task identity.
Semantic validation of meanings and segment reconstruction remains the caller's job.
"""

from __future__ import annotations

import os

import json
import re
from typing import Any, Sequence


class _ObjectPairs(list):
    """Distinguish JSON objects from arrays while preserving duplicate keys."""


def _normalized(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _matching_task(value: dict[str, Any], tasks: Sequence[dict[str, Any]]) -> int | None:
    """Require both an explicit item ID and consistent field identity."""
    item_id = value.get("item_id")
    identities = [value[key] for key in ("field_name", "normalized_field") if key in value]
    if not identities or any(not _normalized(identity) for identity in identities):
        return None
    matches = [
        index for index, task in enumerate(tasks)
        if item_id == f"item_{index}"
        and all(_normalized(identity) == _normalized(task.get("field_name"))
                for identity in identities)
    ]
    return matches[0] if len(matches) == 1 else None


def _decoded_value(value: Any, path: str, notes: list[dict[str, Any]]) -> Any:
    if isinstance(value, _ObjectPairs):
        result: dict[str, Any] = {}
        result_groups = 0
        for key, child in value:
            decoded = _decoded_value(child, f"{path}.{key}", notes)
            if key not in result:
                result[key] = decoded
                if path == "$" and key == "results":
                    result_groups += 1
                continue
            if path != "$" or key != "results":
                raise ValueError(f"缩写响应存在重复JSON键：{path}.{key}。")
            if not isinstance(result[key], list) or not isinstance(decoded, list):
                raise ValueError("缩写响应重复的根results必须全部为数组。")
            result[key].extend(decoded)
            result_groups += 1
        if result_groups > 1:
            notes.append({"action": "merge_duplicate_root_results", "path": "$.results",
                          "array_count": result_groups, "result_count": len(result["results"])})
        return result
    if isinstance(value, list):
        return [_decoded_value(child, f"{path}[{index}]", notes)
                for index, child in enumerate(value)]
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"缩写响应含有非法JSON常量：{value}。")


def _restore_result_tail_segments(
    parsed: dict[str, Any],
    values: list[Any],
    tasks: Sequence[dict[str, Any]],
    notes: list[dict[str, Any]],
) -> None:
    """Move a task's own misplaced final segment only when it completes its name.

    Neither sibling records nor root-level values establish segment ownership.
    Preserve their values unchanged and let the caller validate meaning content.
    """
    for index, record in enumerate(values):
        if not isinstance(record, dict):
            continue
        task_index = _matching_task(record, tasks)
        if task_index is None:
            continue
        item_id = record["item_id"]
        if (parsed.get("item_id") == item_id
                or sum(isinstance(value, dict) and value.get("item_id") == item_id
                       for value in values) != 1):
            continue
        segments = record.get("segments")
        token = record.get("token")
        meanings = record.get("meanings")
        if (not isinstance(segments, list) or not segments
                or not isinstance(token, str) or not _normalized(token)
                or not isinstance(meanings, list) or not meanings
                or not all(isinstance(meaning, dict) for meaning in meanings)):
            continue
        if not all(isinstance(segment, dict)
                   and isinstance(segment.get("token"), str)
                   and bool(_normalized(segment["token"]))
                   and isinstance(segment.get("meanings"), list)
                   and bool(segment["meanings"])
                   and all(isinstance(meaning, dict) for meaning in segment["meanings"])
                   for segment in segments):
            continue
        reconstructed = "".join(_normalized(segment["token"]) for segment in segments)
        if reconstructed + _normalized(token) != _normalized(tasks[task_index]["field_name"]):
            continue
        segments.append({"token": record.pop("token"), "meanings": record.pop("meanings")})
        notes.append({"action": "move_result_tail_segment", "path": f"$.results[{index}].segments",
                      "item_id": item_id, "task_index": task_index})


def parse_abbreviation_response(
    raw: str,
    tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse JSON and return conservative structural repairs for audit logging.

    Malformed or truncated JSON and ambiguous duplicate object keys are rejected.
    Duplicated root ``results`` arrays are concatenated without deduplicating IDs,
    so the caller can reject conflicting records instead of silently picking one.
    """
    if not isinstance(raw, str):
        raise ValueError("缩写响应正文不是文本。")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text,
                          flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    # Parse the entire body: extracting a valid prefix could hide truncation or
    # a second malformed object. Never insert missing braces or commas.
    pairs = json.loads(text, object_pairs_hook=_ObjectPairs, parse_constant=_reject_constant)
    if not isinstance(pairs, _ObjectPairs):
        raise ValueError("缩写响应JSON必须是对象。")
    notes: list[dict[str, Any]] = []
    parsed = _decoded_value(pairs, "$", notes)
    values = parsed.get("results")
    if values is None and "results" not in parsed:
        values = []
    if not isinstance(values, list):
        return parsed, notes  # Existing validation reports the incorrect type.

    # A generated result can be accidentally emitted alongside ``results``.
    # Move it only if its own ID and field identify an input and all required
    # containers already exist. No ID, segment, or meaning is invented.
    root_index = _matching_task(parsed, tasks)
    root_complete = (root_index is not None
                     and isinstance(parsed.get("segments"), list)
                     and bool(parsed.get("segments"))
                     and isinstance(parsed.get("field_meanings"), list)
                     and bool(parsed.get("field_meanings")))
    if root_complete:
        if any(isinstance(value, dict) and value.get("item_id") == parsed["item_id"]
               for value in values):
            raise ValueError("缩写响应根记录与results中的item_id重复，无法安全归属。")
        keys = ("item_id", "field_name", "normalized_field", "segments", "field_meanings")
        record = {key: parsed.pop(key) for key in keys if key in parsed}
        values.append(record)
        parsed["results"] = values
        notes.append({"action": "move_root_record_to_results", "path": "$.results",
                      "item_id": record["item_id"], "task_index": root_index})

    # A root meaning list is assignable only for a single requested task and one
    # clearly identified result. Other result-array objects (e.g. stray segments)
    # remain untouched, allowing the semantic validator to discard unsafe splits.
    if (len(tasks) == 1 and "field_meanings" in parsed
            and not any(key in parsed for key in ("item_id", "field_name", "normalized_field"))
            and isinstance(parsed["field_meanings"], list) and parsed["field_meanings"]):
        records = [value for value in values if isinstance(value, dict)
                   and any(key in value for key in ("item_id", "field_name", "normalized_field"))]
        if (len(records) == 1 and _matching_task(records[0], tasks) == 0
                and "field_meanings" not in records[0]):
            record = records[0]
            record["field_meanings"] = parsed.pop("field_meanings")
            notes.append({"action": "move_root_field_meanings_to_result",
                          "path": f"$.results[{values.index(record)}].field_meanings",
                          "item_id": record["item_id"], "task_index": 0})
    _restore_result_tail_segments(parsed, values, tasks, notes)
    return parsed, notes
