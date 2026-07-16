"""Deterministic random or group-aware dataset splitting."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_FIELDS = (
    "classification.level_1",
    "classification.level_2",
    "classification.level_3",
    "classification.level_4",
    "data_level",
)
MISSING = object()


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    try:
        with source.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid input JSON: {source}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list")
    if not data:
        raise ValueError("Input dataset is empty")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every dataset item must be a JSON object")
    return data


def _atomic_write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _get_nested(item: dict[str, Any], key: str) -> Any:
    value: Any = item
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def _label_value(item: dict[str, Any], field: str) -> str:
    value = _get_nested(item, field)
    return "" if value is MISSING or value is None else str(value)


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    if total and ratios[0] and not counts[0]:
        donor = max(
            (index for index in (1, 2) if counts[index]),
            key=counts.__getitem__,
            default=None,
        )
        if donor is not None:
            counts[donor] -= 1
            counts[0] += 1
    return counts


def _split_random(
    data: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    rng = random.Random(seed)
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        strata[tuple(_label_value(item, field) for field in LABEL_FIELDS)].append(item)

    splits: list[list[dict[str, Any]]] = [[], [], []]
    for items in strata.values():
        items = items.copy()
        rng.shuffle(items)
        counts = _allocate_counts(len(items), ratios)
        start = 0
        for index, count in enumerate(counts):
            splits[index].extend(items[start:start + count])
            start += count
    for split in splits:
        rng.shuffle(split)

    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(data) >= len(active):
        for empty_index in (index for index in active if not splits[index]):
            donor = max(
                (
                    index for index in active
                    if len(splits[index]) > 1
                ),
                key=lambda index: (
                    len(splits[index]) - len(data) * ratios[index],
                    len(splits[index]),
                ),
                default=None,
            )
            if donor is None:
                break
            splits[empty_index].append(splits[donor].pop())
    return tuple(splits)


def _split_group(
    data: list[dict[str, Any]],
    group_key: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], ...]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    missing: list[int] = []
    for index, item in enumerate(data):
        key = _get_nested(item, group_key)
        if key is MISSING:
            missing.append(index)
            continue
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError(
                f"group key '{group_key}' must resolve to a scalar at item {index}"
            ) from exc
        groups.setdefault(key, []).append(item)
    if missing:
        examples = ", ".join(map(str, missing[:10]))
        raise ValueError(f"group key '{group_key}' is missing at item(s): {examples}")

    targets = [len(data) * ratio for ratio in ratios]
    splits: list[list[dict[str, Any]]] = [[], [], []]
    sizes = [0, 0, 0]
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(groups) < len(active):
        raise ValueError(
            f"Only {len(groups)} groups are available for {len(active)} non-empty splits"
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda pair: len(pair[1]), reverse=True)
    for position, (_, items) in enumerate(group_items):
        if position < len(active):
            target_index = active[position]
        else:
            target_index = min(
                active,
                key=lambda candidate: (
                    (sizes[candidate] + len(items) - targets[candidate]) ** 2
                    / max(targets[candidate], 1),
                    sizes[candidate] / max(targets[candidate], 1),
                    candidate,
                ),
            )
        splits[target_index].extend(items)
        sizes[target_index] += len(items)
    return tuple(splits)


def _build_report(splits: tuple[list[dict[str, Any]], ...]) -> dict[str, Any]:
    names = ("train", "val", "test")
    report: dict[str, Any] = {
        "sizes": dict(zip(names, map(len, splits))),
        "distributions": {},
        "warnings": [],
    }
    for field in LABEL_FIELDS:
        distributions = {}
        label_sets = []
        for name, split in zip(names, splits):
            counter = Counter(_label_value(item, field) or "<EMPTY>" for item in split)
            distributions[name] = dict(sorted(counter.items()))
            label_sets.append(set(counter) - {"<EMPTY>"})
        report["distributions"][field] = distributions
        for warning_type, labels in (
            ("labels_missing_from_train", (label_sets[1] | label_sets[2]) - label_sets[0]),
            ("train_labels_missing_from_val", label_sets[0] - label_sets[1]),
            ("train_labels_missing_from_test", label_sets[0] - label_sets[2]),
        ):
            if labels:
                report["warnings"].append({
                    "type": warning_type, "field": field, "labels": sorted(labels)
                })
    return report


def split_dataset(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    split_type: str = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    group_key: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split a normalized JSON list and return the generated report."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Every split ratio must be between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, val, and test ratios must sum to 1")
    if train_ratio == 0:
        raise ValueError("train ratio must be greater than zero")

    data = _load_json_list(input_file)
    if split_type == "random":
        splits = _split_random(data, ratios, random_seed)
    elif split_type == "group":
        if not group_key:
            raise ValueError("group_key is required for a group split")
        splits = _split_group(data, group_key, ratios, random_seed)
    else:
        raise ValueError("split_type must be 'random' or 'group'")

    destination = Path(output_dir).expanduser().resolve()
    output_paths = [destination / name for name in (
        "train.json", "val.json", "test.json", "split_report.json"
    )]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace it."
        )

    report = _build_report(splits)
    # Validate all outputs before replacing any existing file.
    destination.mkdir(parents=True, exist_ok=True)
    for split, path in zip(splits, output_paths[:3]):
        _atomic_write_json(split, path)
    _atomic_write_json(report, output_paths[3])
    return report
