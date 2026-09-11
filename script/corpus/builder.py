"""Build retrieval corpus JSON from short classification guide documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from script.preprocessing.processor import normalize_label


DEFAULT_LABEL_COLUMN = "四级子类"
DEFAULT_CONTENT_COLUMN = "内容"
STANDARD_FIELDS = (
    "id",
    "data_description_and_example",
    "dataset",
    "category_leaf_level",
    "category_subbranch_level",
    "category_branch_level",
    "category_root_level",
    "sensitivity_level",
    "reference_standard",
)


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Corpus construction requires pandas. XLSX input also requires openpyxl."
        ) from exc
    return pd


def _clean_text(value: Any) -> str:
    pd = _load_pandas()
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def _read_table(path: str | Path):
    pd = _load_pandas()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Corpus source not found: {source}")

    if source.suffix.lower() == ".xlsx":
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Corpus source must be an .xlsx or .csv file")

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise ValueError("Duplicate source columns: " + ", ".join(duplicates))
    return source, frame


def _load_processed_labels(path: str | Path | None) -> tuple[Path | None, set[str]]:
    if path is None:
        return None, set()
    processed = Path(path).expanduser().resolve()
    if not processed.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {processed}")
    try:
        with processed.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid processed JSON: {processed}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Processed JSON must contain a list")

    labels = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Processed item {index} must be an object")
        classification = item.get("classification", {})
        if not isinstance(classification, dict):
            raise ValueError(f"Processed item {index} classification must be an object")
        label = _clean_text(classification.get("category_leaf_level"))
        if label:
            labels.add(label)
    return processed, labels


def _load_processed_metadata(
    path: str | Path | None,
) -> tuple[Path | None, dict[str, dict[str, set[str]]]]:
    if path is None:
        return None, {}
    processed = Path(path).expanduser().resolve()
    if not processed.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {processed}")
    try:
        with processed.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid processed JSON: {processed}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Processed JSON must contain a list")

    metadata: dict[str, dict[str, set[str]]] = {}
    fields = (
        "category_root_level",
        "category_branch_level",
        "category_subbranch_level",
        "sensitivity_level",
    )
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Processed item {index} must be an object")
        classification = item.get("classification", {})
        grading = item.get("grading", {})
        if not isinstance(classification, dict) or not isinstance(grading, dict):
            raise ValueError(f"Processed item {index} has invalid classification/grading")
        leaf = normalize_label(_clean_text(classification.get("category_leaf_level")))
        if not leaf:
            continue
        values = metadata.setdefault(leaf, {field: set() for field in fields})
        for field in fields[:-1]:
            value = _clean_text(classification.get(field))
            if value:
                values[field].add(value)
        sensitivity = _clean_text(grading.get("sensitivity_level"))
        if sensitivity:
            values["sensitivity_level"].add(sensitivity)
    return processed, metadata


def _single_processed_value(
    metadata: dict[str, dict[str, set[str]]], leaf: str, field: str
) -> str:
    values = metadata.get(leaf, {}).get(field, set())
    return next(iter(values)) if len(values) == 1 else ""


def _display_path(path: Path, project_root: Path | None) -> str:
    if project_root is not None:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            pass
    return path.name


def _document_id(dataset: str, label: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{dataset}\x1f{label}\x1f{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{dataset}-{digest}"


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_bundle(
    output_dir: Path,
    corpus: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    outputs = (output_dir / "corpus.json", output_dir / "build_report.json")
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite to replace it."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging.", dir=output_dir.parent
    ) as staging_name:
        staging = Path(staging_name)
        staged = (staging / "corpus.json", staging / "build_report.json")
        _write_json(staged[0], corpus)
        _write_json(staged[1], report)

        output_dir.mkdir(parents=True, exist_ok=True)
        for source, destination in zip(staged, outputs):
            os.replace(source, destination)


def build_corpus(
    dataset: str,
    input_file: str | Path,
    output_dir: str | Path,
    *,
    processed_file: str | Path | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    content_column: str = DEFAULT_CONTENT_COLUMN,
    missing_policy: str = "skip",
    overwrite: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build corpus.json and build_report.json from a guide table."""
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'")

    source, frame = _read_table(input_file)
    label_column = label_column.strip()
    content_column = content_column.strip()
    missing_columns = [
        column for column in (label_column, content_column)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Corpus source is missing column(s): " + ", ".join(missing_columns)
        )

    root = Path(project_root).expanduser().resolve() if project_root else None
    processed, processed_labels = _load_processed_labels(processed_file)
    documents: list[dict[str, Any]] = []
    seen_pairs: dict[tuple[str, str], int] = {}
    skipped_missing: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    normalized_label_rows: list[int] = []

    for index, row in frame.iterrows():
        source_row = int(index) + 2
        raw_label = _clean_text(row[label_column])
        content = _clean_text(row[content_column])
        if not raw_label or not content:
            missing = []
            if not raw_label:
                missing.append(label_column)
            if not content:
                missing.append(content_column)
            detail = {"source_row": source_row, "missing_columns": missing}
            if missing_policy == "error":
                raise ValueError(
                    f"Missing {', '.join(missing)} at source row {source_row}"
                )
            skipped_missing.append(detail)
            continue

        label = normalize_label(raw_label)
        if label != raw_label:
            normalized_label_rows.append(source_row)
        pair = (label, content)
        if pair in seen_pairs:
            duplicate_rows.append({
                "source_row": source_row,
                "duplicate_of_source_row": seen_pairs[pair],
            })
            continue
        seen_pairs[pair] = source_row

        metadata: dict[str, Any] = {
            "dataset": dataset,
            "level_4": label,
            "source": _display_path(source, root),
            "source_row": source_row,
        }
        if label != raw_label:
            metadata["source_level_4"] = raw_label
        documents.append({
            "id": _document_id(dataset, label, content),
            "text": content,
            "metadata": metadata,
        })

    if skipped_missing:
        warnings.warn(
            f"Skipping {len(skipped_missing)} corpus row(s) with missing label/content",
            UserWarning,
            stacklevel=2,
        )
    if not documents:
        raise ValueError("No valid corpus documents were generated")

    ids = [document["id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Corpus document ID collision detected")

    corpus_labels = {document["metadata"]["level_4"] for document in documents}
    label_counts = Counter(
        document["metadata"]["level_4"] for document in documents
    )
    content_lengths = [len(document["text"]) for document in documents]
    report: dict[str, Any] = {
        "dataset": dataset,
        "source": _display_path(source, root),
        "processed_source": (
            _display_path(processed, root) if processed is not None else None
        ),
        "input_rows": len(frame),
        "exported_documents": len(documents),
        "skipped_missing": {
            "count": len(skipped_missing),
            "rows": skipped_missing,
        },
        "skipped_exact_duplicates": {
            "count": len(duplicate_rows),
            "rows": duplicate_rows,
        },
        "normalized_labels": {
            "count": len(normalized_label_rows),
            "source_rows": normalized_label_rows,
        },
        "corpus_statistics": {
            "unique_level_4": len(corpus_labels),
            "labels_with_multiple_documents": sum(
                count > 1 for count in label_counts.values()
            ),
            "content_length": {
                "min": min(content_lengths),
                "max": max(content_lengths),
                "average": round(sum(content_lengths) / len(content_lengths), 2),
            },
        },
        "processed_label_coverage": {
            "available": processed is not None,
            "processed_unique_level_4": len(processed_labels),
            "overlap": len(corpus_labels & processed_labels),
            "processed_labels_without_corpus": sorted(
                processed_labels - corpus_labels
            ),
            "corpus_labels_unused_by_processed": sorted(
                corpus_labels - processed_labels
            ),
        },
    }

    _write_bundle(Path(output_dir), documents, report, overwrite=overwrite)
    return report


def build_standard_corpus(
    dataset: str,
    input_file: str | Path,
    output_file: str | Path,
    *,
    processed_file: str | Path | None = None,
    content_column: str,
    leaf_column: str,
    root_column: str | None = None,
    branch_column: str | None = None,
    subbranch_column: str | None = None,
    sensitivity_column: str | None = None,
    reference_standard: str = "",
    missing_policy: str = "skip",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a classification guide table to the standard flat corpus JSON."""
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'")
    source, frame = _read_table(input_file)
    column_map = {
        "data_description_and_example": content_column,
        "category_leaf_level": leaf_column,
        "category_root_level": root_column,
        "category_branch_level": branch_column,
        "category_subbranch_level": subbranch_column,
        "sensitivity_level": sensitivity_column,
    }
    required = [column for column in (content_column, leaf_column) if column not in frame]
    optional_missing = [
        column for column in column_map.values()
        if column is not None and column not in frame
    ]
    if required or optional_missing:
        raise ValueError(
            "Corpus source is missing column(s): "
            + ", ".join(sorted(set(required + optional_missing)))
        )

    processed, processed_metadata = _load_processed_metadata(processed_file)
    records: list[dict[str, str]] = []
    skipped_rows: list[int] = []
    duplicate_rows: list[dict[str, int]] = []
    ambiguous_fields: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}
    hierarchy_fields = (
        "category_root_level",
        "category_branch_level",
        "category_subbranch_level",
        "sensitivity_level",
    )

    for index, row in frame.iterrows():
        source_row = int(index) + 2
        content = _clean_text(row[content_column])
        leaf = normalize_label(_clean_text(row[leaf_column]))
        if not content or not leaf:
            if missing_policy == "error":
                raise ValueError(f"Missing content or leaf label at source row {source_row}")
            skipped_rows.append(source_row)
            continue
        pair = (leaf, content)
        if pair in seen:
            duplicate_rows.append({
                "source_row": source_row,
                "duplicate_of_source_row": seen[pair],
            })
            continue
        seen[pair] = source_row

        values: dict[str, str] = {}
        for field in hierarchy_fields:
            column = column_map[field]
            value = _clean_text(row[column]) if column else ""
            if not value:
                candidates = processed_metadata.get(leaf, {}).get(field, set())
                if len(candidates) == 1:
                    value = next(iter(candidates))
                elif len(candidates) > 1:
                    ambiguous_fields.append({
                        "source_row": source_row,
                        "field": field,
                        "values": sorted(candidates),
                    })
            values[field] = value

        records.append({
            "id": _document_id(dataset.lower(), leaf, content),
            "data_description_and_example": content,
            "dataset": dataset,
            "category_leaf_level": leaf,
            "category_subbranch_level": values["category_subbranch_level"],
            "category_branch_level": values["category_branch_level"],
            "category_root_level": values["category_root_level"],
            "sensitivity_level": values["sensitivity_level"],
            "reference_standard": reference_standard,
        })

    if not records:
        raise ValueError("No valid corpus documents were generated")
    destination = Path(output_file).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {destination}. Pass --overwrite."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
    ) as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, destination)
    return {
        "dataset": dataset,
        "source": str(source),
        "processed_source": str(processed) if processed else None,
        "input_rows": len(frame),
        "exported_documents": len(records),
        "skipped_rows": skipped_rows,
        "duplicate_rows": duplicate_rows,
        "ambiguous_processed_fields": ambiguous_fields,
        "output_file": str(destination),
    }


def normalize_standard_corpus(
    input_file: str | Path,
    output_file: str | Path,
    *,
    dataset: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize an existing flat corpus JSON and set its dataset identifier."""
    source = Path(input_file).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Corpus JSON must contain a list of objects")
    records = []
    for index, item in enumerate(data):
        missing = [field for field in STANDARD_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Corpus item {index} is missing: {', '.join(missing)}")
        record = {field: _clean_text(item.get(field)) for field in STANDARD_FIELDS}
        record["dataset"] = dataset
        records.append(record)
    destination = Path(output_file).expanduser().resolve()
    if destination.exists() and destination != source and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {destination}. Pass --overwrite."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, records)
    return {"input_rows": len(data), "exported_documents": len(records), "output_file": str(destination)}
