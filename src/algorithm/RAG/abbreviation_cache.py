"""Persistent abbreviation caches independent of experiment fingerprints."""
from __future__ import annotations

import os

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Optional

from abbreviation_generator import (
    AbbreviationStore, FieldExpansion, Meaning, Segment, PROMPT_VERSION, normalize_field,
)
from domain_config import resolve_domain

DEFAULT_ABBREVIATION_CACHE = Path(os.environ.get("RAG_WORKSPACE_ROOT", "/Users/andiandian/Desktop/trandatacls")).expanduser().resolve() / "transclass_repo" / 'data/cache/abbreviations'


def cache_path(root: Path, domain: str, model: str, prompt_version: str = PROMPT_VERSION) -> Path:
    scope = json.dumps({'model': model, 'prompt_version': prompt_version}, sort_keys=True)
    namespace = hashlib.sha256(scope.encode()).hexdigest()[:20]
    return root / resolve_domain(domain).key / namespace / 'abbreviations.sqlite3'


def discover_legacy_caches(roots):
    paths = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root.exists():
            paths.update(path.resolve() for path in root.rglob('abbreviations.sqlite3'))
    return sorted(paths, key=lambda path: (-path.stat().st_mtime_ns, str(path)))


def _read_expansion(row) -> FieldExpansion:
    def meanings(values):
        if not isinstance(values, list) or not values:
            raise ValueError('含义列表为空')
        result = []
        for value in values:
            english, chinese = value['english'], value['chinese']
            confidence = float(value['confidence'])
            if not isinstance(english, str) or not english.strip() or not isinstance(chinese, str) or not chinese.strip():
                raise ValueError('含义为空')
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('置信度不合法')
            result.append(Meaning(english, chinese, confidence))
        return tuple(result)
    normalized = normalize_field(row['original_field'])
    segments = tuple(Segment(value['token'], meanings(value['meanings']))
                     for value in json.loads(row['segments_json']))
    if not normalized or normalized != row['normalized_field'] or ''.join(item.token for item in segments) != normalized:
        raise ValueError('拆分不能还原字段')
    if row['source'] not in ('llm', 'llm_unsplit', 'cache_compose'):
        raise ValueError('不迁移未验证的规则结果')
    return FieldExpansion(row['original_field'], normalized, segments,
                          meanings(json.loads(row['meanings_json'])), row['source'],
                          row['model'], row['prompt_version'])


def migrate_caches(root: Path, sources, scopes: Optional[set[tuple[str, str, str]]] = None):
    """Merge validated records without deleting originals or overwriting shared entries.

    Source connections are query-only. SQLite may create WAL sidecars but no source
    records are changed; live committed WAL records are visible. Repeated imports
    are idempotent, and newly completed legacy batches are picked up next run.
    """
    report = {'imported': 0, 'existing': 0, 'invalid': 0, 'skipped_scope': 0,
              'sources': [], 'destinations': {}, 'errors': []}
    stores = {}
    try:
        for source in sources:
            source = Path(source).resolve()
            counts = {'path': str(source), 'imported': 0, 'existing': 0, 'invalid': 0}
            connection = None
            try:
                if not source.is_file():
                    raise FileNotFoundError(source)
                connection = sqlite3.connect(str(source), timeout=30)
                connection.execute('PRAGMA query_only=ON')
                connection.row_factory = sqlite3.Row
                rows = connection.execute('SELECT * FROM field_expansions ORDER BY created_at DESC').fetchall()
                for row in rows:
                    domain = resolve_domain(row['domain']).key
                    scope = (domain, row['model'], row['prompt_version'])
                    if scopes is not None and scope not in scopes:
                        report['skipped_scope'] += 1
                        continue
                    destination = cache_path(root, *scope)
                    if source == destination.resolve():
                        continue
                    try:
                        expansion = _read_expansion(row)
                    except (ValueError, TypeError, KeyError):
                        counts['invalid'] += 1
                        report['invalid'] += 1
                        continue
                    if scope not in stores:
                        stores[scope] = AbbreviationStore(destination, domain, row['model'], row['prompt_version'])
                    store = stores[scope]
                    if store.get_field(expansion.field_name, expansion.model) is not None:
                        counts['existing'] += 1
                        report['existing'] += 1
                        continue
                    store.save_field(expansion)
                    counts['imported'] += 1
                    report['imported'] += 1
                    report['destinations'][str(destination.resolve())] = {
                        'domain': domain, 'model': expansion.model, 'prompt_version': expansion.prompt_version,
                    }
            except (OSError, sqlite3.Error, ValueError) as exc:
                report['errors'].append({'path': str(source), 'error': str(exc)})
            finally:
                if connection is not None:
                    connection.close()
            report['sources'].append(counts)
    finally:
        for store in stores.values():
            store.close()
    return report


def open_shared_store(root: Path, domain: str, model: str, legacy_roots):
    scope = (resolve_domain(domain).key, model, PROMPT_VERSION)
    migration = migrate_caches(root, discover_legacy_caches(legacy_roots), {scope})
    if migration['errors']:
        raise RuntimeError('旧词典迁移失败，避免重复付费请求：' + json.dumps(migration['errors'], ensure_ascii=False))
    path = cache_path(root, *scope)
    return AbbreviationStore(path, domain, model, PROMPT_VERSION), migration


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='合并历史词典，保留各领域、模型和提示词版本')
    parser.add_argument('legacy_roots', nargs='+', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_ABBREVIATION_CACHE)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = migrate_caches(args.cache_dir, discover_legacy_caches(args.legacy_roots))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report['errors'] else 0)
