"""Dataset validation, reproducible run identities, and offline reports."""
from __future__ import annotations

import os

import hashlib
import json
from collections import Counter
from pathlib import Path

from vector_index import canonical_label, clean_text


def prediction_metrics(rows, prediction_key):
    labeled = [row for row in rows if clean_text(row.get("gold_label"))]
    pairs = [(canonical_label(row["gold_label"]),
              canonical_label(row.get(prediction_key))) for row in labeled]
    labels = sorted({value for pair in pairs for value in pair if value})
    per_class = {}
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        support = sum(gold == label for gold, _ in pairs)
        predicted = sum(pred == label for _, pred in pairs)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"support": support, "precision": precision,
                            "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall)
                            if precision + recall else 0.0}
    failed = sum(not pred for _, pred in pairs)
    return {
        "evaluated_rows": len(pairs),
        "predicted_rows": len(pairs) - failed,
        "failed_rows": failed,
        "failure_rate": failed / len(pairs) if pairs else None,
        "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(labels) if labels else None,
        "macro_f1_label_policy": "union_of_gold_and_nonempty_predictions",
        "per_class": per_class,
    }


def grouped_metrics(rows, prediction_key):
    result = {}
    for key in ("domain", "inference_domain", "label_status"):
        result[key] = {
            value: prediction_metrics([row for row in rows if row.get(key) == value], prediction_key)
            for value in sorted({row.get(key, "") for row in rows} - {"", None})
        }
    return result


def candidate_counts(rows, key):
    counts = [len(row.get(key, [])) for row in rows]
    return {"min": min(counts, default=0), "max": max(counts, default=0),
            "mean": sum(counts) / len(counts) if counts else 0,
            "distribution": dict(sorted(Counter(counts).items()))}


def dataset_audit(examples, standards):
    categories = {canonical_label(item.category) for item in standards}
    missing = sorted({item.gold_label for item in examples
                      if canonical_label(item.gold_label) not in categories})
    if missing:
        raise ValueError(f"测试标签不在语料库中（规范化后）：{missing}")
    return {"rows": len(examples), "original_documents": len(standards),
            "unique_categories": len(categories),
            "test_categories": len({canonical_label(item.gold_label) for item in examples}),
            "domains": dict(Counter(item.domain for item in examples)),
            "label_status": dict(Counter(item.label_status for item in examples)),
            "input_mode": "field_name_only", "missing_labels": missing}


def run_manifest(args, input_path, corpus_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    excluded = {"output_dir", "cache_dir", "skip_rerank", "validate_only", "input_json",
                "input_xlsx", "corpus", "dataset_id", "abbreviation_cache_dir", "build_index"}
    parameters = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key not in excluded}
    source_dir = Path(__file__).resolve().parent
    sources = {p.name: digest(p) for p in sorted(source_dir.glob("*.py"))
               if not p.name.startswith("test_")}
    corpus_paths = [corpus_path] if isinstance(corpus_path, Path) else list(corpus_path)
    identity = {"version": 2, "input_sha256": digest(input_path),
                "corpus_sha256": [digest(path) for path in corpus_paths], "input_mode": "field_name_only",
                "parameters": parameters, "source_sha256": sources}
    model_dir = getattr(args, "model_dir", None)
    if model_dir is not None and Path(model_dir).expanduser().is_dir():
        from domain_config import model_identity
        identity["embedding_model_sha256"] = model_identity(model_dir)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"fingerprint": fingerprint, "identity": identity,
            "input_path": str(input_path), "corpus_paths": [str(path) for path in corpus_paths]}


def summarize_runs(run_directories):
    """Aggregate explicitly selected runs; never silently select stale results."""
    reports = {}
    for directory in run_directories:
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        dataset_id = summary.get("dataset_id", directory.parent.name)
        if dataset_id in reports:
            raise ValueError(f"同一数据集选择了多次实验：{dataset_id}")
        reports[dataset_id] = summary
    aggregate = {}
    for stage in ("retrieval", "rerank"):
        values = []
        for summary in reports.values():
            report = summary.get(stage)
            if stage == "retrieval" and report:
                report = report.get("classification_report")
            if report and report.get("evaluated_rows", 0):
                values.append(report)
        total = sum(value["evaluated_rows"] for value in values)
        aggregate[stage] = {
            "datasets_evaluated": len(values), "evaluated_rows": total,
            "sample_weighted_accuracy": sum(value["accuracy"] * value["evaluated_rows"] for value in values) / total if total else None,
            "mean_dataset_accuracy": sum(value["accuracy"] for value in values) / len(values) if values else None,
            "mean_dataset_macro_f1": sum(value["macro_f1"] for value in values) / len(values) if values else None,
            "failed_rows": sum(value["failed_rows"] for value in values),
        }
    return {"datasets": reports, "aggregate": aggregate,
            "note": "mean_dataset_macro_f1 为各数据集 Macro-F1 的等权平均，不是混合标签后的 Macro-F1"}


def summarize_cli():
    import argparse
    parser = argparse.ArgumentParser(description="汇总明确指定的实验目录")
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize_runs(args.run_directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


DATA_REVISION = "ce9433b3fef805b864c6e941d721319e942335ce"


def download_resources(kind, destination, revision=None, token=None):
    """Download a pinned snapshot to local resources, validate before publishing.

    Only test/corpus files or the dense SentenceTransformer model are fetched.
    Exceptions are reported by the CLI without including credentials or headers.
    """
    import tempfile
    from datetime import datetime, timezone
    from huggingface_hub import HfApi, hf_hub_download
    from domain_config import DOMAINS, require_local_model, validate_known_domain
    api = HfApi(token=token)
    is_data = kind == "data"
    repo = "DocAILab/DCG" if is_data else "BAAI/bge-m3"
    repo_type = "dataset" if is_data else "model"
    revision = revision or (DATA_REVISION if is_data else "main")
    info = (api.dataset_info(repo, revision=revision) if is_data
            else api.model_info(repo, revision=revision))
    resolved = info.sha
    available = set(api.list_repo_files(repo, repo_type=repo_type, revision=resolved))
    if is_data:
        wanted = [f"{c.dataset}/{c.dataset}_{suffix}.json"
                  for c in DOMAINS.values() for suffix in ("test", "Corpus")]
    else:
        wanted = ["config.json", "modules.json", "1_Pooling/config.json",
                  "tokenizer_config.json", "sentencepiece.bpe.model"]
        weight = next((name for name in ("model.safetensors", "pytorch_model.bin")
                       if name in available), None)
        if weight is None:
            raise ValueError("BGE-M3 仓库没有支持的完整权重文件")
        wanted.append(weight)
        wanted += [name for name in ("tokenizer.json", "special_tokens_map.json",
                   "config_sentence_transformers.json", "sentence_bert_config.json")
                   if name in available]
    absent = set(wanted) - available
    if absent:
        raise ValueError(f"远端资源结构不匹配，缺少：{sorted(absent)}")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "resource_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("repo_id") == repo and previous.get("resolved_revision") == resolved:
            complete = True
            for name in wanted:
                path = destination / name
                if not path.is_file():
                    complete = False
                    break
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != previous.get("sha256", {}).get(name):
                    complete = False
                    break
            if complete:
                print(f"已有资源通过版本及 SHA-256 校验，跳过下载：{destination}")
                return previous
    with tempfile.TemporaryDirectory(prefix=".rag-download-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        for name in wanted:
            print(f"下载 {repo}@{resolved[:12]}: {name}", flush=True)
            hf_hub_download(repo_id=repo, filename=name, repo_type=repo_type,
                            revision=resolved, token=token, local_dir=stage,
                            local_dir_use_symlinks=False)
        if is_data:
            from llm_client import read_test_json, load_corpus
            for domain, config in DOMAINS.items():
                test, corpus = config.data_paths(stage)
                raw = json.loads(corpus.read_text(encoding="utf-8"))
                examples = read_test_json(test)
                validate_known_domain(examples, raw, domain)
                dataset_audit(examples, load_corpus(corpus, domain=domain))
        else:
            require_local_model(stage)
        hashes = {}
        for name in wanted:
            digest = hashlib.sha256()
            with (stage / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
        destination.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
        # A previous weight format must not shadow the newly selected snapshot.
        if not is_data:
            for name in ("model.safetensors", "pytorch_model.bin"):
                if name not in wanted and (destination / name).exists():
                    (destination / name).unlink()
        manifest = {"repo_id": repo, "repo_type": repo_type, "requested_revision": revision,
                    "resolved_revision": resolved, "sha256": hashes,
                    "downloaded_at": datetime.now(timezone.utc).isoformat()}
        path = destination / "resource_manifest.json"
        temp_manifest = stage / "resource_manifest.json"
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_manifest, path)
    print(f"资源已校验并保存：{destination}")
    return manifest


def resource_cli(argv):
    import argparse
    from domain_config import DATA_DIR, MODEL_DIR
    parser = argparse.ArgumentParser(description="下载三领域数据或本地 BGE-M3")
    parser.add_argument("kind", choices=("data", "model"))
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--revision", help="指定 Hugging Face commit/tag/branch；最终 commit 会写入清单")
    args = parser.parse_args(argv)
    try:
        download_resources(args.kind, args.data_root if args.kind == "data" else args.model_dir,
                           args.revision, os.getenv("HF_TOKEN") or None)
    except ImportError as exc:
        raise SystemExit("缺少下载依赖，请先按 readme.md 安装 huggingface-hub==0.24.7") from exc
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or type(exc).__name__ == "GatedRepoError":
            message = "资源访问未获授权；请先在 Hugging Face 数据集网页申请访问并设置 HF_TOKEN。"
        elif isinstance(exc, ValueError) and status is None:
            message = str(exc)
        else:
            message = f"下载失败（{type(exc).__name__}，HTTP {status or '未返回'}）；请检查网络、资源版本及权限后重试。"
        raise SystemExit(message) from None
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("--download-data", "--download-model"):
        resource_cli([sys.argv[1].removeprefix("--download-")] + sys.argv[2:])
    else:
        summarize_cli()
