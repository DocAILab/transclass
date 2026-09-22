"""Per-record domain routing and corpus configuration."""
from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomainConfig:
    key: str
    dataset: str
    chinese: str
    english: str

    def data_paths(self, data_root: Path) -> tuple[Path, Path]:
        directory = data_root / self.dataset
        return (directory / f"{self.dataset}_test.json",
                directory / f"{self.dataset}_Corpus.json")


DOMAINS = {
    "finance": DomainConfig("finance", "DCG_FIN", "金融", "financial"),
    "dcg_education": DomainConfig("dcg_education", "DCG_EDU", "教育", "education"),
    "dcg_vehicle": DomainConfig("dcg_vehicle", "DCG_VEH", "汽车", "automotive"),
}
GENERIC = DomainConfig("generic", "", "通用", "general")


def inference_domain(value, use_domain: bool = True) -> str:
    """Only the three exact record values authorize a specialized route."""
    return value if use_domain and isinstance(value, str) and value in DOMAINS else "generic"


ALIASES = {
    "finance": "finance", "dcg_fin": "finance", "fin": "finance",
    "education": "dcg_education", "dcg_education": "dcg_education",
    "dcg_edu": "dcg_education", "edu": "dcg_education",
    "vehicle": "dcg_vehicle", "dcg_vehicle": "dcg_vehicle",
    "dcg_veh": "dcg_vehicle", "veh": "dcg_vehicle",
}


def resolve_domain(value: str) -> DomainConfig:
    if value == "generic":
        return GENERIC
    key = ALIASES.get(value.strip().lower())
    if key is None:
        raise ValueError(f"不支持的领域 {value!r}；请选择 finance、education 或 vehicle")
    return DOMAINS[key]


def validate_known_domain(examples, corpus, domain: str) -> None:
    """Validate declarations only, without deriving a domain from gold labels.

    Legacy corpora without domain/dataset declarations use the explicitly
    selected domain. DCG declarations, when present, must all agree.
    """
    expected = resolve_domain(domain).key
    for example in examples:
        if example.domain and resolve_domain(example.domain).key != expected:
            raise ValueError(f"测试样本 {example.source_id} 的领域 {example.domain} 与指定领域 {expected} 不一致")
    if isinstance(corpus, list):
        for index, record in enumerate(corpus, 1):
            if not isinstance(record, dict):
                continue  # The corpus reader reports structural errors.
            for key in ("domain", "dataset"):
                value = record.get(key)
                if value and resolve_domain(str(value)).key != expected:
                    raise ValueError(f"语料第 {index} 条 {key}={value} 与指定领域 {expected} 不一致")


# All default resources belong to the directory containing these delivery files.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models" / "BAAI_bge-m3"


def require_local_model(path: Path) -> Path:
    """Fail before paid API calls rather than silently download another model."""
    path = Path(path).expanduser().resolve()
    required = ["config.json", "modules.json", "1_Pooling/config.json",
                "tokenizer_config.json", "sentencepiece.bpe.model"]
    missing = [name for name in required if not (path / name).is_file()]
    if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append("model.safetensors 或 pytorch_model.bin")
    if missing:
        raise ValueError(f"本地 BGE-M3 不完整：{path}；缺少 {', '.join(missing)}。"
                         "请先执行 bash run_rag.sh --download-model，或设置 --model-dir。")
    return path


def model_identity(path) -> str:
    """Content-based identity keeps index reuse independent of install location."""
    import hashlib
    path = Path(path).expanduser()
    if not path.is_dir():
        # Compatibility for helper APIs accepting explicit remote model names.
        return str(path)
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != "resource_manifest.json" and ".cache" not in p.relative_to(path).parts
                   and p.suffix in (".json", ".model", ".bin", ".safetensors", ".pt"))
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
