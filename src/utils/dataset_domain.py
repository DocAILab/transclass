"""Dataset-domain descriptions used in prompts; no record matching is involved."""
from __future__ import annotations

from pathlib import Path


DOMAIN_DESCRIPTIONS = {
    "finance": "金融业务数据，涉及账户、交易、客户、合约、信托与系统配置等信息。",
    "shougang": "钢铁制造与生产经营数据，涉及炼钢、热轧、冷轧、库存、物流、计划、设备与经营管理等信息。",
    "infra": "工业基础设施与检测管理数据，涉及监测、检测、生产设计及相关管理信息。",
    "education": "教育机构人员与学生信息数据，涉及学籍、教职工、课程、科研与人力资源等信息。",
}
GENERIC_DOMAIN_DESCRIPTION = "业务数据分类任务；请根据字段及其表级上下文理解具体业务语义。"


def resolve_domain_description(data_dir: Path, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    return DOMAIN_DESCRIPTIONS.get(data_dir.name.strip().lower(), GENERIC_DOMAIN_DESCRIPTION)
