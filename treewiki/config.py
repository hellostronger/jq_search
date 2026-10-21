from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "llm": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "index_model": "gpt-4o-mini",
        "chat_model": "gpt-4o",
    },
    "mineru": {
        "token": "",
        "model_version": "pipeline",
        "language": "ch",
        "is_ocr": False,
        "enable_formula": True,
    },
    "index": {
        "summaries": True,
        "classic_batch": 8,
        "summary_workers": 6,
    },
    "query": {
        "mode": "cascade",
        "max_rounds": 4,
        "max_pages": 30,
        "page_chars": 3500,
        "locate_retries": 1,
        "answer_attempts": 3,
        "summary_chars": 6000,
        "lexical_pages": 12,
    },
}


def kb_home() -> Path:
    return Path(os.environ.get("TREWIKI_HOME", "kb"))


def config_path(home: Path | None = None) -> Path:
    return (home or kb_home()) / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env(cfg: dict) -> None:
    llm = cfg["llm"]
    if not llm.get("base_url") and os.environ.get("OPENAI_BASE_URL"):
        llm["base_url"] = os.environ["OPENAI_BASE_URL"]
    if not llm.get("api_key") and os.environ.get("OPENAI_API_KEY"):
        llm["api_key"] = os.environ["OPENAI_API_KEY"]
    mineru = cfg["mineru"]
    if not mineru.get("token") and os.environ.get("MINERU_API_TOKEN"):
        mineru["token"] = os.environ["MINERU_API_TOKEN"]


def load_config(home: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = config_path(home)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if isinstance(file_cfg, dict):
            _deep_merge(cfg, file_cfg)
    _apply_env(cfg)
    return cfg


def save_config(cfg: dict, home: Path | None = None) -> Path:
    p = config_path(home)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p
