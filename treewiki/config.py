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
        "embedding_model": "BAAI/bge-m3",
        "embedding_api_key": "",
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
        "summary_workers": 1,
    },
    "wiki": {
        "enabled": True,
        "leaf_chars": 3000,
        "batch_chars": 9000,
        "max_batches": 30,
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
        "wiki_max_pages": 8,
        "wiki_page_chars": 3000,
    },
    "entities": {
        "enabled": True,
        "min_frequency": 2,
        "similarity_threshold": 0.5,
        "context_window": 60,
        "max_block_size": 200,
        "extract_ngrams": True,
        "ngram_max": 6,
        "ngram_min_frequency": 3,
        "ngram_min_entropy": 0.5,
        "query_expansion": True,
        "expand_limit": 6,
    },
    "neo4j": {
        "uri": "bolt://127.0.0.1:7687",
        "user": "neo4j",
        "password": "",
        "database": "neo4j",
        "embedding_model": "",
        "embedding_api_key": "",
        "embedding_batch": 16,
        "embedding_timeout": 180,
        "vector_k": 40,
        "fulltext_k": 40,
        "graph_k": 40,
        "graph_entity_k": 12,
        "rrf_k": 60,
        "top_k": 30,
        "entity_df_cap": 200,
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
    if not llm.get("embedding_api_key"):
        llm["embedding_api_key"] = (
            os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_EMBEDDING_API_KEY")
            or ""
        )
    if not llm.get("embedding_model") and os.environ.get("EMBEDDING_MODEL"):
        llm["embedding_model"] = os.environ["EMBEDDING_MODEL"]
    mineru = cfg["mineru"]
    if not mineru.get("token") and os.environ.get("MINERU_API_TOKEN"):
        mineru["token"] = os.environ["MINERU_API_TOKEN"]
    neo4j = cfg["neo4j"]
    for key, env in (
        ("uri", "NEO4J_URI"),
        ("user", "NEO4J_USER"),
        ("password", "NEO4J_PASSWORD"),
        ("database", "NEO4J_DATABASE"),
        ("embedding_model", "EMBEDDING_MODEL"),
        ("embedding_api_key", "EMBEDDING_API_KEY"),
    ):
        if os.environ.get(env):
            neo4j[key] = os.environ[env]


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
