from __future__ import annotations

from typing import Callable

from ..config import load_config
from ..core.entities import EntityIndex, Variant, add_mention, align_variants
from ..core.mentions import NgramMiner, rule_spans
from ..core.store import KB


def build_entities(
    kb: KB,
    cfg: dict | None = None,
    stage: Callable[[str], None] = print,
    doc_ids: list[str] | None = None,
) -> EntityIndex:
    """扫描所有来源页 → 抽取提及 → 变体图聚类 → 写入 index/entities.json。"""
    cfg = cfg or load_config(kb.home)
    ecfg = cfg.get("entities", {})
    if not ecfg.get("enabled", True):
        raise RuntimeError("实体索引已禁用（entities.enabled=false）")

    sources = kb.sources()
    docs = [d for d in (doc_ids or sources) if d in sources]
    if not docs:
        stage("没有可用的来源，实体索引为空")
        return kb.load_entities()

    pages_by_doc: dict[str, list[str]] = {}
    rules_by_doc: dict[str, list[list]] = {}
    for doc_id in docs:
        pages = kb.read_pages(doc_id, 1, kb.page_count(doc_id))
        pages_by_doc[doc_id] = pages
        rules_by_doc[doc_id] = [rule_spans(page) for page in pages]
        stage(f"[{doc_id}] 扫描 {len(pages)} 页 …")

    miner: NgramMiner | None = None
    if ecfg.get("extract_ngrams", True):
        miner = NgramMiner(ecfg)
        for pages in pages_by_doc.values():
            for page in pages:
                miner.observe(page)
        selected = miner.finalize()
        stage(f"n-gram 候选 {len(selected)} 个（词频≥{miner.min_freq}，邻接熵≥{miner.min_entropy}）")

    variants: dict[str, Variant] = {}
    context = max(0, int(ecfg.get("context_window", 60)))
    for doc_id, pages in pages_by_doc.items():
        for page_no, (text, base) in enumerate(
            zip(pages, rules_by_doc[doc_id]), start=1
        ):
            spans = miner.extract(text, base) if miner else base
            for span in spans:
                left = max(0, span.start - context)
                right = min(len(text), span.end + context)
                add_mention(
                    variants,
                    span.surface,
                    doc_id,
                    page_no,
                    context=text[left:right],
                    source=span.source,
                )
        stage(f"[{doc_id}] 完成：{len(variants)} 个表面形式")

    index = align_variants(variants, docs, ecfg)
    kb.save_entities(index)
    stats = index.summary()
    stage(
        f"实体索引完成：{stats['entities']} 实体 / {stats['aliases']} 别名 "
        f"/ {stats['mentions']} 提及 → {kb.entities_path}"
    )
    return index
