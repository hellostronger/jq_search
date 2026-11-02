from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ..core.store import KB
from ..pipelines.neo4j_hybrid import Neo4jHybrid, parse_page_id
from ..pipelines.query import (
    _lexical_pages,
    anchor_groups,
    anchor_terms,
    question_anchors,
)

CHANNELS = (
    "vector",
    "fulltext",
    "graph",
    "rrf",
    "legacy_old",
    "legacy_entity",
    "legacy_old30",
    "legacy_entity30",
)
KS = (5, 10, 30)


def _ids_to_pairs(ids: list[str]) -> list[tuple[str, int]]:
    out = []
    for pid in ids:
        try:
            out.append(parse_page_id(pid))
        except ValueError:
            continue
    return out


def _metrics(
    ranked: list[tuple[str, int]],
    gold: set[tuple[str, int]],
    spans: list[list[int]],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in KS:
        top = ranked[:k]
        got = set(top)
        recall = len(gold & got) / len(gold) if gold else 0.0
        mrr = 0.0
        for rank, pair in enumerate(top, start=1):
            if pair in gold:
                mrr = 1.0 / rank
                break
        out[f"hit@{k}"] = 1.0 if gold & got else 0.0
        out[f"recall@{k}"] = recall
        out[f"mrr@{k}"] = mrr
        out[f"coverage@{k}"] = (
            sum(1 for a, b in spans if any(a <= page <= b for _, page in top)) / len(spans)
            if spans
            else None
        )
    return out


def _span_list(item: dict) -> list[list[int]]:
    spans = item.get("gold_spans")
    if spans:
        return [[int(x) for x in s] for s in spans if len(s) >= 2]
    span = item.get("gold_span")
    if span and len(span) >= 2:
        return [[int(span[0]), int(span[1])]]
    return []


def run_neo4j_probe(
    kb: KB,
    items: list[dict],
    cfg: dict,
    top_k: int | None = None,
    stage: Callable[[str], None] = print,
) -> list[dict]:
    """零 LLM：三路召回 + RRF，与旧字面锚词在同一批 gold 页上对比。"""
    top_k = int(top_k or cfg.get("neo4j", {}).get("top_k", 30))
    items = [i for i in items if i.get("gold_pages") and i.get("question")]
    index = kb.load_entities()
    lows_cache: dict[str, list[str]] = {}

    def lows_of(doc_id: str) -> list[str]:
        if doc_id not in lows_cache:
            total = kb.page_count(doc_id)
            lows_cache[doc_id] = [
                t.lower() for t in kb.read_pages(doc_id, 1, total)
            ]
        return lows_cache[doc_id]

    hybrid = Neo4jHybrid(
        cfg, cache_dir=Path(kb.home) / "cache" / "embeddings", kb=kb
    )
    rows: list[dict] = []
    try:
        stats = hybrid.stats()
        if not stats.get("pages"):
            raise RuntimeError("Neo4j 里没有 TWPage，先运行 tw neo4j build")
        stage(f"Neo4j 现状：{stats}；校验 embedding")
        hybrid.embedder.encode(["embedding 连通性测试"])
        for n, item in enumerate(items, start=1):
            doc_id = item.get("doc_id") or "src_0001"
            question = item["question"]
            gold = {(doc_id, int(p)) for p in item["gold_pages"]}
            spans = _span_list(item)
            try:
                result = hybrid.search(question, k=top_k)
                channels = {
                    name: _ids_to_pairs(ids) for name, ids in result["channels"].items()
                }
                channels["rrf"] = _ids_to_pairs(result["fused"])
                terms = result.get("terms") or []
                graph_entities = result.get("graph_entities") or []
                error = None
            except Exception as exc:  # 一条失败不该终止整轮
                channels = {name: [] for name in ("vector", "fulltext", "graph", "rrf")}
                terms, graph_entities, error = [], [], f"{type(exc).__name__}: {exc}"

            lows = lows_of(doc_id)
            old = [(doc_id, p) for p in _lexical_pages(lows, anchor_terms(question), 12)]
            old30 = [(doc_id, p) for p in _lexical_pages(lows, anchor_terms(question), 30)]
            ent_terms = question_anchors(question, index)
            groups = anchor_groups(ent_terms, index, 6)
            ent = [(doc_id, p) for p in _lexical_pages(lows, groups, 12)]
            ent30 = [(doc_id, p) for p in _lexical_pages(lows, groups, 30)]

            metrics = {
                name: _metrics(pairs, gold, spans) for name, pairs in channels.items()
            }
            metrics["legacy_old"] = _metrics(old, gold, spans)
            metrics["legacy_entity"] = _metrics(ent, gold, spans)
            metrics["legacy_old30"] = _metrics(old30, gold, spans)
            metrics["legacy_entity30"] = _metrics(ent30, gold, spans)

            rows.append(
                {
                    "id": item["id"],
                    "tier": item["tier"],
                    "question": question,
                    "gold_pages": sorted({p for _, p in gold}),
                    "gold_doc": doc_id,
                    "spans": spans,
                    "channels": {name: pairs[:top_k] for name, pairs in channels.items()},
                    "legacy": {"old": old, "entity": ent, "old30": old30, "entity30": ent30},
                    "metrics": metrics,
                    "terms": terms,
                    "graph_entities": graph_entities,
                    "error": error,
                }
            )
            if stage and (n % 25 == 0 or n == len(items)):
                stage(f"neo4j 检索 {n}/{len(items)}")
    finally:
        hybrid.close()
    return rows


def _rate(rows: list[dict], channel: str, metric: str) -> float | None:
    vals = [
        r["metrics"][channel].get(metric)
        for r in rows
        if r["metrics"].get(channel, {}).get(metric) is not None
    ]
    return round(sum(vals) / len(vals), 3) if vals else None


def _block(rows: list[dict]) -> dict[str, Any]:
    block: dict[str, Any] = {"n": len(rows), "errors": sum(1 for r in rows if r["error"])}
    for metric in ("hit@5", "hit@10", "recall@10", "mrr@10", "hit@30", "recall@30", "coverage@30"):
        block[metric] = {ch: _rate(rows, ch, metric) for ch in CHANNELS}
    return block


def summarize_neo4j(rows: list[dict]) -> dict[str, Any]:
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    out = {tier: _block(group) for tier, group in sorted(by_tier.items())}
    out["all"] = _block(rows)
    # 三路各自命中了多少题，以及 RRF 是否比单路多召回
    hits = {
        ch: sum(1 for r in rows if (r["metrics"][ch].get("hit@10") or 0) > 0)
        for ch in ("vector", "fulltext", "graph")
    }
    hits["rrf"] = sum(1 for r in rows if (r["metrics"]["rrf"].get("hit@10") or 0) > 0)
    hits["oracle"] = sum(
        1
        for r in rows
        if any((r["metrics"][ch].get("hit@10") or 0) > 0 for ch in ("vector", "fulltext", "graph"))
    )
    out["channel_hits@10"] = hits
    return out
