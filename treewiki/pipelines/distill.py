from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.llm import LLM
from ..core.store import KB
from ..core.tree import Node, iter_nodes
from ..core.wiki import (
    WikiPage,
    WikiStore,
    build_wiki_tree,
    merge_into,
    page_id_for,
)

EXTRACT_SYS = """你是知识库蒸馏器。从文档片段中抽取可复用的结构化知识，供后续 wiki 检索。
只依据原文，不推测、不补全原文没有的信息。只输出 JSON。"""

EXTRACT_USER = """来源文档：{doc_name}
补充说明：{hint}

待蒸馏内容：
{body}

请抽取并输出 JSON：
{{
  "summary": "本批内容要点（2-4 句）",
  "entities": [
    {{"name": "实体名", "type": "org|person|product|technology|concept|other",
      "aliases": ["别名"],
      "description": "1-2 句客观描述",
      "claims": [{{"claim": "一条明确事实（保留数字、日期、版本、专有名词）", "pages": [页码]}}]}}
  ],
  "topics": [
    {{"name": "主题/方法/概念名", "description": "1-2 句描述",
      "claims": [{{"claim": "一条明确事实", "pages": [页码]}}]}}
  ],
  "relations": [
    {{"from": "实体A", "predicate": "关系动词", "to": "实体B", "pages": [页码]}}
  ]
}}

要求：
- entities 是具体的组织、人物、产品、技术、概念等；topics 是方法、领域、主题。
- 每条 claim 必须能在原文找到依据，页码用文档绝对页码。
- relations 只在原文明确表达实体间关系时输出，from/to 必须是实体名。
- 每批最多 12 个实体、10 个主题、12 条关系；宁缺毋滥。"""


def _leaves(tree) -> list[Node]:
    leaves = [n for n, _ in iter_nodes(tree.structure) if not n.nodes]
    return leaves or list(tree.structure)


def _leaf_text(node: Node, pages: list[str], limit: int) -> str:
    parts = []
    for i in range(node.start_index, node.end_index + 1):
        if 1 <= i <= len(pages):
            text = (pages[i - 1] or "").strip()
            if text:
                parts.append(f"【第 {i} 页】\n{text}")
    text = "\n\n".join(parts)
    return text[:limit] + ("…" if len(text) > limit else "")


def _batch_leaves(
    leaves: list[Node], pages: list[str], leaf_chars: int, batch_chars: int
) -> Iterable[list[dict]]:
    batch: list[dict] = []
    size = 0
    for node in leaves:
        text = _leaf_text(node, pages, leaf_chars)
        if not text:
            continue
        item = {
            "title": node.title,
            "start": node.start_index,
            "end": node.end_index,
            "text": text,
            "pages": list(range(node.start_index, node.end_index + 1)),
        }
        cost = len(text) + len(node.title) + 40
        if batch and size + cost > batch_chars:
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += cost
    if batch:
        yield batch


def _pages(raw: Any, allowed: list[int]) -> list[int]:
    out = set()
    if isinstance(raw, (list, tuple)):
        for p in raw:
            try:
                n = int(p)
            except (TypeError, ValueError):
                continue
            if n in allowed:
                out.add(n)
    return sorted(out) or list(allowed)


def _claims(raw: Any, allowed: list[int], doc_id: str) -> list[dict]:
    out = []
    for c in raw or []:
        if isinstance(c, str):
            text, pages = c.strip(), list(allowed)
        elif isinstance(c, dict):
            text = str(c.get("claim") or c.get("text") or "").strip()
            pages = _pages(c.get("pages"), allowed)
        else:
            continue
        if text:
            out.append({"doc": doc_id, "pages": pages, "claim": text[:500]})
    return out


def _merge_claims(a: list[dict], b: list[dict]) -> list[dict]:
    seen = {(c["doc"], c["claim"], tuple(c["pages"])) for c in a}
    for c in b:
        key = (c["doc"], c["claim"], tuple(c["pages"]))
        if key not in seen:
            seen.add(key)
            a.append(c)
    return a


def _batch_key(batch: list[dict]) -> str:
    return "|".join(f"{item['start']}-{item['end']}" for item in batch)


def _cache_file(kb: KB, doc_id: str) -> Path:
    return kb.wiki_dir / "cache" / f"{doc_id}.jsonl"


def _load_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("key") and isinstance(rec.get("data"), dict):
            out[rec["key"]] = rec["data"]
    return out


def _append_cache(path: Path, key: str, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "data": data}, ensure_ascii=False) + "\n")


def _absorb(
    data: dict,
    allowed: list[int],
    doc_id: str,
    entities: dict[str, dict],
    topics: dict[str, dict],
    relations: dict[tuple[str, str, str], dict],
    summaries: list[str],
) -> None:
    """把一批抽取结果并进累计桶（与是否来自缓存无关）。"""
    summary = str(data.get("summary") or "").strip()
    if summary:
        summaries.append(summary)

    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()[:80]
        if not name:
            continue
        pid = page_id_for(name, "entity")
        entry = entities.setdefault(
            pid,
            {"title": name, "type": str(e.get("type") or "other")[:30],
             "description": "", "aliases": [], "claims": [], "relations": []},
        )
        desc = str(e.get("description") or "").strip()
        if desc and desc not in entry["description"]:
            entry["description"] = (entry["description"] + "；" + desc).strip("；")[:600]
        for a in e.get("aliases") or []:
            a = str(a).strip()
            if a and a != name and a not in entry["aliases"]:
                entry["aliases"].append(a[:80])
        _merge_claims(entry["claims"], _claims(e.get("claims"), allowed, doc_id))

    for t in data.get("topics") or []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name") or "").strip()[:80]
        if not name:
            continue
        pid = page_id_for(name, "topic")
        entry = topics.setdefault(
            pid,
            {"title": name, "type": "topic", "description": "", "aliases": [], "claims": []},
        )
        desc = str(t.get("description") or "").strip()
        if desc and desc not in entry["description"]:
            entry["description"] = (entry["description"] + "；" + desc).strip("；")[:600]
        _merge_claims(entry["claims"], _claims(t.get("claims"), allowed, doc_id))

    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        frm = str(r.get("from") or r.get("source") or "").strip()[:80]
        to = str(r.get("to") or r.get("target") or "").strip()[:80]
        pred = str(r.get("predicate") or r.get("relation") or "").strip()[:80]
        if not (frm and to and pred):
            continue
        key = (frm, pred, to)
        pages_r = _pages(r.get("pages"), allowed)
        if key in relations:
            relations[key]["pages"] = sorted(set(relations[key]["pages"]) | set(pages_r))
        else:
            relations[key] = {
                "from": frm, "predicate": pred, "to": to,
                "doc": doc_id, "pages": pages_r,
            }


def distill_source(
    doc_id: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    stage: Callable[[str], None] = print,
) -> dict:
    tree = kb.load_tree(doc_id)
    pages = kb.read_pages(doc_id, 1, kb.page_count(doc_id))
    source = kb.sources().get(doc_id, {})
    doc_name = tree.doc_name
    hint = tree.user_description or source.get("description", "")

    wcfg = cfg.get("wiki", {})
    leaf_chars = int(wcfg.get("leaf_chars", 3000))
    batch_chars = int(wcfg.get("batch_chars", 9000))
    max_batches = int(wcfg.get("max_batches", 30))
    # 单批抽取要输出实体/主题/关系三份清单，4000 token 会被截断成非法 JSON
    extract_max_tokens = int(wcfg.get("extract_max_tokens", 12000))

    batches = list(_batch_leaves(_leaves(tree), pages, leaf_chars, batch_chars))[:max_batches]
    if not batches:
        raise RuntimeError("没有可蒸馏的页面内容")

    entities: dict[str, dict] = {}
    topics: dict[str, dict] = {}
    relations: dict[tuple[str, str, str], dict] = {}
    summaries: list[str] = []
    failures: list[str] = []
    # 抽取结果按批的页区间做键缓存：重跑时命中就不必再调模型，进程被停掉也不丢已完成批
    cache_file = _cache_file(kb, doc_id)
    cache = _load_cache(cache_file)
    reused = 0

    for idx, batch in enumerate(batches, 1):
        key = _batch_key(batch)
        allowed = sorted({p for item in batch for p in item["pages"]})
        data = cache.get(key)
        if data is None:
            stage(f"wiki 蒸馏：批次 {idx}/{len(batches)}（{len(batch)} 个叶节点）…")
            body = "\n\n".join(
                f"【片段 {i}】{item['title']}（第 {item['start']}-{item['end']} 页）\n{item['text']}"
                for i, item in enumerate(batch, 1)
            )
            try:
                data = llm.chat_json(
                    [
                        {"role": "system", "content": EXTRACT_SYS},
                        {
                            "role": "user",
                            "content": EXTRACT_USER.format(
                                doc_name=doc_name, hint=hint or "（无）", body=body
                            ),
                        },
                    ],
                    model=cfg["llm"]["index_model"],
                    max_tokens=extract_max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                failures.append(f"批次 {idx}: {e}")
                stage(f"wiki 蒸馏：批次 {idx} 失败，跳过（{e}）")
                continue
            _append_cache(cache_file, key, data)
            cache[key] = data
        else:
            reused += 1
            stage(f"wiki 蒸馏：批次 {idx}/{len(batches)} 复用缓存（第 {allowed[0]}-{allowed[-1]} 页）")
        _absorb(data, allowed, doc_id, entities, topics, relations, summaries)

    if len(failures) == len(batches):
        raise RuntimeError("wiki 蒸馏全部失败：" + "；".join(failures[:3]))

    store = WikiStore(kb)
    # 关系端点解析：先实体后主题；都没有才补实体桩页，保证 [[互链]] 不断
    name_index = {e["title"]: pid for pid, e in entities.items()}
    name_index.update({t["title"]: pid for pid, t in topics.items()})
    for rel in relations.values():
        for name in (rel["from"], rel["to"]):
            if name in name_index:
                continue
            pid = page_id_for(name, "entity")
            entities.setdefault(
                pid,
                {
                    "title": name, "type": "other", "description": "",
                    "aliases": [], "claims": [], "relations": [],
                },
            )
            name_index[name] = pid
    entity_ids = set(entities)
    topic_ids = set(topics)

    # 来源摘要页
    source_page = WikiPage(
        id=doc_id,
        title=doc_name,
        type="source",
        description=hint or tree.doc_description,
        summary=(tree.doc_description or (summaries[0] if summaries else ""))[:400],
    )
    merge_into(
        source_page,
        description=hint or tree.doc_description,
        summary=(tree.doc_description or (summaries[0] if summaries else ""))[:400],
        sources=[doc_id],
        key_points=summaries,
        links=sorted(entity_ids | topic_ids),
    )
    store.save(source_page)

    # 实体页 / 主题页（关系按 from 端点落到对应页）
    for kind, buckets in (("entity", entities), ("topic", topics)):
        for pid, entry in buckets.items():
            page = store.load(pid) if store.find_path(pid) else WikiPage(
                id=pid, title=entry["title"], type=kind
            )
            rels = [
                {"predicate": r["predicate"], "target": name_index[r["to"]],
                 "doc": r["doc"], "pages": r["pages"]}
                for r in relations.values() if r["from"] == entry["title"]
            ]
            merge_into(
                page,
                description=entry["description"],
                summary=entry["description"][:200],
                tags=[entry.get("type", "")] if kind == "entity" and entry.get("type") else [],
                aliases=entry.get("aliases", []),
                sources=[doc_id],
                citations=entry["claims"],
                relations=rels,
                links=[r["target"] for r in rels],
            )
            store.save(page)

    store.write_index()
    tree_data = build_wiki_tree(store, kb)
    stats = {
        "entities": len(entities),
        "topics": len(topics),
        "relations": len(relations),
        "batches": len(batches),
        "reused": reused,
        "failed": len(failures),
        "pages": tree_data["page_count"],
    }
    store.append_log(doc_id, doc_name, stats)
    return stats
