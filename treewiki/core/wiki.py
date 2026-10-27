from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

TYPE_DIRS = {
    "entity": "entities",
    "topic": "topics",
    "source": "sources",
    "analysis": "analyses",
}
TYPE_LABELS = {"source": "来源", "entity": "实体", "topic": "主题", "analysis": "分析"}
PREFIX_TYPES = {"ent_": "entity", "top_": "topic", "src_": "source", "ana_": "analysis"}


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "").strip()).lower()
    text = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-_")
    if not text:
        text = hashlib.sha1((text or "page").encode("utf-8")).hexdigest()[:10]
    return text[:60]


def page_id_for(name: str, kind: str) -> str:
    prefix = {"entity": "ent", "topic": "top", "source": "src", "analysis": "ana"}[kind]
    return f"{prefix}_{slugify(name)}"


def type_of(page_id: str) -> str | None:
    for prefix, kind in PREFIX_TYPES.items():
        if page_id.startswith(prefix):
            return kind
    return None


@dataclass
class WikiPage:
    id: str
    title: str
    type: str
    description: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    updated: str = ""
    stale: bool = False
    body: str = ""

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["body"] = self.body or render_body(self)
        return data


def _pages_label(pages: Iterable[int]) -> str:
    pages = sorted({int(p) for p in pages if int(p) > 0})
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p{pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"p{pages[0]}-{pages[-1]}"
    return "p" + ",".join(str(p) for p in pages)


def _cite_text(c: dict) -> str:
    label = _pages_label(c.get("pages") or [])
    return f"[[{c.get('doc', '')}]] {label}".strip()


def render_body(page: WikiPage) -> str:
    parts: list[str] = []
    if page.description:
        parts.append("## 概述\n" + page.description.strip())

    if page.key_points:
        parts.append("## 关键要点\n" + "\n".join(f"- {x}" for x in page.key_points))

    facts = [c for c in page.citations if c.get("claim")]
    if facts:
        parts.append(
            "## 关键事实\n"
            + "\n".join(f"- {c['claim']}（{_cite_text(c)}）" for c in facts)
        )

    if page.relations:
        parts.append(
            "## 关系\n"
            + "\n".join(
                f"- {r.get('predicate', '相关')} → [[{r.get('target', '')}]]"
                f"（[[{r.get('doc', '')}]] {_pages_label(r.get('pages') or [])}）".rstrip()
                for r in page.relations
            )
        )

    rel_targets = {r.get("target") for r in page.relations}
    extra_links = [x for x in page.links if x not in rel_targets and x not in page.sources]
    if extra_links:
        parts.append("## 关联\n" + "\n".join(f"- [[{x}]]" for x in extra_links))

    if page.sources and page.type != "source":
        parts.append("## 来源\n" + "\n".join(f"- [[{x}]]" for x in page.sources))

    return "\n\n".join(parts).strip()


def dump_page(page: WikiPage) -> str:
    meta: dict[str, Any] = {
        "id": page.id,
        "title": page.title,
        "type": page.type,
        "tags": page.tags,
        "aliases": page.aliases,
        "description": page.description,
        "summary": page.summary,
        "sources": page.sources,
        "citations": page.citations,
        "relations": page.relations,
        "key_points": page.key_points,
        "links": page.links,
        "updated": page.updated or today(),
        "stale": page.stale,
    }
    meta = {k: v for k, v in meta.items() if v not in ([], "", None) or k in ("id", "title", "type", "updated", "stale")}
    head = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
    body = page.body.strip() or render_body(page)
    return f"---\n{head}\n---\n\n{body}\n"


def parse_page(text: str) -> WikiPage:
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            raw = text[3:end].lstrip("\n")
            meta = yaml.safe_load(raw) or {}
            body = text[end + 4 :].lstrip("\n")
    allowed = {
        "id", "title", "type", "description", "summary", "tags", "aliases",
        "sources", "citations", "relations", "key_points", "links", "updated", "stale",
    }
    data = {k: v for k, v in meta.items() if k in allowed}
    data.setdefault("id", "")
    data.setdefault("title", data.get("id", ""))
    data.setdefault("type", type_of(data["id"]) or "topic")
    data["body"] = body.strip()
    return WikiPage(**data)


def merge_into(
    page: WikiPage,
    *,
    description: str = "",
    summary: str = "",
    tags: Iterable[str] = (),
    aliases: Iterable[str] = (),
    sources: Iterable[str] = (),
    citations: Iterable[dict] = (),
    relations: Iterable[dict] = (),
    key_points: Iterable[str] = (),
    links: Iterable[str] = (),
) -> WikiPage:
    description = (description or "").strip()
    if description and description not in page.description:
        if page.description and page.description not in description:
            page.description = (page.description.rstrip("。；") + "；" + description)[:1000]
        else:
            page.description = description
    if summary and not page.summary:
        page.summary = summary[:400]

    for attr, values in (
        ("tags", tags), ("aliases", aliases), ("sources", sources),
        ("key_points", key_points), ("links", links),
    ):
        current = getattr(page, attr)
        for v in values:
            v = str(v).strip()
            if v and v not in current:
                current.append(v)

    seen = {(c.get("doc"), c.get("claim") or "", tuple(c.get("pages") or [])) for c in page.citations}
    for c in citations:
        if not isinstance(c, dict) or not c.get("doc"):
            continue
        pages = [int(p) for p in (c.get("pages") or []) if str(p).isdigit()]
        item = {"doc": str(c["doc"]), "pages": sorted(set(pages)), "claim": str(c.get("claim") or "").strip()}
        key = (item["doc"], item["claim"], tuple(item["pages"]))
        if key not in seen:
            seen.add(key)
            page.citations.append(item)

    rseen = {(r.get("predicate"), r.get("target"), r.get("doc")): r for r in page.relations}
    for r in relations:
        if not isinstance(r, dict):
            continue
        target, predicate = str(r.get("target") or "").strip(), str(r.get("predicate") or "").strip()
        doc = str(r.get("doc") or "").strip()
        if not target or not predicate:
            continue
        pages = sorted({int(p) for p in (r.get("pages") or []) if str(p).isdigit()})
        key = (predicate, target, doc)
        if key in rseen:
            rseen[key]["pages"] = sorted(set(rseen[key].get("pages") or []) | set(pages))
        else:
            item = {"predicate": predicate, "target": target, "doc": doc, "pages": pages}
            page.relations.append(item)
            rseen[key] = item
    return page


class WikiStore:
    def __init__(self, kb):
        self.kb = kb
        self.root = kb.wiki_dir
        for d in TYPE_DIRS.values():
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.tree_path = kb.home / "index" / "wiki_tree.json"

    def page_path(self, page_id: str) -> Path:
        kind = type_of(page_id) or "entity"
        return self.root / TYPE_DIRS[kind] / f"{page_id}.md"

    def find_path(self, page_id: str) -> Path | None:
        p = self.page_path(page_id)
        if p.is_file():
            return p
        for d in TYPE_DIRS.values():
            q = self.root / d / f"{page_id}.md"
            if q.is_file():
                return q
        return None

    def load(self, page_id: str) -> WikiPage:
        p = self.find_path(page_id)
        if not p:
            raise FileNotFoundError(f"未找到 wiki 页: {page_id}")
        return parse_page(p.read_text(encoding="utf-8"))

    def save(self, page: WikiPage) -> None:
        page.updated = page.updated or today()
        page.body = page.body.strip() or render_body(page)
        p = self.page_path(page.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dump_page(page), encoding="utf-8")

    def list(self, type: str | None = None) -> list[WikiPage]:
        kinds = [type] if type in TYPE_DIRS else list(TYPE_DIRS)
        out: list[WikiPage] = []
        for kind in kinds:
            for p in sorted((self.root / TYPE_DIRS[kind]).glob("*.md")):
                try:
                    out.append(parse_page(p.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return out

    def write_index(self) -> None:
        pages = self.list()
        lines = [
            "# 知识库索引",
            "",
            f"更新时间：{today()} ｜ 共 {len(pages)} 页",
            "",
        ]
        for kind in ("source", "entity", "topic", "analysis"):
            items = [p for p in pages if p.type == kind]
            if not items:
                continue
            lines.append(f"## {TYPE_LABELS[kind]}（{len(items)}）")
            for p in sorted(items, key=lambda x: x.title):
                summary = (p.summary or p.description).replace("\n", " ")[:160]
                lines.append(f"- [[{p.id}]] {p.title}" + (f" — {summary}" if summary else ""))
            lines.append("")
        (self.root / "index.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    def append_log(self, doc_id: str, doc_name: str, stats: dict) -> None:
        log = self.root / "log.md"
        if not log.exists():
            log.write_text("# Wiki 操作日志\n", encoding="utf-8")
        entry = (
            f"\n## [{today()}] ingest | {doc_name}\n"
            f"- doc_id: {doc_id}\n"
            f"- 实体 {stats.get('entities', 0)} ｜ 主题 {stats.get('topics', 0)} ｜ "
            f"关系 {stats.get('relations', 0)} ｜ 蒸馏批次 {stats.get('batches', 0)}"
            + (f" ｜ 失败 {stats['failed']}" if stats.get("failed") else "")
            + "\n"
        )
        with open(log, "a", encoding="utf-8") as f:
            f.write(entry)


def build_wiki_tree(store: WikiStore, kb) -> dict:
    pages = store.list()
    root = {
        "node_id": "",
        "title": "知识库",
        "summary": f"共 {len(pages)} 个 wiki 页，覆盖 {len(kb.sources())} 个来源。",
        "pages": [],
        "children": [],
    }
    for kind in ("source", "entity", "topic", "analysis"):
        items = sorted([p for p in pages if p.type == kind], key=lambda x: x.title)
        if not items:
            continue
        group = {
            "title": TYPE_LABELS[kind],
            "summary": f"{len(items)} 个{TYPE_LABELS[kind]}页。",
            "pages": [],
            "children": [
                {
                    "title": p.title,
                    "summary": (p.summary or p.description or "")[:200],
                    "pages": [p.id],
                    "children": [],
                }
                for p in items
            ],
        }
        root["children"].append(group)

    counter = {"n": 0}

    def assign(node: dict) -> None:
        counter["n"] += 1
        node["node_id"] = f"w{counter['n']:04d}"
        for ch in node.get("children") or []:
            assign(ch)

    for ch in root["children"]:
        assign(ch)
    tree = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "root": root,
    }
    store.tree_path.parent.mkdir(parents=True, exist_ok=True)
    store.tree_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return tree
