from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import kb_home, load_config, save_config
from ..core.llm import LLM
from ..core.store import KB
from ..core.tree import iter_nodes, skeleton_text
from ..pipelines.ingest import ingest_file
from ..pipelines.query import query_raw

app = FastAPI(title="TreeWiki")
STATIC_DIR = Path(__file__).parent / "static"
JOBS: dict[str, dict] = {}

WIKI_SECTIONS = (
    ("entities", "实体页"),
    ("topics", "主题页"),
    ("sources", "来源摘要页"),
    ("analyses", "分析页"),
)
SECTION_TYPES = {
    "entities": "entity",
    "topics": "topic",
    "sources": "source",
    "analyses": "analysis",
}
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _kb() -> KB:
    return KB(kb_home())


def _read_wiki_doc(path: Path) -> tuple[dict[str, Any], str, list[str]]:
    """解析 wiki markdown 的 YAML frontmatter、正文和 [[互链]]。"""
    text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            parsed = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            parsed = {}
        if isinstance(parsed, dict):
            meta = parsed
        body = text[m.end():]
    links = list(
        dict.fromkeys(x.strip() for x in _WIKI_LINK_RE.findall(body) if x.strip())
    )
    return meta, body, links


class ConfigPayload(BaseModel):
    llm: dict | None = None
    mineru: dict | None = None
    index: dict | None = None
    query: dict | None = None


class QueryPayload(BaseModel):
    doc_id: str
    question: str
    mode: str = "raw"


# ------------------------------------------------------------------- 配置

@app.get("/api/config")
def get_config():
    cfg = load_config(kb_home())
    return {"config": cfg, "home": str(kb_home())}


@app.put("/api/config")
def put_config(payload: ConfigPayload):
    home = kb_home()
    cfg = load_config(home)
    for section in ("llm", "mineru", "index", "query"):
        incoming = getattr(payload, section)
        if isinstance(incoming, dict):
            cfg.setdefault(section, {}).update(
                {k: v for k, v in incoming.items() if v is not None}
            )
    save_config(cfg, home)
    return {"ok": True, "config": load_config(home)}


@app.post("/api/test-llm")
def test_llm():
    cfg = load_config(kb_home())
    try:
        llm = LLM(cfg)
        t0 = time.time()
        reply = llm.chat(
            [{"role": "user", "content": "请只回复两个字：连通"}],
            model=cfg["llm"]["chat_model"],
            max_tokens=16,
        )
        return {"ok": True, "reply": reply.strip(), "elapsed": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------- 摄取

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), description: str = Form("")):
    description = description.strip()[:500]
    home = kb_home()
    updir = home / ".uploads"
    updir.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload.bin").name
    work_dir = updir / uuid.uuid4().hex[:12]
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / name
    with open(path, "wb") as f:
        f.write(await file.read())

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "stages": [], "doc_id": None, "error": None}

    def run() -> None:
        try:
            cfg = load_config(home)
            kb = KB(home)
            doc_id = ingest_file(
                path,
                kb,
                cfg,
                stage=lambda s: JOBS[job_id]["stages"].append(s),
                description=description,
            )
            JOBS[job_id].update(status="done", doc_id=doc_id)
        except Exception as e:  # noqa: BLE001
            JOBS[job_id].update(status="error", error=str(e))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


# ------------------------------------------------------------------- 查询

@app.get("/api/sources")
def get_sources():
    kb = _kb()
    out = []
    for doc_id, info in kb.sources().items():
        try:
            tree = kb.load_tree(doc_id)
            nodes = [n for n, _ in iter_nodes(tree.structure)]
            tree_info = {
                "nodes": len(nodes),
                "description": info.get("description", "") or tree.user_description,
                "doc_description": tree.doc_description,
            }
        except FileNotFoundError:
            tree_info = {"nodes": 0, "description": "", "doc_description": ""}
        out.append(
            {
                "doc_id": doc_id,
                "name": Path(info.get("path", "")).name,
                "pages": info.get("pages", 0),
                "status": info.get("status", ""),
                "ingested_at": info.get("ingested_at", ""),
                **tree_info,
            }
        )
    return {"sources": out}


@app.get("/api/tree/{doc_id}")
def get_tree(doc_id: str):
    try:
        return {"tree": kb_home_structure(doc_id)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


def kb_home_structure(doc_id: str):
    return _kb().load_tree(doc_id).model_dump()


@app.get("/api/pages/{doc_id}")
def get_pages(
    doc_id: str,
    start: int = Query(1, ge=1),
    end: int | None = Query(None, ge=1),
):
    """读取节点覆盖的原始页（一次最多 5 页），供树索引详情面板预览。"""
    kb = _kb()
    if doc_id not in kb.sources():
        raise HTTPException(404, f"未找到来源 {doc_id}")
    total = kb.page_count(doc_id)
    if total <= 0:
        return {
            "doc_id": doc_id,
            "start": 0,
            "end": 0,
            "total": 0,
            "pages": [],
            "has_more": False,
        }
    start = min(start, total)
    end = max(start, min(end or start, total))
    end = min(end, start + 4)
    texts = kb.read_pages(doc_id, start, end)
    pages = []
    for offset, text in enumerate(texts):
        clipped = len(text) > 6000
        pages.append(
            {
                "page": start + offset,
                "text": text[:6000] + ("…" if clipped else ""),
                "truncated": clipped,
            }
        )
    return {
        "doc_id": doc_id,
        "start": start,
        "end": end,
        "total": total,
        "pages": pages,
        "has_more": end < total,
    }


@app.get("/api/wiki/overview")
def get_wiki_overview():
    """LLMWiki 产物清单：页面、特殊文件与 wiki 树（M2 尚未生成时为空态）。"""
    home = kb_home()
    wiki = home / "wiki"
    sections = []
    pages: list[dict[str, Any]] = []
    by_type: dict[str, int] = {}
    total_links = 0
    for dirname, label in WIKI_SECTIONS:
        d = wiki / dirname
        files = sorted(d.glob("*.md")) if d.is_dir() else []
        sections.append(
            {
                "name": dirname,
                "label": label,
                "count": len(files),
                "path": f"wiki/{dirname}",
            }
        )
        for p in files:
            try:
                meta, body, links = _read_wiki_doc(p)
            except (OSError, UnicodeDecodeError):
                continue
            page_type = str(meta.get("type") or SECTION_TYPES.get(dirname, dirname))
            citations = meta.get("citations")
            tags = meta.get("tags")
            sources = meta.get("sources")
            pages.append(
                {
                    "id": str(meta.get("id") or p.stem),
                    "title": str(meta.get("title") or p.stem),
                    "type": page_type,
                    "section": dirname,
                    "path": p.relative_to(wiki).as_posix(),
                    "tags": tags if isinstance(tags, list) else [],
                    "sources": sources if isinstance(sources, list) else [],
                    "citations": len(citations) if isinstance(citations, list) else 0,
                    "updated": str(meta.get("updated") or ""),
                    "stale": bool(meta.get("stale", False)),
                    "links": links,
                    "chars": len(body),
                }
            )
            by_type[page_type] = by_type.get(page_type, 0) + 1
            total_links += len(links)

    special: dict[str, Any] = {}
    for name in ("index.md", "log.md", "SCHEMA.md"):
        p = wiki / name
        content = ""
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                content = ""
        special[name] = {
            "exists": p.is_file(),
            "chars": len(content),
            "content": content[:12000],
        }

    wiki_tree = None
    tree_path = home / "index" / "wiki_tree.json"
    if tree_path.is_file():
        try:
            raw_tree = json.loads(tree_path.read_text(encoding="utf-8"))
            wiki_tree = {
                "built_at": raw_tree.get("built_at"),
                "page_count": raw_tree.get("page_count"),
                "root": raw_tree.get("root"),
            }
        except (OSError, json.JSONDecodeError):
            wiki_tree = None

    return {
        "status": "ready" if pages else "empty",
        "root": str(wiki),
        "sections": sections,
        "pages": pages,
        "counts": {
            "pages": len(pages),
            "links": total_links,
            "by_type": by_type,
        },
        "special": special,
        "wiki_tree": wiki_tree,
    }


@app.get("/api/wiki/page")
def get_wiki_page(path: str):
    wiki = (kb_home() / "wiki").resolve()
    if not wiki.is_dir():
        raise HTTPException(404, "wiki 目录不存在")
    target = (wiki / path).resolve()
    if target == wiki or wiki not in target.parents:
        raise HTTPException(404, "页面不存在")
    if target.suffix.lower() != ".md" or not target.is_file():
        raise HTTPException(404, "页面不存在")
    try:
        meta, body, links = _read_wiki_doc(target)
    except (OSError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"页面读取失败：{e}")
    return {
        "path": target.relative_to(wiki).as_posix(),
        "meta": meta,
        "body": body,
        "links": links,
    }


@app.post("/api/query")
def run_query(payload: QueryPayload):
    if payload.mode != "raw":
        raise HTTPException(400, "v1 仅支持 raw 模式（wiki/cascade/parallel 见设计文档 M2/M3）")
    cfg = load_config(kb_home())
    try:
        llm = LLM(cfg)
    except Exception as e:
        raise HTTPException(400, f"LLM 配置无效: {e}")
    try:
        result = query_raw(payload.doc_id, payload.question, _kb(), llm, cfg)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return result


@app.get("/api/skeleton/{doc_id}")
def get_skeleton(doc_id: str):
    try:
        tree = _kb().load_tree(doc_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"skeleton": skeleton_text(tree.structure, with_summary=False)}


# ------------------------------------------------------------------- 静态页

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
