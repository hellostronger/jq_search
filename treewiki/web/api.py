from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
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


def _kb() -> KB:
    return KB(kb_home())


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
