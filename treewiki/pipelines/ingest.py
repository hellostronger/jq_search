from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from ..config import load_config
from ..core import mineru, native
from ..core.llm import LLM
from ..core.store import KB, sha256_file
from ..core.tree import (
    TreeDoc,
    add_summaries,
    assign_ids,
    classic_skeleton,
    describe_doc,
    flash_skeleton,
)


def ingest_file(
    path: Path | str,
    kb: KB,
    cfg: dict | None = None,
    stage: Callable[[str], None] = print,
    description: str = "",
) -> str:
    """摄取一个来源：MinerU 解析 → 页缓存 → 建树 → 摘要 → 注册。返回 doc_id。"""
    cfg = cfg or load_config(kb.home)
    description = (description or "").strip()[:500]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    doc_id = kb.new_doc_id()
    dest = kb.raw_dir / f"{doc_id}_{path.name}"
    shutil.copy2(path, dest)
    sha = sha256_file(dest)

    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        stage(f"[{doc_id}] 本地文本解析（md/txt 不经过 MinerU）…")
        res = native.parse_local(dest)
    else:
        stage(f"[{doc_id}] 调用 MinerU 云 API 解析 …")
        res = mineru.parse_cloud(
            dest,
            token=cfg["mineru"]["token"],
            model_version=cfg["mineru"]["model_version"],
            language=cfg["mineru"]["language"],
            is_ocr=cfg["mineru"]["is_ocr"],
            enable_formula=cfg["mineru"]["enable_formula"],
            stage_cb=stage,
        )
    pdir = kb.parse_dir / doc_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "document.md").write_text(res["md"], encoding="utf-8")
    (pdir / "content_list.json").write_text(
        json.dumps(res["content_list"], ensure_ascii=False, indent=1), encoding="utf-8"
    )

    pages = res.get("pages")
    if pages is None:
        pages = mineru.normalize_pages(res["content_list"])
    if not pages:
        raise RuntimeError("解析结果为空")
    kb.write_pages(doc_id, pages)
    stage(f"[{doc_id}] 解析完成：{len(pages)} 页")

    llm = LLM(cfg)
    structure = flash_skeleton(pages)
    if structure is not None:
        stage("[{}] flash 建树：检测到 {} 个标题层级节点".format(doc_id, structure and len(list(iter_nodes_flat(structure)))))
    else:
        stage(f"[{doc_id}] 无标题结构，classic 建树（LLM）…")
        structure = classic_skeleton(
            pages,
            llm,
            cfg["llm"]["index_model"],
            batch_size=cfg["index"]["classic_batch"],
            stage=stage,
        )
    assign_ids(structure)

    if cfg["index"]["summaries"]:
        add_summaries(
            structure,
            pages,
            llm,
            cfg["llm"]["index_model"],
            workers=cfg["index"]["summary_workers"],
            stage=stage,
        )
    desc = describe_doc(pages, structure, llm, cfg["llm"]["index_model"], hint=description)

    tree = TreeDoc(
        doc_id=doc_id,
        doc_name=path.name,
        doc_description=desc,
        user_description=description,
        structure=structure,
    )
    kb.save_tree(tree)
    kb.register(doc_id, dest, sha, len(pages), description=description)
    stage(f"[{doc_id}] 完成：{len(pages)} 页，{tree_stats(tree)}")
    return doc_id


def iter_nodes_flat(structure):
    from ..core.tree import iter_nodes

    for n, _ in iter_nodes(structure):
        yield n


def tree_stats(tree: TreeDoc) -> str:
    nodes = list(iter_nodes_flat(tree.structure))
    leaves = [n for n in nodes if not n.nodes]
    return f"{len(nodes)} 节点 / {len(leaves)} 叶"
