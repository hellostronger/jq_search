from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import kb_home, load_config
from .core.llm import LLM
from .core.store import KB
from .core.tree import iter_nodes, skeleton_text
from .pipelines.ingest import ingest_file
from .pipelines.query import query_raw

app = typer.Typer(no_args_is_help=True, help="TreeWiki — LLMWiki × PageIndex 知识库")
console = Console()


@app.command()
def serve(port: int = 8000):
    """启动 Web UI"""
    import uvicorn

    uvicorn.run("treewiki.web.api:app", host="127.0.0.1", port=port)


@app.command()
def ingest(
    path: Path,
    description: str = typer.Option("", "--desc", "-d", help="补充说明，如版本、用途"),
):
    """摄取一个来源（MinerU 解析 → 建树 → 摘要）"""
    cfg = load_config()
    kb = KB(kb_home())
    doc_id = ingest_file(path, kb, cfg, stage=console.print, description=description)
    console.print(f"[green]完成：{doc_id}[/green]")


@app.command()
def query(
    question: str,
    doc: str = typer.Option(..., help="doc_id，如 src_0001"),
):
    """树遍历问答（raw 路）"""
    cfg = load_config()
    llm = LLM(cfg)
    result = query_raw(doc, question, KB(kb_home()), llm, cfg)
    console.print(result["answer"])
    pages = (result["citations"][0]["pages"] if result["citations"] else [])
    console.print(f"[dim]引用页: {pages} | 定位轮数: {len(result['rounds'])} | "
                  f"LLM 调用: {result['usage']['calls']}[/dim]")


@app.command()
def status():
    """来源统计"""
    kb = KB(kb_home())
    table = Table(title=f"知识库 {kb.home}")
    table.add_column("doc_id")
    table.add_column("文件")
    table.add_column("页数", justify="right")
    table.add_column("节点", justify="right")
    table.add_column("状态")
    for doc_id, info in kb.sources().items():
        try:
            n_nodes = len([n for n, _ in iter_nodes(kb.load_tree(doc_id).structure)])
        except FileNotFoundError:
            n_nodes = 0
        table.add_row(
            doc_id, Path(info.get("path", "")).name, str(info.get("pages", 0)),
            str(n_nodes), info.get("status", ""),
        )
    console.print(table)


@app.command()
def tree(doc_id: str, summary: bool = typer.Option(False, "--summary", help="显示节点摘要")):
    """打印树索引"""
    kb = KB(kb_home())
    console.print(skeleton_text(kb.load_tree(doc_id).structure, with_summary=summary))


def main():
    app()


if __name__ == "__main__":
    main()
