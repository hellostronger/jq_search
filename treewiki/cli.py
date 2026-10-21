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


@app.command()
def evalset(
    out: Path = Path("eval_out/dataset.jsonl"),
    doc_id: str = "src_0001",
    seed: int = 20261020,
    public: bool = False,
):
    """构造验证集：问题 + 标准答案 + 标准引用（gold 页由文章标记机械确定）

    --public 去掉原文整句与文章标题，产出可公开提交的脱敏版。
    """
    from .eval.dataset import build_dataset, sanitize_items, write_jsonl

    ds = build_dataset(KB(kb_home()), doc_id, seed=seed)
    items = sanitize_items(ds["items"]) if public else ds["items"]
    write_jsonl(out, items)
    console.print(
        f"[green]写出 {len(ds['items'])} 条 → {out}[/green]  "
        f"文章 {ds['articles']} / 页 {ds['pages']} / 事实候选 {ds['candidate_facts']} "
        f"(A {ds['candidate_a']} · B {ds['candidate_b']} · S {ds['candidate_s']}) "
        f"时序 {ds['candidate_timeline']} 未收录 {ds['candidate_absent']} "
        f"重复并入 {ds['candidate_duplicate_gold']} · 冲突剔除 {ds['candidate_duplicate_conflict']} "
        f"同问并入 {ds['candidate_merged']} · 同问冲突 {ds['candidate_multi_answer']} "
        f"残句 {ds['candidate_bad_clause']} · 缺主体 {ds['candidate_underspecified']} "
        f"· 去重 {ds['candidate_deduped']}"
    )


@app.command()
def evalrun(
    dataset: Path = Path("eval_out/dataset.jsonl"),
    results: Path = Path("eval_out/retrieval.jsonl"),
    mode: str = "retrieval",
    workers: int = 4,
    limit: int = 0,
    tiers: str = "A,B,S,C,D",
):
    """评测：retrieval/tree/floor 零成本探针，full 并发调用真实模型跑端到端"""
    from .eval.runner import main as run

    run(dataset, results, mode, workers, limit, tiers, kb_home())


def main():
    app()


if __name__ == "__main__":
    main()
