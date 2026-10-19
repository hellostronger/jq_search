from __future__ import annotations

from typing import Any

from ..core.llm import LLM
from ..core.store import KB
from ..core.tree import Node, TreeDoc, find_node, iter_nodes

LOCALIZE_SYS = """你是文档检索专家，正在一棵文档树索引中定位答案位置（类似人类专家翻目录）。
给你问题、文档描述和当前候选节点（含编号、标题、页范围、摘要及子节点）。
选择下一步动作，只输出 JSON：
{"expand": ["节点编号"], "read": ["节点编号"], "reason": "简短理由"}
- "expand"：进入某分支，返回其子节点
- "read"：该节点是要读的叶子，读取其覆盖页的原文
两者二选一，可给多个编号。叶节点用 read，中间节点用 expand。"""

ANSWER_SYS = """你是严谨的问答助手，只能依据提供的文档页原文回答。
输出 JSON：
{"answer": "答案（引用处标注 [p页码]，例如：收入为 247 亿美元 [p1]）", "pages": [页码列表], "sufficient": true}
若原文确实没有答案，sufficient 设为 false 并在 answer 中说明。禁止编造。"""


def _listing(nodes: list[Node]) -> str:
    lines: list[str] = []
    for n in nodes:
        line = f"[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index})"
        if n.summary:
            line += f" — {n.summary[:200]}"
        lines.append(line)
        for ch in n.nodes[:12]:
            lines.append(
                f"    [{ch.node_id}] {ch.title} (p{ch.start_index}-{ch.end_index})"
                + (f" — {ch.summary[:120]}" if ch.summary else "")
            )
        if len(n.nodes) > 12:
            lines.append(f"    … 共 {len(n.nodes)} 个子节点")
    return "\n".join(lines)


def _leaves_of(n: Node) -> list[Node]:
    return [x for x, _ in iter_nodes([n]) if not x.nodes] or [n]


def _dedupe(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    out = []
    for n in nodes:
        if n.node_id and n.node_id not in seen:
            seen.add(n.node_id)
            out.append(n)
    return out


def _pages_of(nodes: list[Node], total: int, cap: int) -> list[int]:
    pages: set[int] = set()
    for n in nodes:
        for p in range(n.start_index, n.end_index + 1):
            if 1 <= p <= total:
                pages.add(p)
    return sorted(pages)[:cap]


def _read_context(kb: KB, doc_id: str, pages: list[int], page_chars: int) -> str:
    texts = kb.read_pages(doc_id, pages[0], pages[-1]) if pages else []
    parts = []
    for off, p in enumerate(pages):
        text = (texts[off] if off < len(texts) else "") or "（空页）"
        if len(text) > page_chars:
            text = text[:page_chars] + "…"
        parts.append(f"【第 {p} 页】\n{text}")
    return "\n\n".join(parts)


def query_raw(
    doc_id: str,
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
) -> dict:
    """PageIndex 式树遍历问答（raw 路）：定位 → 读页 → 带页级引用作答。"""
    tree: TreeDoc = kb.load_tree(doc_id)
    total = kb.page_count(doc_id)
    qcfg = cfg["query"]
    chat_model = cfg["llm"]["chat_model"]
    trace: list[dict] = []

    frontier = list(tree.structure)
    read_nodes: list[Node] = []
    for rnd in range(1, qcfg["max_rounds"] + 1):
        if not frontier:
            break
        data = llm.chat_json(
            [
                {"role": "system", "content": LOCALIZE_SYS},
                {
                    "role": "user",
                    "content": (
                        f"文档：{tree.doc_name} —— {tree.doc_description}\n"
                        f"问题：{question}\n\n"
                        f"当前候选节点：\n{_listing(frontier)}"
                    ),
                },
            ],
            model=chat_model,
        )
        expand = [x for x in (data.get("expand") or []) if isinstance(x, str)]
        read = [x for x in (data.get("read") or []) if isinstance(x, str)]
        trace.append(
            {"round": rnd, "expand": expand, "read": read, "reason": data.get("reason", "")}
        )
        for nid in read:
            n = find_node(tree.structure, nid)
            if n:
                read_nodes.extend(_leaves_of(n))
        if read_nodes:
            break
        next_frontier: list[Node] = []
        for nid in expand:
            n = find_node(tree.structure, nid)
            if n is None:
                continue
            if n.nodes:
                next_frontier.extend(n.nodes)
            else:
                read_nodes.extend(_leaves_of(n))
        if not next_frontier and not read_nodes:
            for n in frontier:  # 兜底：读当前层
                read_nodes.extend(_leaves_of(n))
            break
        frontier = next_frontier[:10]

    if not read_nodes:
        for n in frontier:
            read_nodes.extend(_leaves_of(n))
    read_nodes = _dedupe(read_nodes)[:8]
    pages = _pages_of(read_nodes, total, qcfg["max_pages"])

    for attempt in range(2):
        context = _read_context(kb, doc_id, pages, qcfg["page_chars"])
        data = llm.chat_json(
            [
                {"role": "system", "content": ANSWER_SYS},
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        f"文档页原文（节点：{'; '.join(f'{n.title}(p{n.start_index}-{n.end_index})' for n in read_nodes)}）：\n\n"
                        f"{context}"
                    ),
                },
            ],
            model=chat_model,
        )
        if data.get("sufficient", True) or attempt == 1:
            break
        # 答案不足：扩大读取范围（其余候选叶子）再试一次
        extra = _dedupe(
            [x for n in frontier for x in _leaves_of(n)] + read_nodes
        )[:12]
        pages = _pages_of(extra, total, qcfg["max_pages"])
        read_nodes = extra

    cited: list[int] = []
    for p in data.get("pages") or []:
        try:
            cited.append(int(p))
        except (TypeError, ValueError):
            continue
    cited = [p for p in cited if p in pages] or pages

    return {
        "doc_id": doc_id,
        "answer": str(data.get("answer", "")),
        "citations": [{"doc_id": doc_id, "pages": cited}],
        "read_pages": pages,
        "read_nodes": [
            {"id": n.node_id, "title": n.title, "pages": [n.start_index, n.end_index]}
            for n in read_nodes
        ],
        "rounds": trace,
        "usage": llm.usage(),
    }
