from __future__ import annotations

import re
from typing import Any

from ..core.llm import LLM
from ..core.store import KB
from ..core.tree import Node, TreeDoc, find_node, iter_nodes

LOCALIZE_SYS = """你是文档检索专家，正在一棵文档树索引中定位答案位置（类似人类专家翻目录）。
给你问题、文档描述和当前候选节点（含编号、标题、页范围、摘要及子节点）。
选择下一步动作，只输出 JSON：
{"expand": ["节点编号"], "read": ["节点编号"], "reason": "简短理由"}
- "expand"：进入某分支，返回其子节点
- "read"：读取该节点覆盖页的原文；叶节点直接读，中间节点会读取其代表性叶子
- 必须至少给出一个动作，编号必须原样来自候选节点列表，禁止臆造
- 问题需要全局统计/汇总时，优先 expand 覆盖全篇的分支，或 read 多个不同分支"""

ANSWER_SYS = """你是严谨的问答助手，只能依据提供的节点摘要和文档页原文回答。
节点摘要用于定位与全局概览，不能作为页码引用的依据；具体事实以原文为准。
输出 JSON：
{"answer": "答案（引用处标注 [p页码]，例如：收入为 247 亿美元 [p1]）", "pages": [页码列表], "sufficient": true}
- answer 必须非空；证据不足时说明缺少什么信息，并把 sufficient 设为 false
- pages 只能填写原文部分实际出现的页码
禁止编造。"""

_CITE_PARTS = re.compile(r"[,，;；、\s]+")
_CITE_ITEM = re.compile(r"^[pP]?\s*(\d+)(?:\s*[-–~至]\s*[pP]?\s*(\d+))?$")
_DIGITS = re.compile(r"\d{3,}")


def _listing(nodes: list[Node]) -> str:
    lines: list[str] = []
    for n in nodes:
        line = f"[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index})"
        if n.summary:
            line += f" — {n.summary[:200]}"
        lines.append(line)
        children = _spread_sample(n.nodes, 12)
        for ch in children:
            lines.append(
                f"    [{ch.node_id}] {ch.title} (p{ch.start_index}-{ch.end_index})"
                + (f" — {ch.summary[:120]}" if ch.summary else "")
            )
        if len(n.nodes) > len(children):
            lines.append(f"    … 共 {len(n.nodes)} 个子节点（仅列出代表性 {len(children)} 个）")
    return "\n".join(lines)


def _leaves_of(n: Node) -> list[Node]:
    return [x for x, _ in iter_nodes([n]) if not x.nodes] or [n]


def _spread_sample(items: list[Node], k: int) -> list[Node]:
    """按文档顺序均匀取样，避免只看开头（根节点展开时会有几百个叶子）。"""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    step = (len(items) - 1) / (k - 1)
    out: list[Node] = []
    seen: set[str] = set()
    for i in range(k):
        n = items[round(i * step)]
        key = n.node_id or str(id(n))
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _dedupe(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    out = []
    for n in nodes:
        key = n.node_id or str(id(n))
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _read_group(n: Node, cap: int) -> list[Node]:
    leaves = _leaves_of(n)
    return leaves if len(leaves) <= cap else _spread_sample(leaves, cap)


def _pages_of(nodes: list[Node], total: int, cap: int) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for n in nodes:
        for p in range(n.start_index, n.end_index + 1):
            if 1 <= p <= total and p not in seen:
                seen.add(p)
                ordered.append(p)
                if len(ordered) >= cap:
                    return sorted(ordered)
    return sorted(ordered)


def _page_runs(pages: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for p in pages:
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def _read_context(kb: KB, doc_id: str, pages: list[int], page_chars: int) -> str:
    parts: list[str] = []
    for run in _page_runs(pages):
        texts = kb.read_pages(doc_id, run[0], run[-1])
        for off, p in enumerate(run):
            text = (texts[off] if off < len(texts) else "") or "（空页）"
            if len(text) > page_chars:
                text = text[:page_chars] + "…"
            parts.append(f"【第 {p} 页】\n{text}")
    return "\n\n".join(parts)


def _norm_id(value: str) -> str | None:
    value = value.strip().strip("[]【】()（）\"'")
    if not value:
        return None
    if value.lower().startswith("p") and value[1:].isdigit():
        return value[1:].zfill(4)
    if value.isdigit():
        return value.zfill(4) if len(value) < 4 else value
    m = _DIGITS.search(value)
    return m.group(0) if m else None


def _extract_ids(value: Any) -> list[str]:
    """把模型给的各种形状（list / dict / 逗号串 / 数字）统一成节点编号。"""
    if value is None:
        return []
    if isinstance(value, dict):
        keys = [k for k in value if _norm_id(str(k))]
        if keys:
            return [x for k in keys for x in _extract_ids(k)]
        return [x for v in value.values() for x in _extract_ids(v)]
    if isinstance(value, (list, tuple, set)):
        return [x for v in value for x in _extract_ids(v)]
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [str(value).zfill(4)]
    if isinstance(value, str):
        out: list[str] = []
        for token in _CITE_PARTS.split(value.strip()):
            nid = _norm_id(token)
            if nid:
                out.append(nid)
        return out
    return []


def _visible(frontier: list[Node]) -> list[Node]:
    out: list[Node] = []
    for n in frontier:
        out.append(n)
        out.extend(_spread_sample(n.nodes, 12))
    return out


def _localize(
    llm: LLM,
    model: str,
    tree: TreeDoc,
    question: str,
    frontier: list[Node],
    retries: int,
) -> tuple[list[str], list[str], str, int]:
    valid = {n.node_id for n in _visible(frontier) if n.node_id}
    base = [
        {"role": "system", "content": LOCALIZE_SYS},
        {
            "role": "user",
            "content": (
                f"文档：{tree.doc_name} —— {tree.doc_description}\n"
                f"问题：{question}\n\n"
                f"当前候选节点：\n{_listing(frontier)}"
            ),
        },
    ]
    messages = list(base)
    reason = ""
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            data = llm.chat_json(messages, model=model)
        except Exception as exc:  # 定位失败不能把整题拖垮，交给兜底取样
            reason = f"定位调用失败：{exc}"
            if attempt < retries:
                messages = list(base) + [
                    {
                        "role": "user",
                        "content": (
                            "上一次调用失败。请只输出 JSON，编号必须来自候选节点列表："
                            f"{', '.join(sorted(valid))}"
                        ),
                    }
                ]
            continue
        raw_expand = _extract_ids(data.get("expand"))
        raw_read = _extract_ids(data.get("read"))
        reason = str(data.get("reason", "") or "")
        invalid = [x for x in raw_expand + raw_read if x not in valid]
        expand = [x for x in raw_expand if x in valid]
        read = [x for x in raw_read if x in valid]
        if invalid:
            reason = f"{reason}（忽略无效编号：{', '.join(invalid)}）"
        if expand or read:
            return expand, read, reason, attempts
        if attempt < retries:
            messages = list(base) + [
                {
                    "role": "user",
                    "content": (
                        "上一次回复没有给出可用的 expand/read。只能从这些编号中选择："
                        f"{', '.join(sorted(valid))}。请重新输出 JSON，至少给出一个动作。"
                    ),
                }
            ]
    return [], [], reason, attempts


def _ancestor_map(structure: list[Node]) -> dict[str, list[Node]]:
    parents: dict[str, list[Node]] = {}

    def walk(nodes: list[Node], chain: list[Node]) -> None:
        for n in nodes:
            parents[n.node_id] = chain
            walk(n.nodes, chain + [n])

    walk(structure, [])
    return parents


def _summary_context(tree: TreeDoc, read_nodes: list[Node], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    parents = _ancestor_map(tree.structure)
    picked: list[tuple[Node, list[Node]]] = []
    seen: set[str] = set()
    for root in tree.structure:
        if root.summary and root.node_id not in seen:
            seen.add(root.node_id)
            picked.append((root, []))
    for n in read_nodes:
        chain = parents.get(n.node_id, [])
        for anc in chain + [n]:
            if anc.summary and anc.node_id not in seen:
                seen.add(anc.node_id)
                picked.append((anc, parents.get(anc.node_id, [])))
    lines: list[str] = []
    used = 0
    for n, chain in picked:
        path = " / ".join([x.title for x in chain] + [n.title])
        line = f"[{n.node_id}] {path}（p{n.start_index}-{n.end_index}）：{n.summary[:1200]}"
        room = max_chars - used
        if room <= 0:
            break
        if len(line) > room:
            lines.append(line[:room] + "…")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _parse_citations(
    raw_pages: Any, answer: str, read_pages: list[int]
) -> tuple[list[int], str]:
    allowed = set(read_pages)
    model_pages: list[int] = []
    for p in _extract_ids(raw_pages):
        try:
            model_pages.append(int(p))
        except (TypeError, ValueError):
            continue
    model_pages = [p for p in model_pages if p in allowed]
    if model_pages:
        return list(dict.fromkeys(model_pages)), "model"

    text_pages: list[int] = []
    for m in re.finditer(r"[\[【]([^\]】]{1,80})[\]】]", answer or ""):
        for part in _CITE_PARTS.split(m.group(1).strip()):
            hit = _CITE_ITEM.match(part)
            if not hit:
                continue
            a = int(hit.group(1))
            b = int(hit.group(2) or a)
            if b < a:
                a, b = b, a
            text_pages.extend(range(a, b + 1) if b - a <= 20 else [a, b])
    text_pages = [p for p in text_pages if p in allowed]
    if text_pages:
        return list(dict.fromkeys(text_pages)), "text"
    return [], "none"


def _expand_retry(read_nodes: list[Node], frontier: list[Node], cap: int) -> list[Node]:
    extra = list(read_nodes)
    seen = {n.node_id or str(id(n)) for n in extra}
    pool: list[Node] = []
    for n in frontier:
        for leaf in _leaves_of(n):
            key = leaf.node_id or str(id(leaf))
            if key not in seen:
                seen.add(key)
                pool.append(leaf)
    extra.extend(_spread_sample(pool, max(0, cap - len(extra))))
    return _dedupe(extra)[:cap]


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
    read_cap = 8

    frontier = list(tree.structure)
    read_nodes: list[Node] = []
    for rnd in range(1, int(qcfg["max_rounds"]) + 1):
        if not frontier:
            break
        expand, read, reason, attempts = _localize(
            llm,
            chat_model,
            tree,
            question,
            frontier,
            int(qcfg.get("locate_retries", 1)),
        )
        trace.append(
            {
                "round": rnd,
                "expand": expand,
                "read": read,
                "reason": reason,
                "attempts": attempts,
            }
        )
        got = False
        for nid in read:
            n = find_node(tree.structure, nid)
            if n:
                read_nodes.extend(_read_group(n, read_cap))
                got = True
        if got:
            break
        next_frontier: list[Node] = []
        for nid in expand:
            n = find_node(tree.structure, nid)
            if n is None:
                continue
            if n.nodes:
                next_frontier.extend(n.nodes)
            else:
                read_nodes.extend(_read_group(n, read_cap))
        if not next_frontier:
            break
        frontier = _spread_sample(next_frontier, 10)

    fallback = "none"
    if not read_nodes:
        leaves = [n for n, _ in iter_nodes(tree.structure) if not n.nodes]
        read_nodes = _spread_sample(leaves, read_cap)
        fallback = "spread_sample" if read_nodes else "summary_only"
    read_nodes = _dedupe(read_nodes)[:read_cap]
    pages = _pages_of(read_nodes, total, int(qcfg["max_pages"]))
    if not pages and fallback == "none":
        fallback = "summary_only"

    summary_ctx = _summary_context(
        tree, read_nodes, int(qcfg.get("summary_chars", 6000))
    )
    node_line = "; ".join(
        f"{n.title}(p{n.start_index}-{n.end_index})" for n in read_nodes
    )
    answer_attempts = max(1, int(qcfg.get("answer_attempts", 3)))
    answer_data: dict = {}
    nudge = ""
    expanded = False
    for attempt in range(1, answer_attempts + 1):
        context = _read_context(kb, doc_id, pages, int(qcfg["page_chars"]))
        user = f"问题：{question}\n\n"
        if summary_ctx:
            user += (
                "节点摘要（用于定位与全局概览，不能作为页码引用依据）：\n"
                f"{summary_ctx}\n\n"
            )
        user += f"文档页原文（节点：{node_line or '无'}）：\n\n{context or '（本次未读取到原文页）'}"
        if nudge:
            user += "\n\n" + nudge
        data = llm.chat_json(
            [
                {"role": "system", "content": ANSWER_SYS},
                {"role": "user", "content": user},
            ],
            model=chat_model,
        )
        answer_data = data
        answer = str(data.get("answer", "") or "").strip()
        sufficient = bool(data.get("sufficient", True))
        if answer and sufficient:
            break
        if attempt >= answer_attempts:
            break
        if answer and not sufficient and not expanded:
            read_nodes = _expand_retry(read_nodes, frontier, 12)
            pages = _pages_of(read_nodes, total, int(qcfg["max_pages"]))
            expanded = True
            nudge = (
                "上一次判定证据不足。上面已扩大读取范围；请基于全部已读页与节点摘要"
                "重新作答，给出结论并在确实无法回答时说明缺少什么。"
            )
        else:
            nudge = (
                "上一次 answer 为空。必须输出非空 answer；若证据不足，"
                "也请明确说明缺少什么信息并把 sufficient 设为 false。"
            )

    answer = str(answer_data.get("answer", "") or "").strip()
    if answer:
        cited, cite_source = _parse_citations(answer_data.get("pages"), answer, pages)
    else:
        cited, cite_source = [], "none"
    return {
        "doc_id": doc_id,
        "answer": answer,
        "answer_status": "ok" if answer else "empty",
        "citations": [{"doc_id": doc_id, "pages": cited}],
        "citation_source": cite_source,
        "read_pages": pages,
        "read_nodes": [
            {"id": n.node_id, "title": n.title, "pages": [n.start_index, n.end_index]}
            for n in read_nodes
        ],
        "fallback": fallback,
        "rounds": trace,
        "usage": llm.usage(),
    }
