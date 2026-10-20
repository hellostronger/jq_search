from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class Node(BaseModel):
    title: str
    start_index: int = 1
    end_index: int = 1
    nodes: list["Node"] = Field(default_factory=list)
    node_id: str = ""
    summary: str = ""


class TreeDoc(BaseModel):
    doc_id: str
    doc_name: str
    doc_description: str = ""
    user_description: str = ""
    structure: list[Node] = Field(default_factory=list)


def iter_nodes(nodes: list[Node], parent: Node | None = None) -> Iterator[tuple[Node, Node | None]]:
    for n in nodes:
        yield n, parent
        yield from iter_nodes(n.nodes, n)


def find_node(structure: list[Node], node_id: str) -> Node | None:
    for n, _ in iter_nodes(structure):
        if n.node_id == node_id:
            return n
    return None


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------- flash 建树

def _collect_headings(pages: list[str]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for i, text in enumerate(pages):
        for line in text.splitlines():
            m = HEADING_RE.match(line.strip())
            if m:
                headings.append((i + 1, len(m.group(1)), m.group(2).strip()))
    counts = Counter((lv, t) for _, lv, t in headings)
    # 反复出现的同名同级标题多为页眉/页脚，剔除
    return [(p, lv, t) for p, lv, t in headings if counts[(lv, t)] <= 3]


def flash_skeleton(pages: list[str], min_headings: int = 3) -> list[Node] | None:
    """从 MinerU 标题层级直接构建树骨架（不调用 LLM）。无结构时返回 None。"""
    headings = _collect_headings(pages)
    if len(headings) < min_headings:
        return None
    root = Node(title="__root__", start_index=1, end_index=len(pages))
    stack: list[tuple[int, Node]] = [(0, root)]
    for page, level, title in headings:
        while stack and stack[-1][0] >= level:
            stack.pop()
        node = Node(title=title[:120], start_index=page, end_index=page)
        stack[-1][1].nodes.append(node)
        stack.append((level, node))

    flat = [n for n, _ in iter_nodes(root.nodes)]
    for idx, node in enumerate(flat):
        if not node.nodes:
            if idx + 1 < len(flat):
                nxt = flat[idx + 1].start_index
                node.end_index = max(node.start_index, min(nxt - 1, len(pages)))
            else:
                node.end_index = len(pages)
    _fix_parents(root)
    return root.nodes


def _fix_parents(node: Node) -> None:
    if not node.nodes:
        return
    for ch in node.nodes:
        _fix_parents(ch)
    node.end_index = max([node.start_index] + [ch.end_index for ch in node.nodes])


# -------------------------------------------------------------- classic 建树

CLASSIC_SECTION_SYS = "你是专业的文档结构分析师。给文档片段划分章节。只输出 JSON。"
CLASSIC_GROUP_SYS = "你是专业的文档结构分析师。把扁平章节列表组织成层次树。只输出 JSON。"
SUMMARY_SYS = "你是精准的文档摘要器。只输出 JSON。"


def classic_skeleton(
    pages: list[str],
    llm: Any,
    model: str,
    batch_size: int = 8,
    stage: Callable[[str], None] | None = None,
) -> list[Node]:
    sections: list[dict] = []
    for b in range(0, len(pages), batch_size):
        chunk = pages[b : b + batch_size]
        if stage:
            stage(f"classic 建树：处理第 {b + 1}-{b + len(chunk)} 页 …")
        body = "\n\n".join(
            f"【第 {b + i + 1} 页】\n{_truncate(p, 1800)}" for i, p in enumerate(chunk)
        )
        data = llm.chat_json(
            [
                {"role": "system", "content": CLASSIC_SECTION_SYS},
                {
                    "role": "user",
                    "content": (
                        f"{body}\n\n请划分该片段的章节（扁平列表、按顺序），"
                        f"输出 JSON：{{\"sections\":[{{\"title\":\"章节标题\",\"start\":页码,\"end\":页码}}]}}，"
                        f"页码为绝对页码（{b + 1} 到 {b + len(chunk)}）。"
                    ),
                },
            ],
            model=model,
            max_tokens=2500,
        )
        for s in data.get("sections") or []:
            try:
                start, end = int(s["start"]), int(s["end"])
            except (KeyError, TypeError, ValueError):
                continue
            title = str(s.get("title", "")).strip()[:120]
            if title and b + 1 <= start <= end <= b + len(chunk):
                sections.append({"title": title, "start": start, "end": end})
    if not sections:
        return [Node(title="全文", start_index=1, end_index=len(pages))]

    if stage:
        stage("classic 建树：归组层次 …")
    listing = "\n".join(
        f"{i}. {s['title']} (p{s['start']}-{s['end']})"
        for i, s in enumerate(sections, 1)
    )
    data = llm.chat_json(
        [
            {"role": "system", "content": CLASSIC_GROUP_SYS},
            {
                "role": "user",
                "content": (
                    f"扁平章节列表：\n{listing}\n\n将它们组织为层次树（父节点覆盖子节点页范围），"
                    "只输出 JSON：{\"nodes\":[{\"title\":\"...\",\"start\":n,\"end\":n,\"nodes\":[...]}]}，"
                    "叶子即上面的章节，页范围不得超出其内容。"
                ),
            },
        ],
        model=model,
        max_tokens=3500,
    )
    structure = coerce_nodes(data.get("nodes"), len(pages))
    if not structure:
        structure = [
            Node(title=s["title"], start_index=s["start"], end_index=s["end"])
            for s in sections
        ]
    return structure


def coerce_nodes(raw: Any, total_pages: int) -> list[Node]:
    out: list[Node] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()[:120]
        if not title:
            continue
        try:
            start = max(1, int(item.get("start", item.get("start_index", 1))))
            end = min(total_pages, int(item.get("end", item.get("end_index", start))))
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        children = coerce_nodes(item.get("nodes") or item.get("children"), total_pages)
        for ch in children:
            start = min(start, ch.start_index)
            end = max(end, ch.end_index)
        out.append(Node(title=title, start_index=start, end_index=end, nodes=children))
    return out


# ---------------------------------------------------------------- 摘要与描述

def _fallback_rollup(n: Node) -> str:
    parts = []
    for ch in n.nodes:
        text = _truncate(ch.summary, 160)
        parts.append(f"{ch.title}：{text}" if text else ch.title)
    return _truncate("；".join(parts), 1200)


def add_summaries(
    structure: list[Node],
    pages: list[str],
    llm: Any,
    model: str,
    workers: int = 6,
    stage: Callable[[str], None] | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    """生成叶摘要与父节点摘要。失败会重试并返回失败清单，不静默留空。"""
    report: dict[str, Any] = {"leaves": 0, "leaf_failures": [], "rollup_failures": []}
    leaves = [n for n, _ in iter_nodes(structure) if not n.nodes]
    report["leaves"] = len(leaves)
    if not leaves:
        return report
    if stage:
        stage(f"生成 {len(leaves)} 个叶节点摘要 …")

    def one(n: Node) -> dict | None:
        body = "\n\n".join(
            f"【第 {i} 页】\n{_truncate(pages[i - 1], 2500)}"
            for i in range(n.start_index, n.end_index + 1)
            if 1 <= i <= len(pages)
        )
        if not body.strip():
            n.summary = ""
            return None
        messages = [
            {"role": "system", "content": SUMMARY_SYS},
            {
                "role": "user",
                "content": (
                    f"节点「{n.title}」（第 {n.start_index}-{n.end_index} 页）内容：\n{body}\n\n"
                    "用 3-6 条要点总结该节的关键事实与数据（保留数字、名称等具体信息）。"
                    "输出 JSON：{\"summary\":\"...\"}"
                ),
            },
        ]
        last_error = ""
        for attempt in range(retries + 1):
            try:
                data = llm.chat_json(messages, model=model, max_tokens=900)
                summary = str(data.get("summary", "") or "").strip()[:2000]
                if summary:
                    n.summary = summary
                    return None
                last_error = "摘要为空"
            except Exception as exc:
                last_error = str(exc)
            if attempt < retries:
                messages = messages + [
                    {"role": "user", "content": "上一次没有给出有效的 summary。请重新输出 JSON。"}
                ]
        n.summary = ""
        return {
            "id": n.node_id,
            "title": n.title,
            "pages": [n.start_index, n.end_index],
            "error": last_error,
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        failures = list(ex.map(one, leaves))
    report["leaf_failures"] = [f for f in failures if f]
    if stage and report["leaf_failures"]:
        ids = ", ".join(f["id"] for f in report["leaf_failures"][:8])
        stage(f"⚠ {len(report['leaf_failures'])} 个叶节点摘要失败（{ids}）")

    rollup_failures: list[dict] = []

    def rollup(n: Node) -> None:
        if not n.nodes:
            return
        for ch in n.nodes:
            rollup(ch)
        parts = [
            f"- {ch.title}: {_truncate(ch.summary, 300) or '（无摘要）'}"
            for ch in n.nodes
        ]
        messages = [
            {"role": "system", "content": SUMMARY_SYS},
            {
                "role": "user",
                "content": (
                    f"父节点「{n.title}」的子节点摘要：\n" + "\n".join(parts) +
                    "\n\n用 2-4 句话概括该节点。输出 JSON：{\"summary\":\"...\"}"
                ),
            },
        ]
        last_error = ""
        for attempt in range(retries + 1):
            try:
                data = llm.chat_json(messages, model=model, max_tokens=600)
                summary = str(data.get("summary", "") or "").strip()[:1200]
                if summary:
                    n.summary = summary
                    return
                last_error = "摘要为空"
            except Exception as exc:
                last_error = str(exc)
            if attempt < retries:
                messages = messages + [
                    {"role": "user", "content": "上一次没有给出有效的 summary。请重新输出 JSON。"}
                ]
        n.summary = _fallback_rollup(n)
        rollup_failures.append(
            {
                "id": n.node_id,
                "title": n.title,
                "error": last_error,
                "fallback": True,
            }
        )

    for top in structure:
        rollup(top)
    report["rollup_failures"] = rollup_failures
    if stage and report["rollup_failures"]:
        stage(f"⚠ {len(report['rollup_failures'])} 个父节点摘要失败（已用子标题兜底）")
    return report


def describe_doc(
    pages: list[str],
    structure: list[Node],
    llm: Any,
    model: str,
    hint: str = "",
    retries: int = 1,
) -> str:
    head = "\n\n".join(
        f"【第 {i} 页】\n{_truncate(pages[i - 1], 1500)}"
        for i in range(1, min(4, len(pages)) + 1)
    )
    skel = skeleton_text(structure, max_chars=3000)
    extra = f"用户补充说明：{hint}\n\n" if hint else ""
    messages = [
        {"role": "system", "content": SUMMARY_SYS},
        {
            "role": "user",
            "content": (
                f"{extra}文档开头：\n{head}\n\n目录：\n{skel}\n\n"
                "用 1-3 句话描述该文档的主题与内容。输出 JSON：{\"description\":\"...\"}"
            ),
        },
    ]
    for attempt in range(retries + 1):
        try:
            data = llm.chat_json(messages, model=model, max_tokens=300)
            desc = str(data.get("description", "") or "").strip()[:800]
            if desc:
                return desc
        except Exception:
            pass
        if attempt < retries:
            messages = messages + [
                {"role": "user", "content": "上一次没有给出有效的 description。请重新输出 JSON。"}
            ]
    return ""


# ---------------------------------------------------------------- 渲染工具

def assign_ids(structure: list[Node]) -> None:
    counter = {"n": 0}

    def walk(n: Node) -> None:
        counter["n"] += 1
        n.node_id = f"{counter['n']:04d}"
        for ch in n.nodes:
            walk(ch)

    for top in structure:
        walk(top)


def skeleton_text(
    structure: list[Node], with_summary: bool = False, max_chars: int = 15000
) -> str:
    lines: list[str] = []

    def walk(nodes: list[Node], depth: int) -> None:
        for n in nodes:
            lines.append(f"{'  ' * depth}[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index})")
            if with_summary and n.summary:
                lines.append(f"{'  ' * (depth + 1)}摘要: {_truncate(n.summary, 300)}")
            walk(n.nodes, depth + 1)

    walk(structure, 0)
    return "\n".join(lines)[:max_chars]
