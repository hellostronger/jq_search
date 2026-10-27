from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..core.entities import EntityIndex, normalize
from ..core.llm import LLM
from ..core.store import KB
from ..core.tree import Node, TreeDoc, find_node, iter_nodes
from ..core.wiki import WikiPage, WikiStore, render_body

LOCALIZE_SYS = """你是文档检索专家，正在一棵文档树索引中定位答案位置（类似人类专家翻目录）。
给你问题、文档描述和当前候选节点（含编号、标题、页范围、摘要及子节点）。
选择下一步动作，只输出 JSON：
{"expand": ["节点编号"], "read": ["节点编号"], "reason": "简短理由"}
- "expand"：进入某分支，返回其子节点
- "read"：读取该节点覆盖页的原文；叶节点直接读，中间节点会读取其代表性叶子
- 必须至少给出一个动作，编号必须原样来自候选节点列表，禁止臆造
- 若另列了「字面命中节点」，其编号同样可选，通常是问题关键词在原文里直接命中的最直接证据
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
_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9+.\-_]{1,}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


def anchor_terms(question: str) -> list[str]:
    """取问题里最有区分度的字面锚词：优先拉丁词（实体名通常是拉丁文）；
    纯中文问题只取 4 字以上片段，过短的词（"风险""溢价"）太泛，不做锚词。"""
    latin = [t.lower() for t in _LATIN_TERM.findall(question) if len(t) >= 3]
    cjk = [t for t in _CJK_RUN.findall(question) if len(t) >= 4]
    return list(dict.fromkeys(latin or cjk))[:4]


def anchor_groups(
    terms: list[str], index: EntityIndex | None, limit: int = 6
) -> list[list[str]]:
    """把每个锚词扩成一组别名（同义形式之间是 OR，组与组之间仍是 AND）。
    实体索引缺失时原样返回单元素组，行为与未接入前一致。"""
    groups: list[list[str]] = []
    for term in terms:
        names = [term]
        if index is not None:
            names.extend(index.expand(term, limit=limit))
        seen: set[str] = set()
        group: list[str] = []
        for name in names:
            low = (name or "").strip().lower()
            if len(low) < 2 or low in seen or low in _ANCHOR_STOP:
                continue
            seen.add(low)
            group.append(low)
        groups.append(group or [term.lower()])
    return groups


# 查询锚词的通用词黑名单：这些词在新闻语料里到处都有，当锚词只会把 gold 页冲掉。
# 它们仍可留在实体索引里供浏览，但不参与「用实体名替换长锚词」这一步。
_ANCHOR_STOP = frozenset(
    """
    突破 下跌 上涨 下降 增长 增加 减少 回落 回升 走高 走低 升至 降至 涨至 跌至 达到 超过 低于
    合计 分别 分别为 累计 实现 显示 预计 出现 发生 成为 包括 位于 来自 相比 同比 环比 季度 年度 月份
    全年 本月 上月 本周 上周 当日 期间 以来 同期 目前 最新 全球 中国 美国 科技 股份 市场 行业
    企业 公司 集团 银行 证券 基金 机构 部门 数据 报告 消息 公告 记者 报道 分析 研究 投资 融资
    业务 战略 计划 项目 原因 结果 影响 方式 价格 成本 收入 营收 利润 市值 销量 规模 占比 股价
    业绩 产能 订单 份额 目标 增速 利率 收益率 概率 万亿 亿元 美元 港元 个月 年期 ai etf st
    一度 我国 上市 销售 比例 金额 工业 商品 电子 年化 数量 完成 提升 涨幅 预测
    什么 怎么 如何 为什么 多少 哪些 哪个 哪家 怎样 怎么样 什么样 怎么办 什么意思
    什么时候 是否 有没有 能不能 会不会
    """.split()
)
# 以谓词收尾的短复合串（同比增长、收入增长）也是通用词，只挡四字以内的
_ANCHOR_PREDICATES = (
    "增长", "下降", "上涨", "下跌", "上升", "回落", "突破", "合计", "分别",
    "达到", "超过", "升至", "降至", "实现", "显示", "预计",
)


def _usable_anchor(entity) -> bool:
    """实体名能否当查询锚词：太泛、太短或只是度量指标时，不如保留原长锚词。

    只按「名字本身是否通用」过滤，不按频次或来源删短词：实测在可重放语料上
    按频次删两字 n-gram 会把真实主体一起删掉（召回 -13pp，精度也下降）。
    """
    name = normalize(entity.canonical)
    if len(name) < 2 or entity.type == "METRIC":
        return False
    if name in _ANCHOR_STOP:
        return False
    if any(name.endswith(word) and len(name) <= len(word) + 2 for word in _ANCHOR_PREDICATES):
        return False
    return True


def question_anchors(
    question: str, index: EntityIndex | None, limit: int = 4
) -> list[str]:
    """问题锚词：优先用实体索引里命中的已知名称。

    中文问题常把实体和谓语连成一段（"宁德时代新能源科技股份有限公司的收入如何"），
    按字符长度切出来的整段既不是实体也命中不了原文；命中实体时用实体名替换掉
    包含它的长片段，后续再按别名组召回。
    """
    base = anchor_terms(question)
    if index is None or not index.entities:
        return base

    norm_q = normalize(question)
    hits: list[tuple[int, str]] = []
    for entity in index.entities:
        if not _usable_anchor(entity):
            continue
        min_len = 2 if entity.frequency >= 3 else 3
        best = ""
        for name in (entity.canonical, *entity.aliases):
            key = normalize(name)
            if len(key) >= min_len and key in norm_q and len(key) > len(normalize(best)):
                best = name
        if best:
            hits.append((len(normalize(best)), best))
    if not hits:
        return base

    hits.sort(key=lambda item: (-item[0], item[1]))
    found: list[str] = []
    seen: set[str] = set()
    for _, name in hits:
        key = normalize(name)
        if not key or key in seen or key in _ANCHOR_STOP:
            continue
        # 长名优先：iPad 命中时，同串里的 Pad / AD / IP 不应各自再当锚词，
        # 否则组间 AND 会把本来能中的页冲掉。
        if any(key in normalize(longer) for longer in found):
            continue
        seen.add(key)
        found.append(name)
    base = [t for t in base if not any(normalize(f) in normalize(t) for f in found)]
    return list(dict.fromkeys(found + base))[:limit]


def _as_groups(terms: list[str] | list[list[str]]) -> list[list[str]]:
    groups: list[list[str]] = []
    for item in terms:
        if isinstance(item, str):
            if item:
                groups.append([item])
            continue
        group = [t for t in item if t]
        if group:
            groups.append(group)
    return groups


def _window(text: str, terms: list[str], limit: int) -> str:
    """摘要窗口：命中锚词的句子必须落在窗口内，否则长摘要里的实体会被截掉。"""
    if len(text) <= limit:
        return text
    low = text.lower()
    positions = [low.find(t) for t in terms if t and low.find(t) >= 0]
    start = 0
    if positions:
        start = min(max(min(positions) - limit // 3, 0), len(text) - limit)
    end = start + limit
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")


def _lexical_pages(
    texts: list[str], terms: list[str] | list[list[str]], cap: int
) -> list[int]:
    """返回命中锚词的页：组内任一别名命中即可，组间必须全部命中。"""
    groups = _as_groups(terms)
    if not groups or cap <= 0:
        return []
    scored: list[tuple[int, int]] = []
    for idx, text in enumerate(texts, start=1):
        low = text.lower()
        counts = [sum(low.count(t) for t in group) for group in groups]
        if all(c > 0 for c in counts):
            scored.append((idx, sum(counts)))
    ranked = sorted(scored, key=lambda x: (-x[1], x[0]))
    return sorted(page for page, _ in ranked[:cap])


def _nodes_for_pages(leaves: list[Node], pages: list[int], cap: int) -> list[Node]:
    """把命中页映射到叶节点。按文档顺序均匀取样：事件类问题的证据常散落在
    时间线两端（首发与后续反转），只取命中最多的节点会丢掉尾部进展。"""
    hit = set(pages)
    picked = [
        n for n in leaves if any(n.start_index <= p <= n.end_index for p in hit)
    ]
    return _spread_sample(picked, cap)


def _listing(nodes: list[Node], terms: list[str] | None = None) -> str:
    terms = terms or []
    lines: list[str] = []
    for n in nodes:
        line = f"[{n.node_id}] {n.title} (p{n.start_index}-{n.end_index})"
        if n.summary:
            line += f" — {_window(n.summary, terms, 200)}"
        lines.append(line)
        children = _spread_sample(n.nodes, 12)
        for ch in children:
            lines.append(
                f"    [{ch.node_id}] {ch.title} (p{ch.start_index}-{ch.end_index})"
                + (f" — {_window(ch.summary, terms, 120)}" if ch.summary else "")
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
    terms: list[str] | None = None,
    hint: list[Node] | None = None,
) -> tuple[list[str], list[str], str, int]:
    terms = terms or []
    hint = hint or []
    valid = {n.node_id for n in _visible(frontier) if n.node_id}
    valid |= {n.node_id for n in hint if n.node_id}
    content = (
        f"文档：{tree.doc_name} —— {tree.doc_description}\n"
        f"问题：{question}\n\n"
        f"当前候选节点：\n{_listing(frontier, terms)}"
    )
    if hint:
        content += (
            "\n\n字面命中节点（问题关键词在原文里直接命中，优先 read）：\n"
            f"{_listing(hint, terms)}"
        )
    base = [
        {"role": "system", "content": LOCALIZE_SYS},
        {"role": "user", "content": content},
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


def _summary_context(
    tree: TreeDoc, read_nodes: list[Node], max_chars: int, terms: list[str] | None = None
) -> str:
    if max_chars <= 0:
        return ""
    terms = terms or []
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
        line = f"[{n.node_id}] {path}（p{n.start_index}-{n.end_index}）：{_window(n.summary, terms, 1200)}"
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


# 多轮追问的省略式开头：这些词说明主体在上文
_FOLLOWUP_HEADS = ("那么", "那", "它", "其", "该", "这", "上述", "前面", "刚才", "这些")
_SUBJECT_STOP = (
    "销量", "营收", "收入", "净利", "利润", "售价", "价格", "定价", "出货量", "交付",
    "市值", "股价", "份额", "占比", "规模", "增速", "涨幅", "跌幅", "成本", "费用",
    "产能", "利率", "毛利", "业绩", "融资", "募资", "数量", "金额", "总额", "余额",
)


def question_subject(question: str) -> str:
    """取问句开头的主体（公司/产品名），用于把省略式追问补全。

    只在第一个数字、拉丁词或时间词处截断，例如
    "赛力斯汽车9月批发销量为多少" → "赛力斯汽车"。
    """
    q = re.sub(r"[？?]\s*$", "", (question or "").strip())
    q = re.sub(r"(?:多少|是多少|是什么|怎么样|呢)$", "", q).strip()
    q = re.sub(r"^[^\u4e00-\u9fffA-Za-z]+", "", q)
    m = re.match(r"[A-Za-z][A-Za-z0-9+.\-_ ]*", q)
    if m:
        return m.group(0).strip()
    m = re.match(
        r"[\u4e00-\u9fff]{2,}?(?=\d|[A-Za-z]|第[一二三四]季度|上半年|下半年|全年|本季度|当季|的|，|,)",
        q,
    )
    if m:
        return m.group(0)
    cut = len(q)
    for word in _SUBJECT_STOP:
        i = q.find(word)
        if 0 < i < cut:
            cut = i
    subject = q[:cut].strip("的，, ")
    return subject if 2 <= len(subject) <= 12 else ""


def resolve_followup(question: str, history: list[dict] | None) -> str:
    """把省略主语的追问改写成自足问句：从上一轮问题里取主体补在前面。

    "赛力斯汽车9月批发销量多少？" → "那零售销量是多少？"
    改写为 "赛力斯汽车零售销量是多少？"。只有以指代词开头时才改写，
    正常的新问题原样返回，避免污染单轮评测。
    """
    q = (question or "").strip()
    if not q or not history:
        return q
    head = next((h for h in _FOLLOWUP_HEADS if q.startswith(h)), "")
    if not head:
        return q
    prev = next(
        (h.get("content") or "" for h in reversed(history) if h.get("role") == "user"), ""
    )
    subject = question_subject(prev)
    if not subject:
        return q
    core = q[len(head) :].lstrip("的，, ")
    return f"{subject}{core}" if core else q


def query_raw(
    doc_id: str,
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    history: list[dict] | None = None,
) -> dict:
    """PageIndex 式树遍历问答（raw 路）：定位 → 读页 → 带页级引用作答。

    history 为之前轮次的 {"role","content"} 列表；追问会先补全主体再检索。
    """
    resolved = resolve_followup(question, history)
    question = resolved
    tree: TreeDoc = kb.load_tree(doc_id)
    total = kb.page_count(doc_id)
    qcfg = cfg["query"]
    chat_model = cfg["llm"]["chat_model"]
    trace: list[dict] = []
    read_cap = 8

    ecfg = cfg.get("entities", {})
    index = kb.load_entities() if ecfg.get("query_expansion", True) else None
    terms = question_anchors(question, index)
    groups = anchor_groups(terms, index, int(ecfg.get("expand_limit", 6)))
    window_terms = list(dict.fromkeys(t for group in groups for t in group))
    lex_pages = _lexical_pages(
        kb.read_pages(doc_id, 1, total), groups, int(qcfg.get("lexical_pages", 12))
    )
    lex_nodes = _nodes_for_pages(
        [n for n, _ in iter_nodes(tree.structure) if not n.nodes], lex_pages, read_cap
    )

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
            window_terms,
            lex_nodes,
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
    if not read_nodes and not lex_nodes:
        leaves = [n for n, _ in iter_nodes(tree.structure) if not n.nodes]
        read_nodes = _spread_sample(leaves, read_cap)
        fallback = "spread_sample" if read_nodes else "summary_only"
    read_nodes = _dedupe(lex_nodes + read_nodes)[:read_cap]
    pages = _pages_of(read_nodes, total, int(qcfg["max_pages"]))
    if not pages and fallback == "none":
        fallback = "summary_only"

    summary_ctx = _summary_context(
        tree, read_nodes, int(qcfg.get("summary_chars", 6000)), window_terms
    )
    node_line = "; ".join(
        f"{n.title}(p{n.start_index}-{n.end_index})" for n in read_nodes
    )
    answer_attempts = max(1, int(qcfg.get("answer_attempts", 3)))
    history_ctx = ""
    if history:
        lines = []
        for h in history[-4:]:
            role = "用户" if h.get("role") == "user" else "助手"
            content = str(h.get("content") or "").strip()
            if content:
                lines.append(f"{role}：{content[:200]}")
        if lines:
            history_ctx = "前几轮对话（仅用于理解指代，页码仍以本次读到的原文为准）：\n" + "\n".join(lines) + "\n\n"
    answer_data: dict = {}
    nudge = ""
    expanded = False
    for attempt in range(1, answer_attempts + 1):
        context = _read_context(kb, doc_id, pages, int(qcfg["page_chars"]))
        user = f"问题：{question}\n\n"
        if history_ctx:
            user = history_ctx + user
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
        "mode": "raw",
        "doc_id": doc_id,
        "question": question,
        "resolved_question": resolved,
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
        "lexical": {
            "terms": terms,
            "groups": groups,
            "pages": lex_pages,
            "nodes": [n.node_id for n in lex_nodes],
        },
        "rounds": trace,
        "usage": llm.usage(),
    }


# --------------------------------------------------------------------- wiki 路

WIKI_SELECT_SYS = """你是知识库检索专家。给你问题与 wiki 页面录（实体/主题/来源/分析页），选出最相关的页面。
只输出 JSON：
{{"pages": ["页面id"], "sufficient": true, "reason": "简短理由"}}
- 最多选 {limit} 页，按相关度排序；优先能直接回答问题的页面。
- 若目录中没有任何页面能支撑回答，sufficient 设为 false，pages 为空。
- 页面 id 必须来自目录，不得编造。"""

WIKI_ANSWER_SYS = """你是严谨的知识库问答助手。给你的 wiki 页是对来源文档的蒸馏，事实后标注了来源文档与页码。
只能依据这些 wiki 页回答；输出 JSON：
{"answer": "答案，引用处标注 [文档id:p页码]，例如：收入 247 亿美元 [src_0001:p1]", "citations": [{"doc": "src_0001", "pages": [1]}], "sufficient": true}
如果 wiki 内容不足以回答（缺数据、粒度太粗、页标记 stale），sufficient 设为 false，并在 answer 中说明缺少什么。禁止编造。"""

COMPARE_SYS = """你在核对两条问答结果：一条来自 wiki 蒸馏层，一条来自原始文档页。
只输出 JSON：{"consistent": true, "note": "一致或关键差异（50 字内）"}
- 数字、日期、主体、结论一致才算一致；表述差异不算冲突。"""

_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _wiki_listing(pages: list[WikiPage], doc_id: str | None) -> str:
    lines: list[str] = []
    for p in pages[:120]:
        scope = "来源" if p.type == "source" else (",".join(p.sources[:3]) or "—")
        summary = (p.summary or p.description or "").replace("\n", " ")
        line = f"[{p.id}] ({p.type} | {scope}) {p.title}"
        if summary:
            line += f" — {summary[:180]}"
        lines.append(line)
    if len(pages) > 120:
        lines.append(f"… 另有 {len(pages) - 120} 页未列出")
    if doc_id:
        lines.insert(0, f"（当前问题聚焦文档：{doc_id}）")
    return "\n".join(lines)


def _wiki_scope(store: WikiStore, doc_id: str | None) -> list[WikiPage]:
    pages = store.list()
    if not doc_id:
        return pages
    scoped = [p for p in pages if doc_id in p.sources or p.id == doc_id]
    return scoped if len(scoped) >= 3 else pages


def _load_wiki_pages(
    store: WikiStore, ids: list[str], limit: int, doc_id: str | None
) -> list[WikiPage]:
    out: list[WikiPage] = []
    seen: set[str] = {p for p in ids}
    for pid in ids:
        if len(out) >= limit:
            break
        try:
            out.append(store.load(pid))
        except FileNotFoundError:
            continue
    # 沿 [[互链]] 扩一跳，优先同文档页
    for page in list(out):
        for link in _LINK_RE.findall(page.body or ""):
            if len(out) >= limit:
                break
            if link in seen or link == page.id:
                continue
            try:
                other = store.load(link)
            except FileNotFoundError:
                continue
            seen.add(link)
            out.append(other)
    if doc_id:
        out.sort(key=lambda p: 0 if doc_id in p.sources else 1)
    return out


def _wiki_context(pages: list[WikiPage], page_chars: int) -> str:
    parts = []
    for p in pages:
        body = p.body.strip() or render_body(p)
        if len(body) > page_chars:
            body = body[:page_chars] + "…"
        head = f"【wiki 页 {p.id} | {p.type} | {p.title}】"
        if p.stale:
            head += "（stale：内容可能过时）"
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def _cite_pool(pages: list[WikiPage]) -> dict[str, set[int]]:
    pool: dict[str, set[int]] = {}
    for p in pages:
        for c in p.citations:
            doc = str(c.get("doc") or "").strip()
            if not doc:
                continue
            pages_set = pool.setdefault(doc, set())
            for x in c.get("pages") or []:
                try:
                    pages_set.add(int(x))
                except (TypeError, ValueError):
                    continue
    return pool


_PAGE_REF_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\s*:\s*p(\d+)")


def _cites_from_text(text: str, pool: dict[str, set[int]]) -> list[dict]:
    out: dict[str, set[int]] = {}
    for doc, page in _PAGE_REF_RE.findall(text or ""):
        n = int(page)
        if doc in pool and n in pool[doc]:
            out.setdefault(doc, set()).add(n)
    return [{"doc_id": d, "pages": sorted(ps)} for d, ps in out.items()]


def _norm_cites(raw: Any, pool: dict[str, set[int]], answer: str = "") -> list[dict]:
    out: dict[str, set[int]] = {}
    single = next(iter(pool)) if len(pool) == 1 else None
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                doc = str(item.get("doc") or item.get("doc_id") or single or "").strip()
                pages = item.get("pages") or []
            else:
                doc, pages = single or "", [item]
            if not doc or doc not in pool:
                continue
            for x in pages:
                try:
                    n = int(x)
                except (TypeError, ValueError):
                    continue
                if n in pool[doc]:
                    out.setdefault(doc, set()).add(n)
    if out:
        return [{"doc_id": d, "pages": sorted(ps)} for d, ps in out.items()]
    # 结构化引用无效时，从答案正文的 [doc:pN] 标注提取
    from_text = _cites_from_text(answer, pool)
    if from_text:
        return from_text
    # 再退回 wiki 页自身的引用池（截断）
    return [
        {"doc_id": d, "pages": sorted(ps)[:8]}
        for d, ps in pool.items()
        if ps
    ]


def query_wiki(
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    doc_id: str | None = None,
) -> dict:
    """LLMWiki 式问答：选页 → 读 wiki 页（含一跳互链）→ 带 doc:page 引用作答。"""
    qcfg = cfg["query"]
    chat_model = cfg["llm"]["chat_model"]
    store = WikiStore(kb)
    pages = _wiki_scope(store, doc_id)
    if not pages:
        return {
            "mode": "wiki",
            "doc_id": doc_id or "",
            "answer": "wiki 层还没有内容，请先摄取文档或运行 tw distill <doc_id>。",
            "citations": [],
            "wiki_pages": [],
            "sufficient": False,
            "rounds": [],
            "usage": llm.usage(),
        }

    limit = int(qcfg.get("wiki_max_pages", 8))
    data = llm.chat_json(
        [
            {"role": "system", "content": WIKI_SELECT_SYS.format(limit=limit)},
            {
                "role": "user",
                "content": f"问题：{question}\n\nwiki 页面录：\n{_wiki_listing(pages, doc_id)}",
            },
        ],
        model=chat_model,
    )
    pick = [x for x in (data.get("pages") or []) if isinstance(x, str)]
    trace = [{"round": 1, "select": pick, "reason": data.get("reason", "")}]

    if not pick or data.get("sufficient") is False:
        return {
            "mode": "wiki",
            "doc_id": doc_id or "",
            "answer": str(data.get("reason") or "wiki 层没有足够相关的页面。"),
            "citations": [],
            "wiki_pages": [],
            "sufficient": False,
            "rounds": trace,
            "usage": llm.usage(),
        }

    read_pages = _load_wiki_pages(store, pick[:limit], limit + 3, doc_id)
    context = _wiki_context(read_pages, int(qcfg.get("wiki_page_chars", 3000)))
    answer = llm.chat_json(
        [
            {"role": "system", "content": WIKI_ANSWER_SYS},
            {
                "role": "user",
                "content": f"问题：{question}\n\n可用的 wiki 页：\n\n{context}",
            },
        ],
        model=chat_model,
    )
    answer_text = str(answer.get("answer", ""))
    pool = _cite_pool(read_pages)
    citations = _norm_cites(answer.get("citations"), pool, answer_text)
    sufficient = bool(answer.get("sufficient", True))
    if not sufficient:
        citations = []
    elif pool and not citations:
        sufficient = False

    return {
        "mode": "wiki",
        "doc_id": doc_id or "",
        "answer": str(answer.get("answer", "")),
        "citations": citations,
        "wiki_pages": [
            {"id": p.id, "title": p.title, "type": p.type, "stale": p.stale}
            for p in read_pages
        ],
        "sufficient": sufficient and bool(citations),
        "rounds": trace,
        "usage": llm.usage(),
    }


# ------------------------------------------------------------------- 组合模式


def _pick_doc(kb: KB, doc_id: str | None, wiki: dict) -> str | None:
    if doc_id:
        return doc_id
    counts: dict[str, int] = {}
    for c in wiki.get("citations") or []:
        doc = c.get("doc_id")
        if doc:
            counts[doc] = counts.get(doc, 0) + len(c.get("pages") or [])
    if counts:
        return max(counts, key=counts.get)
    sources = [k for k, v in kb.sources().items() if v.get("status") == "active"]
    return sources[0] if len(sources) == 1 else None


def query_cascade(
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    doc_id: str | None = None,
) -> dict:
    """默认模式：wiki 优先；不足或需要原始精度时下钻源树。"""
    wiki = query_wiki(question, kb, llm, cfg, doc_id)
    if wiki["sufficient"]:
        return {**wiki, "mode": "cascade", "path": "wiki"}

    target = _pick_doc(kb, doc_id, wiki)
    if not target:
        return {**wiki, "mode": "cascade", "path": "wiki"}

    try:
        raw = query_raw(target, question, kb, llm, cfg)
    except FileNotFoundError:
        if doc_id:
            raise
        return {**wiki, "mode": "cascade", "path": "wiki"}
    return {
        **raw,
        "mode": "cascade",
        "path": "raw",
        "wiki_attempt": {
            "answer": wiki["answer"],
            "sufficient": wiki["sufficient"],
            "citations": wiki["citations"],
            "rounds": wiki["rounds"],
        },
    }


def _merge_citations(*groups: list[dict]) -> list[dict]:
    merged: dict[str, set[int]] = {}
    for group in groups:
        for c in group or []:
            doc = str(c.get("doc_id") or c.get("doc") or "").strip()
            if not doc:
                continue
            pages = merged.setdefault(doc, set())
            for p in c.get("pages") or []:
                try:
                    pages.add(int(p))
                except (TypeError, ValueError):
                    continue
    return [{"doc_id": d, "pages": sorted(ps)} for d, ps in merged.items()]


def query_parallel(
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    doc_id: str | None = None,
) -> dict:
    """并行模式：wiki 与 raw 同时跑，对比答案，raw 为事实基准。"""
    wiki_first: dict | None = None
    target = doc_id
    if not target:
        wiki_first = query_wiki(question, kb, llm, cfg, doc_id)
        target = _pick_doc(kb, None, wiki_first)
        if not target:
            return {
                **wiki_first,
                "mode": "parallel",
                "answer": wiki_first["answer"] + "\n\n（无可下钻的源文档，仅返回 wiki 路）",
                "verification": {"consistent": False, "note": "无源文档可对比"},
            }

    with ThreadPoolExecutor(max_workers=2) as pool:
        if wiki_first is not None:
            wiki = wiki_first
            raw = pool.submit(query_raw, target, question, kb, llm, cfg).result()
        else:
            fw = pool.submit(query_wiki, question, kb, llm, cfg, doc_id)
            fr = pool.submit(query_raw, target, question, kb, llm, cfg)
            wiki, raw = fw.result(), fr.result()

    if not wiki.get("sufficient") or not wiki.get("answer"):
        return {
            "mode": "parallel",
            "doc_id": raw.get("doc_id", ""),
            "answer": raw["answer"] + "\n\n（wiki 路无可用内容，已跳过双路对比）",
            "citations": raw.get("citations", []),
            "wiki_pages": [],
            "read_pages": raw.get("read_pages", []),
            "read_nodes": raw.get("read_nodes", []),
            "rounds": raw.get("rounds", []),
            "verification": {"consistent": False, "note": "wiki 无可用内容"},
            "wiki_answer": wiki.get("answer", ""),
            "raw_answer": raw["answer"],
            "usage": llm.usage(),
        }

    data = llm.chat_json(
        [
            {"role": "system", "content": COMPARE_SYS},
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"wiki 路答案：\n{wiki['answer']}\n\n"
                    f"原始文档页答案：\n{raw['answer']}"
                ),
            },
        ],
        model=cfg["llm"]["chat_model"],
    )
    consistent = bool(data.get("consistent", False))
    note = str(data.get("note") or "")
    answer = raw["answer"]
    if consistent:
        answer += "\n\n（双路验证一致）"
    else:
        answer += f"\n\n（⚠️ 双路不一致：{note or '以原始文档为准'}；wiki 路答案：{wiki['answer']}）"

    citations = _merge_citations(raw.get("citations"), wiki.get("citations"))

    return {
        "mode": "parallel",
        "doc_id": raw.get("doc_id", ""),
        "answer": answer,
        "citations": citations,
        "wiki_pages": wiki.get("wiki_pages", []),
        "read_pages": raw.get("read_pages", []),
        "read_nodes": raw.get("read_nodes", []),
        "rounds": raw.get("rounds", []),
        "verification": {"consistent": consistent, "note": note},
        "wiki_answer": wiki["answer"],
        "raw_answer": raw["answer"],
        "usage": llm.usage(),
    }


def run_query(
    question: str,
    kb: KB,
    llm: LLM,
    cfg: dict,
    mode: str | None = None,
    doc_id: str | None = None,
) -> dict:
    """统一入口：raw / wiki / cascade（默认） / parallel。"""
    mode = (mode or cfg["query"].get("mode") or "cascade").lower()
    if mode == "raw":
        if not doc_id:
            raise ValueError("raw 模式需要指定 doc_id")
        return query_raw(doc_id, question, kb, llm, cfg)
    if mode == "wiki":
        return query_wiki(question, kb, llm, cfg, doc_id)
    if mode == "cascade":
        return query_cascade(question, kb, llm, cfg, doc_id)
    if mode == "parallel":
        return query_parallel(question, kb, llm, cfg, doc_id)
    raise ValueError(f"未知查询模式: {mode}（可选 raw/wiki/cascade/parallel）")
