"""验证集构造：从原文事实句机械生成问题 + 标准答案 + 标准引用。

标准引用不靠模型判断——语料里每篇文章由 **【文章 A####】** 标记切分，事实句落在
哪一页是构造时已知的事实，所以 gold_pages 是确定量，可以直接算引用召回/精度。

四档问题（tier 由生产代码的实测行为决定，不靠模板猜测）：
  A 字面可命中：全部锚词都出现在标准页上
  B 字面不可命中：改写后锚词落空，只能靠树索引 + 摘要定位
  C 跨天聚合：同一实体在多天多篇出现，考时序梳理与多页引用
  D 未收录：语料里确实没有这个实体，考拒答而不是编造引用
      strict=实体串整本零命中；adjacent=实体串没有但锚词命中了别的产品（幻觉高危）
"""

from __future__ import annotations

import json
import random
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.store import KB
from ..pipelines.query import anchor_terms

MARKER = re.compile(r"\*\*【文章\s*(A\d{3,5})】(.+?)\*\*")
_HEADER_DATE = re.compile(r"发布时间：(\d{4}-\d{2}-\d{2})")
_HEADER_SOURCE = re.compile(r"来源：\s*([^｜|\n]+)")
_MONEY = re.compile(r"\d[\d,，.]*\s*(?:万亿|亿|万)?\s*(?:元|美元|美金)")
_PCT = re.compile(r"\d[\d,，.]*\s*(?:%|％|个百分点)")
_COUNT = re.compile(
    r"\d[\d,，.]*\s*(?:万亿|亿|万)?\s*"
    r"(?:股|倍|台|辆|部|家|款|人次|人|吨|颗|GW|MWh|GWh|Wh|mAh|TB|GB|艘|套|例|笔|次|处|起)"
)
_DATE = re.compile(r"(?:20\d{2}年)?\d{1,2}月\d{1,2}日")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-_ ]{2,}")
_CJK = re.compile(r"[\u4e00-\u9fff]{3,12}")
_CLAUSE = re.compile(r"[，,；;、：。！？\n]")
# 线索词：数字前面必须是这类谓词，否则完形提问会读不通
_CUES = (
    "增长", "上涨", "下跌", "下降", "提升", "回落", "达到", "升至", "降至", "涨至",
    "超过", "突破", "低于", "约", "为", "是", "报", "收报", "定价", "售价", "起售价",
    "起售", "开售", "成交价", "成交", "销量", "出货量", "发货量", "份额", "占比",
    "规模", "投资", "营收", "收入", "净利", "交付", "容量", "速度", "良率", "产能",
    "装机", "渗透率", "转化率", "效率", "密度", "价格", "费用", "补偿", "回购",
    "分红", "募资", "市值", "股价", "库存", "涨幅", "跌幅", "报收", "开盘", "收盘",
    "总价值", "估值", "票价", "工资", "薪资", "成本", "预算", "拨款", "补助", "罚款",
)
_PARA = {
    "同比增长": "较去年同期增长",
    "环比增长": "较上月增长",
    "出货量": "发货量",
    "销量": "卖出的数量",
    "售价": "对外零售定价",
    "价格": "价位",
    "上涨": "走高",
    "下跌": "回落",
    "达到": "升至",
    "超过": "突破",
    "预计": "机构预估",
    "营收": "营业收入",
    "净利润": "净利表现",
    "占比": "所占份额",
    "规模": "体量",
    "交付": "交付使用",
    "约": "大约",
}


@dataclass
class Article:
    aid: str
    title: str
    source: str = ""
    date: str = ""
    frags: list[tuple[int, str]] = field(default_factory=list)

    def paragraphs(self) -> list[tuple[int, str]]:
        out = []
        for page, text in self.frags:
            if "推荐阅读" in text:
                text = text.split("推荐阅读", 1)[0]
            for raw in text.split("\n"):
                line = raw.strip()
                if len(line) < 8 or line.startswith("来源：") or "原文：https" in line:
                    continue
                out.append((page, line))
        return out

    def subject(self) -> str:
        latin = [t.strip() for t in _LATIN.findall(self.title) if len(t.strip()) >= 3]
        if latin:
            return latin[0]
        runs = _CJK.findall(self.title)
        return max(runs, key=len)[:8] if runs else self.title[:8]


def _sid(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % 9973


def split_articles(pages: list[tuple[int, str]]) -> list[Article]:
    """按文章标记切分整本语料，跨页文章保留每段文字所属的页码。"""
    arts: list[Article] = []
    cur: Article | None = None
    for page_no, text in pages:
        matches = list(MARKER.finditer(text))
        if not matches:
            if cur is not None and text.strip():
                cur.frags.append((page_no, text))
            continue
        prev = 0
        for m in matches:
            if cur is not None:
                head = text[prev : m.start()]
                if head.strip():
                    cur.frags.append((page_no, head))
            cur = Article(aid=m.group(1), title=m.group(2).strip(" *｜|").strip())
            arts.append(cur)
            tail = text[m.end() : m.end() + 400]
            d = _HEADER_DATE.search(tail)
            cur.date = d.group(1) if d else ""
            s = _HEADER_SOURCE.search(tail)
            cur.source = s.group(1).strip() if s else ""
            prev = m.end()
        if cur is not None and text[prev:].strip():
            cur.frags.append((page_no, text[prev:]))
    return arts


def _clauses(sent: str) -> list[tuple[int, int, str]]:
    """切成分句并保留每个分句在原句里的偏移，便于定位数字所属语境。"""
    out = []
    start = 0
    for m in _CLAUSE.finditer(sent):
        if m.start() > start:
            out.append((start, m.start(), sent[start : m.start()]))
        start = m.end()
    if start < len(sent):
        out.append((start, len(sent), sent[start:]))
    return out


def _cue_before(left: str) -> str | None:
    """线索词必须紧贴数字（是 left 的结尾），否则完形提问会读成残句。"""
    head = left.rstrip()
    for cue in sorted(_CUES, key=len, reverse=True):
        if head.endswith(cue):
            return cue
    return None


def _clean_clause(left: str) -> bool:
    if left.startswith(_DEICTIC):
        return False
    if any(ch in left for ch in "（）()【】《》“”\"'"):
        return False
    return not any(k in left for k in _BAD_WORDS)


# 分句以这些词开头时，问题离开原文就无法理解，不能当验证集
_DEICTIC = (
    "上述", "其", "该", "这", "那", "此外", "同时", "其中", "另外", "而", "但", "则",
    "并", "以及", "目前", "对于", "由于", "据", "显示", "因此", "不过", "还有", "包括",
    "且", "也", "均", "已", "更", "还", "另", "就是", "只是", "然而", "但是", "而且",
    "在", "为", "与", "和", "或", "及", "于", "按", "比", "是", "至", "到", "从",
    "对", "将", "被", "把", "让", "使", "给", "由", "当", "随着", "伴随", "相比",
)
_BAD_WORDS = ("一是", "二是", "三是", "例如", "比如", "一个是", "其中")


def _cloze(sent: str, m: re.Match, kind: str, art: Article) -> str | None:
    """完形填空式提问：把数字换成疑问词，其余沿用原文分句，保证句子读得通。"""
    for a, b, clause in _clauses(sent):
        if not (a <= m.start() and m.end() <= b):
            continue
        left = clause[: m.start() - a]
        cue = _cue_before(left)
        if not cue or len(left) < len(cue) + 2 or len(left) > 46:
            return None
        if not _clean_clause(left):
            return None
        left = left.rstrip("-－+±约超")
        q = f"{left}多少？" if kind != "date" else f"{left}是在什么时候？"
        if not anchor_terms(q):
            subj = art.subject()
            q = f"{subj}：{q}"
        q = re.sub(r"\s+", "", q)
        return q if 8 <= len(q) <= 64 else None
    return None


def _paraphrase(q: str) -> str | None:
    out = q
    for a, b in _PARA.items():
        out = out.replace(a, b)
    if out == q:
        return None
    # 去掉拉丁实体，逼出问题只能靠语义而非字面命中
    out = _LATIN.sub("", out)
    out = re.sub(r"[：:，,]{2,}|^[：:，,]", "", out)
    out = re.sub(r"\s+", "", out)
    return out if 8 <= len(out) <= 64 and anchor_terms(out) else None


def _figures(sent: str, m: re.Match) -> list[str]:
    core = re.findall(r"\d[\d,，.]*", m.group(0))
    unit = bool(re.search(r"[%％元亿万美元倍]", m.group(0)))
    return [c for c in dict.fromkeys(core) if len(re.sub(r"\D", "", c)) >= 2 or unit][:2]


def build_factual_items(arts: list[Article], rng: random.Random) -> list[dict]:
    """每篇文章最多 2 个事实句，各产出一条字面题和一条改写题。"""
    items: list[dict] = []
    for art in arts:
        cands = []
        for page, para in art.paragraphs():
            for sent in (s.strip() for s in re.split(r"(?<=[。！？；])", para)):
                if not 15 <= len(sent) <= 190:
                    continue
                for kind, rx in (("money", _MONEY), ("pct", _PCT), ("count", _COUNT)):
                    m = rx.search(sent)
                    if m:
                        cands.append((page, sent, m, kind))
                        break
        rng.shuffle(cands)
        picked = 0
        used_q = set()
        for page, sent, m, kind in cands:
            if picked >= 2:
                break
            q = _cloze(sent, m, kind, art)
            if not q or q in used_q:
                continue
            used_q.add(q)
            base = {
                "article": art.aid,
                "article_title": art.title,
                "date": art.date,
                "subject": art.subject(),
                "gold_pages": [page],
                "gold_span": [min(p for p, _ in art.frags), max(p for p, _ in art.frags)],
                "gold_answer": sent,
                "gold_figures": _figures(sent, m),
            }
            items.append({**base, "id": f"{art.aid}-p{page}-{_sid(q)}", "kind": "literal", "question": q})
            para_q = _paraphrase(q)
            if para_q and para_q not in used_q:
                used_q.add(para_q)
                items.append(
                    {**base, "id": f"{art.aid}-p{page}-{_sid(para_q)}", "kind": "para", "question": para_q}
                )
            picked += 1
    return items


def _longest_run(text: str) -> str:
    """问题里最长的中文片段：用来判断这条问题在全库是否唯一可辨。"""
    runs = [re.sub(r"(多少|什么时候)[？?]?$", "", r) for r in re.findall(r"[\u4e00-\u9fff]{2,}", text)]
    runs = [r for r in runs if r and r not in ("多少", "什么时候")]
    if runs:
        return max(runs, key=len)
    latin = _LATIN.findall(text)
    return max((t.strip().lower() for t in latin), key=len, default="")


def _count_pages(lows: list[str], key: str, limit: int = 6) -> int:
    """只关心「是否超过 5 页」，找到 limit 页就停，避免全库扫。"""
    n = 0
    for t in lows:
        if key in t:
            n += 1
            if n >= limit:
                return n
    return n


def classify_by_lexicon(items: list[dict], lows: list[str]) -> None:
    """tier 由生产 anchor_terms 的实测命中率决定，顺带暴露 AND 策略的代价。

    A：全部锚词都落在标准页上——字面兜底直接命中
    B：部分锚词命中——改成 OR/多数命中就能找到，是策略问题不是语料问题
    S：一个锚词都不命中——只能靠树索引和摘要做语义定位
    ambiguous：问题里最长的中文片段在全库出现超过 2 页，说明这条问题不唯一，剔除
    """
    for it in items:
        terms = anchor_terms(it["question"])
        it["anchor_terms"] = terms
        low = lows[it["gold_pages"][0] - 1]
        hits = [t in low for t in terms]
        it["term_hits"] = f"{sum(hits)}/{len(hits)}" if terms else "0/0"
        it["lex_hit"] = bool(terms) and all(hits)
        if it["lex_hit"]:
            it["tier"] = "A"
        elif any(hits):
            it["tier"] = "B"
        else:
            it["tier"] = "S"
        key = _longest_run(it["question"])
        it["key_term"] = key
        it["key_pages"] = _count_pages(lows, key, limit=3) if key else 0
        # 原句题的说法必然逐字来自语料；一片刻都找不到说明抽出来的分句是残句
        broken_literal = it.get("kind") == "literal" and it["key_pages"] == 0
        it["ambiguous"] = (not key) or it["key_pages"] > 2 or broken_literal


_VALUE = re.compile(r"[+\-]?\d[\d,，.]*\s*(?:%|％|万亿|亿|万|元|美元|港元|倍)?")
_SENT_SPLIT = re.compile(r"[。；;！？\n]")


def _norm_value(v: str) -> str:
    return re.sub(r"[,，.．\s]", "", v or "").lower()


def _left_context(answer: str, figure: str) -> str:
    """取标准答案里数字前面的那半句，用它去别处找同一个问法。"""
    idx = answer.find(figure)
    if idx < 0:
        return ""
    return _SENT_SPLIT.split(answer[max(0, idx - 48) : idx])[-1].strip()


def expand_duplicate_gold(items: list[dict], lows: list[str]) -> tuple[int, int]:
    """同一个问法在语料别处出现：数值一致就把那页并入 gold，数值冲突说明问题不唯一。

    公众号每天复述同一批行情，同一句话常带不同数字出现在不同日期；不处理的话，
    答对另一天的正确数字也会被判错。冲突题直接剔除，不给评测留模棱两可的标准。
    """
    expanded = conflicted = 0
    for it in items:
        figs = it.get("gold_figures") or []
        left = _left_context(it.get("gold_answer") or "", figs[0]) if figs else ""
        if len(left) < 6:
            continue
        probe = left.lower()
        gold = {_norm_value(f) for f in figs}
        same: set[int] = set()
        diff: set[int] = set()
        for i, low in enumerate(lows, start=1):
            start = 0
            while True:
                j = low.find(probe, start)
                if j < 0:
                    break
                tail = low[j + len(probe) : j + len(probe) + 20].lstrip()
                m = _VALUE.match(tail)
                if m:
                    value = _norm_value(m.group(0))
                    if any(g and (g in value or value in g) for g in gold):
                        same.add(i)
                    else:
                        diff.add(i)
                start = j + 1
        span = it.get("gold_span") or []
        inside = lambda p: len(span) == 2 and span[0] <= p <= span[1]
        same = {p for p in same if not inside(p)}
        diff = {p for p in diff if not inside(p)}
        if diff:
            it["ambiguous"] = True
            it["conflict_pages"] = sorted(diff)[:5]
            conflicted += 1
            continue
        if same:
            it["gold_pages"] = sorted(set(it["gold_pages"]) | same)
            it["gold_spans"] = ([span] if span else []) + [[p, p] for p in sorted(same)]
            expanded += 1
    return expanded, conflicted


_GENERIC_TERMS = {
    "ai", "agent", "app", "apps", "llm", "api", "pc", "tv", "ip", "url", "web",
    "vs", "etf", "ipo", "ceo", "cpu", "gpu", "ic", "us", "uk", "eu", "a", "b", "2",
}
# 中文标题里的通用词：不构成可追踪的事件实体
_GENERIC_CJK = {
    "中国", "公司", "报告", "市场", "行业", "全球", "第一", "二手", "有关", "相关",
    "预计", "表示", "认为", "介绍", "透露", "消息", "最新", "热门", "此前", "目前",
    "项目", "产品", "技术", "数据", "价格", "方案", "记者", "了解", "显示", "超过",
}


# 标题里的行情/事件碎片不是可追踪实体，不能当时序题的主体
_EVENT_FRAGMENTS = (
    "跌", "涨", "高开", "低开", "收", "盘", "大幅", "飙升", "跳水", "普涨", "普跌",
    "集体", "个股", "曝光", "护城河", "受贿", "特别巨大", "很深", "速递", "午评",
    "晚会", "回应", "发声", "最新", "一度", "突破", "超过", "巨震", "拉升", "走弱",
    "走强", "回调", "翻倍", "腰斩", "破发", "新高", "新低", "创新高", "市值", "市值",
)


def _title_terms(art: Article) -> list[str]:
    terms = [re.sub(r"\s+", " ", t.strip().lower()) for t in _LATIN.findall(art.title)]
    terms += [
        t for t in _CJK.findall(art.title) if 3 <= len(t) <= 8 and t not in _GENERIC_CJK
    ]
    out = []
    for t in terms:
        if not t or t in _GENERIC_TERMS or len(t) < 3:
            continue
        if not _LATIN.match(t) and any(f in t for f in _EVENT_FRAGMENTS):
            continue
        out.append(t)
    return out


def build_timeline_items(arts: list[Article], min_arts: int = 3, cap: int = 220) -> list[dict]:
    """跨天聚合题：同一实体出现在多篇、多天文章的标题里，取最近 8 篇作为标准集。"""
    by_term: dict[str, list[Article]] = {}
    for art in arts:
        for t in set(_title_terms(art)):
            by_term.setdefault(t, []).append(art)
    out = []
    for term, group in sorted(by_term.items(), key=lambda kv: -len(kv[1])):
        if len(group) > 30:
            continue
        uniq = {a.aid: a for a in group}
        dates = {a.date for a in uniq.values() if a.date}
        if len(uniq) < min_arts or len(dates) < 2:
            continue
        shown = sorted(uniq.values(), key=lambda a: a.date)[-8:]
        pages = sorted({p for a in shown for p, _ in a.frags})
        spans = [(min(p for p, _ in a.frags), max(p for p, _ in a.frags)) for a in shown]
        if not pages:
            continue
        out.append(
            {
                "id": f"T-{_sid(term)}",
                "tier": "C",
                "kind": "timeline",
                "question": f"{term} 最近的这批报道里，出现了哪些关键价格和数字？按时间顺序列一下。",
                "entity": term,
                "gold_pages": pages[:30],
                "gold_spans": spans,
                "gold_articles": sorted(a.aid for a in shown),
                "gold_answer": " / ".join(f"{a.date} {a.title[:28]}" for a in shown[:6]),
                "gold_figures": [],
                "lex_hit": True,
                "subject": term,
                "anchor_terms": anchor_terms(term),
            }
        )
        if len(out) >= cap:
            break
    return out


_ENTITIES = [
    "iPhone 19 Air", "iPhone Duo Ultra", "Mate 90 Pro Max", "Mate X8 Ultra", "小米 SU9 Ultra",
    "Model 2 Standard", "汉L EV 2028", "麒麟 9200 Pro", "骁龙 12 Gen 5", "天玑 10000",
    "GPT-7 mini", "Claude 8 Opus", "Gemini 5 Ultra", "Qwen 6 Max", "DeepSeek V5 Lite",
    "Galaxy S29 FE", "Pixel 11 Pro", "Vision Pro 4", "AirPods Ultra 2", "Watch S5 Pro",
    "FreeBuds 8 Pro", "Nova 16 Ultra", "Magic 20 Pro", "Yoga Book 3", "Rubin B400",
    "混元 5 Turbo", "星火 X4 Pro", "文心 6.0 Turbo", "远征 6 Ultra", "G3 Pro 机器狗",
    "钠新 4.0 电池", "麒麟 X3 芯片", "问界 M5 2029", "极氪 9X Ultra", "仰望 U10",
    "理想 I10 Max", "小鹏 GX9", "蔚来 ET9 2029", "零跑 D20 Pro", "埃安 RT Ultra",
    "iPhone 20 Pro Max", "iPhone Air 2", "Mate 100", "P60 Ultra 2029", "小米 18S Pro",
    "SU7 Ultra 2029", "YU10 Max", "银河 M8 2029", "坦克 500 Hi4-Z", "风云 A9 2030",
    "星纪元 ET9 Ultra", "岚图 珠峰 2029", "极狐 N80", "海豹 08 GT", "宋L 2029 款",
    "Rubin C500", "Blackwell Ultra 2", "GB500 NVL9", "昇腾 970", "思元 690",
    "L40X Ultra", "盘古 7.0 Turbo", "豆包 3 Pro Max", "Kimi K3 Ultra", "通义 7.0",
    "混元 8 Turbo", "文心 8.0", "星火 5.0 Ultra", "GLM 8 Max", "MiniMax 4 Pro",
    "MiniMax M3 Ultra", "即梦 5.0 Pro", "可灵 4.0 Ultra", "Seedance 3 Pro", "Sora 4",
]
_ASKS = ["的官方起售价是多少？", "的首发时间和预售数据是多少？"]


def build_absent_items(joined: str, cap: int = 250) -> list[dict]:
    """未收录题：实体串在整本语料里零命中，期望系统拒答。

    adjacent 变体故意挑锚词命不中的写法（"Mate 90" 里没有 "Mate 90" 但 "Mate" 满天飞），
    这是最容易被幻觉击穿的一档。
    """
    out: list[dict] = []
    for ent in _ENTITIES:
        if ent.lower() in joined:
            continue
        strict = all(t not in joined for t in anchor_terms(ent))
        for n, ask in enumerate(_ASKS):
            q = f"{ent}{ask}" if n == 0 else f"{ent} 现在官方报的价钱和备货是多少？"
            out.append(
                {
                    "id": f"X{n}-{_sid(q)}",
                    "tier": "D",
                    "kind": "strict" if strict else "adjacent",
                    "question": q,
                    "entity": ent,
                    "subject": ent,
                    "gold_pages": [],
                    "gold_answer": "",
                    "gold_figures": [],
                    "expect_refusal": True,
                    "anchor_terms": anchor_terms(q),
                    "lex_hit": False,
                }
            )
        if len(out) >= cap:
            break
    return out


def quota(items: list[dict], per_tier: dict[str, int], rng: random.Random, total: int = 1000) -> list[dict]:
    """按档位抽样；某档池子不够时用剩余档位补齐到 total，避免验证集缩水。"""
    pools = {t: [i for i in items if i["tier"] == t] for t in per_tier}
    for pool in pools.values():
        rng.shuffle(pool)
    out = []
    for tier, n in per_tier.items():
        out.extend(pools[tier][:n])
    for tier in per_tier:
        if len(out) >= total:
            break
        take = total - len(out)
        used = {i["id"] for i in out}
        extra = [i for i in pools[tier] if i["id"] not in used][:take]
        out.extend(extra)
    rng.shuffle(out)
    return out


def build_dataset(
    kb: KB,
    doc_id: str,
    seed: int = 20261020,
    per_tier: dict[str, int] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    total = kb.page_count(doc_id)
    pages = [(i, t) for i, t in enumerate(kb.read_pages(doc_id, 1, total), start=1)]
    lows = [t.lower() for t in (text for _, text in pages)]
    joined = "".join(lows)
    arts = split_articles(pages)
    facts = build_factual_items(arts, rng)
    classify_by_lexicon(facts, lows)
    expanded, conflicted = expand_duplicate_gold(facts, lows)
    dropped = sum(1 for i in facts if i["ambiguous"])
    facts = [i for i in facts if not i["ambiguous"]]
    timeline = build_timeline_items(arts)
    absent = build_absent_items(joined)
    items = quota(
        facts + timeline + absent,
        per_tier or {"A": 200, "B": 200, "S": 300, "C": 180, "D": 120},
        rng,
    )
    for n, it in enumerate(items, start=1):
        it["no"] = n
        it["doc_id"] = doc_id
    return {
        "doc_id": doc_id,
        "pages": total,
        "articles": len(arts),
        "joined_chars": len(joined),
        "candidate_facts": len(facts),
        "candidate_ambiguous": dropped,
        "candidate_duplicate_gold": expanded,
        "candidate_duplicate_conflict": conflicted,
        "candidate_a": sum(1 for i in facts if i["tier"] == "A"),
        "candidate_b": sum(1 for i in facts if i["tier"] == "B"),
        "candidate_s": sum(1 for i in facts if i["tier"] == "S"),
        "candidate_timeline": len(timeline),
        "candidate_absent": len(absent),
        "items": items,
    }


# 语料是有版权的公众号文章，公开版验证集只留判分需要的结构化字段
_LEAKY_FIELDS = ("gold_answer", "gold_text", "article_title", "subject")


def sanitize_items(items: list[dict]) -> list[dict]:
    """去掉原文整句与文章标题，保留问题、gold 页码和期望数字：判分照样能跑。"""
    return [{k: v for k, v in it.items() if k not in _LEAKY_FIELDS} for it in items]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
