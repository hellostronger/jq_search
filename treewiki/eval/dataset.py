"""验证集构造：从原文事实句机械生成问题 + 标准答案 + 标准引用。

标准引用不靠模型判断——语料里每篇文章由 **【文章 A####】** 标记切分，事实句落在
哪一页是构造时已知的事实，所以 gold_pages 是确定量，可以直接算引用召回/精度。

四档问题（tier 由生产代码的实测行为决定，不靠模板猜测）：
  A 字面可命中：全部锚词都出现在标准页上
  B 字面不可命中：改写后锚词落空，只能靠树索引 + 摘要定位
  C 跨天聚合：同一实体在多天多篇出现，考时序梳理与多页引用
  D 未收录：语料里确实没有这个实体，考拒答而不是编造引用
      strict=实体串整本零命中；adjacent=实体串没有但锚词命中了别的产品（幻觉高危）

派生档（由 A/B/S 题配对生成，gold 全部来自原题字段）：
  H 多跳：同一篇里两个不同事实合成一问，两处证据、两个数字都答对才算过
  M 多轮：同一主体、两个不同指标；第二轮省掉主语，靠对话历史补全

题目自足性：单看问句必须能认出问的是谁/哪个指标。只留"9月环比走高"这类
脱离原文就不知道在问什么的问法会让评测失真，生成时直接拒绝，已有脱敏集
用 `upgrade_items` 剔除。
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
from ..pipelines.query import anchor_terms, question_subject

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
    for cue in sorted(_CUES, key=lambda w: (-len(w), w)):
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
_BAD_WORDS = ("一是", "二是", "三是", "例如", "比如", "一个是", "其中", "是否", "能否", "可否", "为什么", "如何")


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
        if _subjectless(q):
            return None
        return q if 8 <= len(q) <= 64 else None
    return None


def _paraphrase(q: str) -> str | None:
    out = q
    for a, b in _PARA.items():
        out = out.replace(a, b)
    if out == q:
        return None
    # 先尝试去掉拉丁实体，逼出语义检索；但若删完主体就没了，宁可保留实体
    stripped = _LATIN.sub("", out)
    stripped = re.sub(r"[：:，,]{2,}|^[：:，,]", "", stripped)
    stripped = re.sub(r"\s+", "", stripped)
    if 8 <= len(stripped) <= 64 and anchor_terms(stripped) and not _subjectless(stripped):
        return stripped
    kept = re.sub(r"[：:，,]{2,}|^[：:，,]", "", out)
    kept = re.sub(r"\s+", "", kept)
    if 8 <= len(kept) <= 64 and anchor_terms(kept) and not _subjectless(kept):
        return kept
    return None


# ---- 题目自足性：单看问句必须能认出问的是谁/哪个指标 ----

_TIME_HEAD = re.compile(
    r"^(?:20\d{2}(?:财)?年(?:末|底|初|上半年|下半年)?"
    r"|20\d{2}年\d{1,2}月(?:\d{1,2}日)?|\d{1,2}月(?:\d{1,2}日)?"
    r"|上半年|下半年|前三季度|前\d+个月|第[一二三四]季度(?:末|底|初)?"
    r"|[一二三四]季度(?:末|底|初)?|全年|本月|上月|本周|上周|当日|同期|今天|今年|去年|明年|截至|从|自)"
)
_TIME_TAIL = re.compile(
    r"(?:20\d{2}年(?:\d{1,2}月)?|第[一二三四]季度(?:末|底|初)?"
    r"|[一二三四]季度(?:末|底|初)?|上半年|下半年|前三季度|全年|本月|上月|本周|上周)$"
)
_MOD_HEAD = (
    "同比", "环比", "累计", "合计", "分别", "均", "已", "共", "进一步", "继续",
    "仍", "最新", "目前", "当前", "大约", "约", "超过", "突破", "至少", "或",
    "将", "拟", "被", "实现", "录得", "每", "已经", "上市后",
    "较", "相较", "相比", "对比", "零售", "批发",
)
_PRED_TAIL = (
    "走高", "回落", "上涨", "下跌", "增长", "下降", "上升", "提升", "减少",
    "增加", "收窄", "扩大", "突破", "超过", "达到", "升至", "降至", "涨至",
    "跌至", "净流出", "净流入", "增至", "减至", "攀升至", "报", "为", "是", "至",
)
# 指标名词：整条问句只剩这些词时，说明主体被切掉了
_METRIC_WORDS = (
    "总市值", "市值", "营收", "营业收入", "收入", "净利润", "净利", "利润",
    "规模", "体量", "价格", "售价", "定价", "金额", "价款", "净额", "总额",
    "余额", "销量", "出货量", "发货量", "产量", "产能", "份额", "占比", "比例",
    "涨幅", "跌幅", "增幅", "增速", "上限", "下限", "利率", "收益率", "成本",
    "费用", "估值", "股价", "评分", "数量", "次数", "面积", "容量", "电量",
    "密度", "效率", "良率", "转化率", "渗透率", "概率", "价值", "中位数",
    "销售额", "净营收", "表现", "融资", "募集资金", "所得款项", "持股数量",
    "代码", "排名", "价格区间", "发行价",
)
# 范围/指代/泛称：不构成主体
_SCOPE_WORDS = (
    "全球", "全国", "我国", "海外", "国内", "行业", "市场", "整体", "部分",
    "多个", "各", "每", "两者", "二者", "前者", "后者", "其中", "相关", "主要",
    "全部", "所有", "该", "本", "这", "那", "某些", "若干", "一些", "全美",
    "公司", "企业", "集团", "银行", "机构", "部门", "产品", "业务", "项目",
    "平台", "系统", "设备", "车型", "主体", "客户", "用户", "投资者", "股东",
    "消费者", "受访者", "数据", "报告", "消息", "公告", "机构预估", "显然",
    "上市公司", "归属于",
    "同期", "上年同期", "去年同期", "去年", "今年", "前年", "明年",
    "本月", "上月", "本周", "上周", "当日", "当天",
    "预计", "预告", "估算", "测算", "财年", "季度", "年度", "月份", "成交",
)
_DETERMINER_HEADS = ("三款", "两款", "多款", "几款", "一些", "部分", "多个", "各", "每", "某些", "若干")
_DEICTIC_ANY = ("两者", "二者", "前者", "后者", "各自", "其中", "该公司", "该产品", "该基金", "该车型", "该指数")
_CONJ = "及和与、/"
_STRIP_WORDS = tuple(
    sorted(
        set(_METRIC_WORDS) | set(_PRED_TAIL) | set(_SCOPE_WORDS) | set(_MOD_HEAD),
        key=lambda w: (-len(w), w),
    )
)


def _question_left(question: str) -> str:
    """去掉疑问词和紧贴数字的线索词，留下主体+修饰语。"""
    q = (question or "").rstrip("？?")
    for tail in ("多少", "是在什么时候"):
        if q.endswith(tail):
            q = q[: -len(tail)]
            break
    for cue in sorted(set(_CUES) | set(_PRED_TAIL), key=lambda w: (-len(w), w)):
        if q.endswith(cue):
            q = q[: -len(cue)]
            break
    return q


def _strip_modifiers(left: str) -> str:
    changed = True
    while changed:
        changed = False
        m = _TIME_HEAD.match(left)
        if m:
            left, changed = left[m.end():], True
        for word in _MOD_HEAD:
            if left.startswith(word):
                left, changed = left[len(word):], True
        if left.endswith("的"):
            left, changed = left[:-1], True
    return left.strip()


def _nominal_part(part: str) -> str:
    """把指标/谓词/范围/时间修饰词从片段两端剥掉，剩下主体；只剩这些词时返回空串。

    必须整词匹配后再剥：早先按子串判断把"三星电子...营收"误当成只有"营收"，
    结果真主体被当成缺主体整批丢弃。
    """
    part = part.strip()
    while part:
        if (
            part in _GENERIC_HEADS
            or part in _METRIC_WORDS
            or part in _SCOPE_WORDS
            or part in _PRED_TAIL
            or part in _MOD_HEAD
            or part == "中"
        ):
            return ""
        m = _TIME_TAIL.search(part)
        if m and m.start() > 0:
            part = part[: m.start()]
            continue
        if part.endswith("的") and len(part) > 1:
            part = part[:-1]
            continue
        for word in _STRIP_WORDS:
            if part.startswith(word) and len(part) > len(word):
                part = part[len(word) :]
                break
            if part.endswith(word) and len(part) > len(word):
                part = part[: -len(word)]
                break
        else:
            break
    return part


def _has_nominal(question: str) -> bool:
    """问句里是否还剩一个非指标、非范围词的主体。"""
    core = _strip_modifiers(_question_left(question))
    if not core:
        return False
    if any(word in core for word in _DEICTIC_ANY):
        return False
    if core.startswith(_DETERMINER_HEADS):
        return False
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.\-_]{1,}|[\u4e00-\u9fff]{2,}", core):
        if re.match(r"[A-Za-z]", token):
            return True
        for part in re.split(rf"[{_CONJ}]", token):
            if len(_nominal_part(part)) >= 2:
                return True
    return False


def _subjectless(question: str) -> bool:
    """单看问句无法定位主体（"9月环比走高多少""最新市值升至多少"）。"""
    return not _has_nominal(question)



def _figures(sent: str, m: re.Match) -> list[str]:
    core = re.findall(r"\d[\d,，.]*", m.group(0))
    return [c for c in dict.fromkeys(core)][:2]


def build_factual_items(arts: list[Article], rng: random.Random) -> list[dict]:
    """每篇文章最多 3 个事实句，各产出一条字面题和一条改写题。"""
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
            if picked >= 3:
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
                "gold_match": m.group(0).strip(),
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


_NUM_PART = re.compile(r"[+\-]?\d[\d,，.]*")


def _norm_value(v: str) -> str:
    """数值归一：42 与 4,220 必须区分开，2.40 与 2.4 视为同一个值。"""
    m = _NUM_PART.search(v or "")
    if not m:
        return re.sub(r"\s", "", v or "").lower()
    s = m.group(0).replace(",", "").replace("，", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.lstrip("+")


def _left_context(answer: str, figure: str) -> str:
    """取标准答案里数字前面的那半句，用它去别处找同一个问法。"""
    m = re.search(rf"(?<![\d.．]){re.escape(figure)}(?![\d.．])", answer)
    idx = m.start() if m else answer.find(figure)
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
        anchor = it.get("gold_match") or (figs[0] if figs else "")
        left = _left_context(it.get("gold_answer") or "", anchor) if anchor else ""
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
                    if any(g and g == value for g in gold):
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


_CLEAN_TEXT = re.compile(r"[\s\-－—_：:，,。、|（）()『』“”\"'’]")
_GENERIC_HEADS = (
    "总市值", "总认缴规模", "售价", "配售价", "配对外", "分别", "营收", "净利润",
    "营业收入", "收入", "利润", "规模", "价格", "市值", "估值", "金额", "销量",
    "出货量", "份额", "占比", "评分", "店铺", "项目", "成本", "费用", "利率",
    "收益率", "涨幅", "跌幅", "同比", "环比", "均价", "总价", "合计", "累计", "平均",
    "平台", "交易", "规划", "投前", "营业利润率", "毛利率", "净利率", "年产能",
    "产能", "合理股价", "合同", "目标",
)
# 以相对/指代/承接连词开头：主体在上文，单看问题无法定位是哪家公司、哪个指标
_DEICTIC_HEADS = (
    "较", "相较", "相比", "对比", "占", "则", "而", "但", "且", "并", "其", "该", "这",
    "上述", "前述", "目前", "刚结束", "调整后", "该公司", "公司", "现有", "本次", "此次",
    "虽然", "尽管", "业绩变动", "公告显示", "数据显示", "报告显示", "相关内容",
    "报道称", "消息称", "据悉", "据了解", "就", "今年", "去年", "过去", "十年",
    "在", "的", "他", "她", "它", "前者", "后者", "上半年", "前三季度", "第一阶段",
    "北美",
)
_UNDERSPEC_YEAR = re.compile(r"^20\d{2}(?:财)?年.*(?:分别|同比|环比|增长|下降|不低于)")


def _norm_text(s: str) -> str:
    return _CLEAN_TEXT.sub("", s or "").lower()


def _underspecified(question: str) -> bool:
    """问题左半句没有可辨认的主体：以通用指标词或相对/指代词开头都算。"""
    if any(ch in question for ch in "：「」——"):
        return True
    if re.search(r"从[^？?]*的多少[？?]?$", question):
        return True
    if _subjectless(question):
        return True
    q = question.rstrip("？?")
    left = q[:-2] if q.endswith("多少") else q
    if left.startswith(_DEICTIC_HEADS) or left.startswith(_GENERIC_HEADS):
        return True
    return bool(_UNDERSPEC_YEAR.match(left))


def refine_questions(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """原句题的左半句必须真出现在标准答案里；同一问法对应不同答案说明丢了主体。

    改写题会把"规模"写成"体量"，逐字对不上是正常的，所以只校验原句题；
    同一问法命中多篇时，数值一致就把这些片段并进同一道题的 gold（重复报道），
    数值冲突才剔除（问法里丢了主体，答案不唯一）。
    """
    kept = []
    bad_clause = underspec = 0
    for it in items:
        if _underspecified(it["question"]):
            underspec += 1
            continue
        if it["kind"] == "literal":
            q = it["question"].rstrip("？?")
            left = q[:-2] if q.endswith("多少") else ""
            if len(left) >= 6 and _norm_text(left) not in _norm_text(it.get("gold_answer") or ""):
                bad_clause += 1
                continue
        kept.append(it)
    groups: dict[str, list[dict]] = {}
    for it in kept:
        groups.setdefault(_norm_text(it["question"]), []).append(it)
    out = []
    multi = merged = deduped = 0
    for group in groups.values():
        values = {tuple(sorted(_norm_value(f) for f in g.get("gold_figures") or [])) for g in group}
        if len(values) > 1:
            multi += len(group)
            continue
        base = group[0]
        pages = sorted({p for g in group for p in g["gold_pages"]})
        if len(group) > 1 and pages != sorted(base["gold_pages"]):
            same_value = bool(base.get("gold_figures")) and len(values) == 1
            if not same_value:
                multi += len(group)
                continue
            spans = []
            for g in group:
                spans += g.get("gold_spans") or ([g["gold_span"]] if g.get("gold_span") else [])
            base["gold_pages"] = pages
            base["gold_spans"] = [list(s) for s in dict.fromkeys(tuple(s) for s in spans)]
            merged += len(group) - 1
        else:
            deduped += len(group) - 1
        out.append(base)
    return out, {
        "bad_clause": bad_clause,
        "underspecified": underspec,
        "multi_conflict": multi,
        "merged": merged,
        "deduped": deduped,
    }


_GENERIC_TERMS = {
    "ai", "agent", "app", "apps", "llm", "api", "pc", "tv", "ip", "url", "web",
    "vs", "etf", "ipo", "ceo", "cpu", "gpu", "ic", "us", "uk", "eu", "a", "b", "2",
    "token", "seed", "alpha",
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
    "同比", "环比", "下降", "增长", "上涨", "下跌", "下滑", "上升", "重挫", "暴跌",
    "暴涨", "热搜", "提醒", "来了", "简历", "日记", "精选", "深夜", "关口", "到底",
    "全市场", "封单", "概念", "利好", "紧急", "公告", "官宣", "重回", "多只", "休市",
    "突发", "暴雨", "爆仓", "公诉", "诉讼", "辞职", "辞任", "死亡", "遇难", "失联",
    "伤亡", "被查", "被", "股公司",
)
# 以金额/数量单位或"超"收尾的标题词是统计口径碎片，不是实体
_TITLE_UNIT = re.compile(
    r"(?:万亿|万|亿|千)(?:美元|港元|韩元|日元|欧元|元|股|人次)"
    r"|(?:美元|港元|韩元|日元|欧元|元|股|人次|倍|率|额|量|润)$|超$"
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
        if not _LATIN.match(t) and (any(f in t for f in _EVENT_FRAGMENTS) or _TITLE_UNIT.search(t)):
            continue
        out.append(t)
    return out


def build_timeline_items(arts: list[Article], min_arts: int = 3, cap: int = 220) -> list[dict]:
    """跨天聚合题：同一实体出现在多篇、多天文章的标题里，取最近 8 篇作为标准集。"""
    by_term: dict[str, list[Article]] = {}
    for art in arts:
        for t in dict.fromkeys(_title_terms(art)):
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
        spans = [[min(p for p, _ in a.frags), max(p for p, _ in a.frags)] for a in shown]
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
                "gold_articles": [a.aid for a in shown],
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


# 两跳题只合并「问的是同一类量」的事实：
# - 平铺指标（是多少 / 为多少 / 多少）可以互换，合并后用“分别是多少”；
# - 变动、阈值谓词（升至多少 / 突破多少 / 增长多少）必须两边逐字一致。
# 否则合并会改变语义："股价下跌多少" + "股价上涨多少" 不等于 "股价分别是多少"。
_LEVEL_PREDICATES = (
    "分别是多少",
    "大约为多少",
    "大约是多少",
    "约为多少",
    "是多少",
    "为多少",
    "有多少",
    "多少",
)
_OTHER_PREDICATES = (
    "攀升至多少",
    "回落至多少",
    "升至多少",
    "降至多少",
    "涨至多少",
    "跌至多少",
    "增至多少",
    "减至多少",
    "上涨多少",
    "下跌多少",
    "走高多少",
    "回落多少",
    "增长多少",
    "下降多少",
    "提升多少",
    "减少多少",
    "增加多少",
    "收窄多少",
    "扩大多少",
    "突破多少",
    "超过多少",
    "达到多少",
    "净流入多少",
    "净流出多少",
    "蒸发多少",
    "减持多少",
    "增持多少",
    "报多少",
    "多少钱",
)
_HOP_PREDICATES = tuple(
    sorted(set(_LEVEL_PREDICATES) | set(_OTHER_PREDICATES), key=lambda w: (-len(w), w))
)

# 左半句必须以指标名词收尾，并且能独立成名词短语。价值/表现/融资这类词
# 即使能接“是多少”，也不是可比的指标，排除掉避免把"据报道价值"拼进题面。
_HOP_METRIC_TAILS = tuple(
    sorted(
        (
            set(_METRIC_WORDS)
            | {
                "持股比例", "注册资本", "平均利率", "年化利率", "达标率", "覆盖率",
                "净利率", "毛利率", "市占率", "增长率", "费用率", "分红率", "成交额",
                "成交均价", "吸金规模", "装机量", "新增装机", "订单金额", "合同金额",
                "融资额", "出厂价", "零售价", "批发价", "发行价", "储备", "持仓", "仓位",
                "渗透率", "转化率", "产能", "产量", "出货量", "发货量", "销售额", "净营收",
                "总市值", "市值", "估值", "股价", "净值", "份额", "占比", "比例", "概率",
                "收益率", "数量", "金额", "余额", "规模", "体量", "价格", "售价", "均价",
                "成本", "费用", "利率", "上限", "下限", "面积", "容量", "电量", "密度",
                "效率", "良率", "次数", "中位数",
            }
        )
        - {"价值", "表现", "融资", "代码", "排名", "评分", "定价", "价款", "净额", "总额"},
        key=lambda w: (-len(w), w),
    )
)
# 出现这些片段说明左半句还是从句/指代/比较结构，不能直接接"分别是多少"
_HOP_BAD_ANY = (
    "可能", "据报道", "就曾", "是在", "目标", "承诺", "表示", "宣布", "标价", "相比",
    "相较", "对比", "达成", "累计减持", "增持", "减持", "买入", "卖出", "加息的",
    "共有", "一度", "分别", "实现", "预计", "计划", "市场预计", "市场预期",
    # 指代词在上文，单看两跳题不知道说的是谁
    "该公司", "该产品", "该品种", "该股", "该基金", "该业务", "该行", "该款",
)
# 以动词开头的左半句，主体已经丢了（"参股易方达基金非货体量"）
_HOP_VERB_HEAD = (
    "参股", "持股", "收购", "入股", "出售", "增持", "减持", "买入", "卖出", "达成",
    "签署", "实现", "预计", "计划", "拟", "将", "曾", "已",
)
# 以时间/序数/泛指/指代开头，单看问句不知道说的是谁
_HOP_BAD_HEAD = re.compile(
    r"^(\d|第|首|这|那|该|其|此|同|上述|前者|后者|部分|整体|全部|所有|累计|平均|总|综合|"
    r"全市场|今年|去年|明年|本月|上月|同期|一度)"
)
_HOP_HEAD_WORDS = tuple(
    sorted(set(_SCOPE_WORDS) | set(_MOD_HEAD) | {"总", "综合"}, key=lambda w: (-len(w), w))
)
_HOP_TERM_SPLIT = re.compile(r"[的在中对占加及与和或、，,\s]+|\d{1,2}月|第?[一二三四]季度|\d{4}年")


def _split_fact_question(question: str) -> tuple[str, str]:
    """把事实题拆成「主体+指标」和「提问谓词」，谓词必须原样保留。"""
    q = (question or "").strip().rstrip("？?")
    for pred in _HOP_PREDICATES:
        if q.endswith(pred) and len(q) > len(pred):
            return q[: -len(pred)].strip("，,、；; "), pred
    return "", ""


def _hop_left_ok(left: str) -> bool:
    """左半句必须是一个带主体的名词短语，否则两跳题会读不通。"""
    left = left.strip("，,、；; ")
    if not 4 <= len(left) <= 30:
        return False
    if any(m in left for m in _HOP_BAD_ANY) or left.startswith(_HOP_VERB_HEAD):
        return False
    if _HOP_BAD_HEAD.match(left):
        return False
    if re.search(r"(?:总部|区域|地区|国内|海外|全国|中国)加", left):
        return False
    tail = next((t for t in _HOP_METRIC_TAILS if left.endswith(t)), "")
    if not tail:
        return False
    prefix = left[: -len(tail)].rstrip("的").strip()
    if len(prefix) < 2:
        return False
    # 去掉开头的泛指/修饰词后仍要剩下一个实体（"全球"/"总"/"该"都不算）
    core = prefix
    changed = True
    while changed and core:
        changed = False
        for word in _HOP_HEAD_WORDS:
            if core.startswith(word) and len(core) > len(word):
                core = core[len(word) :]
                changed = True
                break
    return len(core) >= 2


def _hop_pred_compatible(pred1: str, pred2: str) -> bool:
    if pred1 in _LEVEL_PREDICATES and pred2 in _LEVEL_PREDICATES:
        return True
    return pred1 == pred2


def _join_lefts(left1: str, left2: str, pred: str) -> str:
    """把两个事实的问句左半句合成一道两跳题，公共主体只写一次。"""
    left1 = left1.strip("，,、；; ")
    left2 = left2.strip("，,、；; ")
    if not left1 or not left2 or left1 == left2:
        return ""
    n = 0
    for a, b in zip(left1, left2):
        if a != b:
            break
        n += 1
    if n >= 3:
        left2 = left2[n:].lstrip("的")
    if len(left2) < 2:
        return ""
    tail = "分别是多少？" if pred in _LEVEL_PREDICATES else f"分别{pred}？"
    q = re.sub(r"\s+", "", f"{left1}、{left2}{tail}")
    return q if 8 <= len(q) <= 72 else ""


def _hop_terms(left: str) -> list[str]:
    """取左半句里可用来找「同一页复述」的实词（去掉指标尾词与时间）。"""
    core = left
    for tail in _HOP_METRIC_TAILS:
        if core.endswith(tail):
            core = core[: -len(tail)]
            break
    parts = [p for p in _HOP_TERM_SPLIT.split(core) if len(p) >= 2]
    return list(dict.fromkeys(parts))[:6]


def _has_value(low: str, value: str) -> bool:
    return any(_norm_value(m.group(0)) == value for m in _VALUE.finditer(low))


def _pair_pages(
    left1: str, left2: str, figs1: list[str], figs2: list[str], lows: list[str], known: set[int]
) -> list[int]:
    """同一页同时复述两处事实时也标成 gold——标准答案不只有来源页。

    只在有原语料时生效：脱敏集升级没有页文本，gold 只能沿用两跳各自已展开的页集。
    """
    if not lows:
        return []
    terms1, terms2 = _hop_terms(left1), _hop_terms(left2)
    values = [v for v in (_norm_value(f) for f in (figs1[:1] + figs2[:1])) if v]
    if len(values) < 2 or not terms1 or not terms2:
        return []
    extra = []
    for page, low in enumerate(lows, start=1):
        if page in known:
            continue
        if not all(_has_value(low, v) for v in values):
            continue
        if not any(t in low for t in terms1) or not any(t in low for t in terms2):
            continue
        extra.append(page)
    return extra


def _hop_fields(a: dict, b: dict, extra_pages: list[int] | None = None) -> dict:
    pages = sorted(
        set(a.get("gold_pages") or []) | set(b.get("gold_pages") or []) | set(extra_pages or [])
    )
    spans = []
    for it in (a, b):
        spans += it.get("gold_spans") or ([it["gold_span"]] if it.get("gold_span") else [])
    figs = [f for f in dict.fromkeys((a.get("gold_figures") or []) + (b.get("gold_figures") or []))]
    return {
        "gold_pages": pages,
        "gold_span": [min(pages), max(pages)] if pages else [],
        "gold_spans": [list(s) for s in dict.fromkeys(tuple(s) for s in spans)],
        "gold_figures": figs[:4],
        "hops": [
            {"pages": sorted(set(a.get("gold_pages") or [])), "figures": (a.get("gold_figures") or [])[:2]},
            {"pages": sorted(set(b.get("gold_pages") or [])), "figures": (b.get("gold_figures") or [])[:2]},
        ],
    }


def derive_multihop(
    items: list[dict], cap: int = 200, per_article: int = 2, lows: list[str] | None = None
) -> list[dict]:
    """同一篇文章里两个不同事实 → 一道两跳题：两处证据都要读到、两个数字都要答对。

    只合并谓词兼容的事实：平铺指标互换为"分别是多少"，变动/阈值谓词必须逐字一致。
    这样题目问的仍然是原事实的那个量，不会把"上涨多少"改写成"股价是多少"。
    lows 传入原语料时可以补标「同时复述两处事实」的页面；脱敏集升级没有页文本。
    """
    by_art: dict[str, list[dict]] = {}
    parsed: dict[str, tuple[str, str]] = {}
    for it in items:
        if it.get("tier") not in ("A", "B", "S") or it.get("kind") not in ("literal", "para"):
            continue
        if not it.get("gold_pages") or not it.get("gold_figures"):
            continue
        left, pred = _split_fact_question(it["question"])
        if not left or not _hop_left_ok(left):
            continue
        parsed[it["id"]] = (left, pred)
        by_art.setdefault(it.get("article") or it["id"], []).append(it)
    out = []
    for art, group in sorted(by_art.items()):
        group = sorted(group, key=lambda x: x["id"])
        pairs = [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]
        # 优先取分布在不同页的两处证据，同页配对留作兜底
        pairs.sort(
            key=lambda p: (
                bool(set(p[0].get("gold_pages") or []) & set(p[1].get("gold_pages") or [])),
                p[0]["id"],
                p[1]["id"],
            )
        )
        made = 0
        seen: set[str] = set()
        for a, b in pairs:
            fa, fb = a.get("gold_figures") or [], b.get("gold_figures") or []
            if not fa or not fb or _norm_value(fa[0]) == _norm_value(fb[0]):
                continue
            left_a, pred_a = parsed[a["id"]]
            left_b, pred_b = parsed[b["id"]]
            if not _hop_pred_compatible(pred_a, pred_b):
                continue
            pred = "多少" if pred_a in _LEVEL_PREDICATES and pred_b in _LEVEL_PREDICATES else pred_a
            q = _join_lefts(left_a, left_b, pred)
            if not q or q in seen or _subjectless(q):
                continue
            seen.add(q)
            extra = _pair_pages(
                left_a,
                left_b,
                fa,
                fb,
                lows or [],
                set(a.get("gold_pages") or []) | set(b.get("gold_pages") or []),
            )
            out.append(
                {
                    "id": f"H-{art}-{_sid(q)}",
                    "tier": "H",
                    "kind": "multihop",
                    "question": q,
                    "article": art,
                    "date": a.get("date") or b.get("date") or "",
                    "subject": art,
                    "anchor_terms": anchor_terms(q),
                    "lex_hit": False,
                    "expect_refusal": False,
                    **_hop_fields(a, b, extra),
                }
            )
            made += 1
            if made >= per_article:
                break
        if len(out) >= cap:
            break
    return out[:cap]


def derive_multiturn(items: list[dict], cap: int = 150, per_subject: int = 2) -> list[dict]:
    """同一主体的两个事实 → 两轮对话：第二轮省掉主语，靠历史补全。

    按「问句开头的主体」分组而不是按文章分组：同一家公司的事实常分散在多篇里，
    "那三季度营收是多少" 正好考跨文章沿用上下文。
    """
    groups: dict[str, list[dict]] = {}
    for it in items:
        if it.get("tier") not in ("A", "B", "S") or it.get("kind") not in ("literal", "para"):
            continue
        if not it.get("gold_pages") or not it.get("gold_figures"):
            continue
        q = it["question"]
        subject = question_subject(q)
        if len(subject) < 2 or not q.startswith(subject) or _subjectless(q):
            continue
        it = {**it, "_subject": subject}
        groups.setdefault(subject, []).append(it)
    out = []
    for subject, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x["id"])
        made = 0
        seen_turn2: set[str] = set()
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                fa, fb = a.get("gold_figures") or [], b.get("gold_figures") or []
                if not fa or not fb or _norm_value(fa[0]) == _norm_value(fb[0]):
                    continue
                q1, q2 = a["question"], b["question"]
                if q1 == q2 or _subjectless(q2):
                    continue
                rest = q2[len(subject) :].lstrip("的，,、 ")
                if len(rest) < 4 or not any(w in rest for w in _METRIC_WORDS):
                    continue
                # “分别为多少”本身指代多个值，接在“那”后面读不通
                if "分别" in rest:
                    continue
                # 带单位/百分号的数字等于把答案写进追问；日期数字（9月）允许保留
                if re.search(
                    r"\d[\d,，.]*\s*(?:%|％|万亿|亿|万|元|美元|港元|倍|台|辆|股|人|吨|GW|GB|TB)",
                    rest,
                ):
                    continue
                # 追问必须直接从时间或指标接上，否则说明中间还夹着别的成分
                if not (
                    re.match(r"^\d|[一二三四]季度|^第[一二三四]季度|^上半年|^下半年|^全年|^本季度|^当季|^同比|^环比", rest)
                    or any(rest.startswith(w) for w in _METRIC_WORDS)
                ):
                    continue
                turn2 = "那" + rest
                if not _subjectless(turn2) or turn2 in seen_turn2:
                    continue
                seen_turn2.add(turn2)
                out.append(
                    {
                        "id": f"M-{_sid(subject)}-{_sid(q1 + '|' + turn2)}",
                        "tier": "M",
                        "kind": "multiturn",
                        "question": q1,
                        "article": a.get("article") or "",
                        "date": a.get("date") or b.get("date") or "",
                        "subject": subject,
                        "anchor_terms": anchor_terms(q1),
                        "lex_hit": False,
                        "expect_refusal": False,
                        "turns": [
                            {
                                "question": q1,
                                "gold_pages": sorted(set(a.get("gold_pages") or [])),
                                "gold_span": a.get("gold_span") or [],
                                "gold_spans": a.get("gold_spans") or [],
                                "gold_figures": (a.get("gold_figures") or [])[:2],
                            },
                            {
                                "question": turn2,
                                "gold_pages": sorted(set(b.get("gold_pages") or [])),
                                "gold_span": b.get("gold_span") or [],
                                "gold_spans": b.get("gold_spans") or [],
                                "gold_figures": (b.get("gold_figures") or [])[:2],
                            },
                        ],
                        **_hop_fields(a, b),
                    }
                )
                made += 1
                if made >= per_subject:
                    break
            if made >= per_subject:
                break
        if len(out) >= cap:
            break
    return out[:cap]


def upgrade_items(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """在已有脱敏验证集上升级：剔除缺主体问法，并由事实题派生多跳/多轮档。

    原始语料缺失时能做的只有这两件事——H/M 的 gold 都来自原题字段，
    不需要回到原文；也因此 M 档数量受「同主体、不同指标、问法平行」的存量限制。
    """
    kept = [it for it in items if not _subjectless(it.get("question") or "")]
    dropped = len(items) - len(kept)
    multihop = derive_multihop(kept)
    multiturn = derive_multiturn(kept)
    derived = multihop + multiturn
    max_no = max((int(it.get("no") or 0) for it in kept), default=0)
    doc_id = kept[0].get("doc_id") if kept else "src_0001"
    for n, it in enumerate(derived, start=1):
        max_no += 1
        it["no"] = max_no
        it["doc_id"] = it.get("doc_id") or doc_id
        # 公开集不含 subject 字段；派生题的主体已经写在问句里，无需额外保留
        it.pop("subject", None)
    return kept + derived, {
        "input": len(items),
        "subjectless_dropped": dropped,
        "kept": len(kept),
        "multihop": len(multihop),
        "multiturn": len(multiturn),
    }


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
    facts, refine = refine_questions(facts)
    multihop = derive_multihop(facts, lows=lows)
    multiturn = derive_multiturn(facts)
    timeline = build_timeline_items(arts)
    absent = build_absent_items(joined)
    items = quota(
        facts + multihop + multiturn + timeline + absent,
        per_tier or {"A": 180, "B": 180, "S": 280, "H": 60, "M": 60, "C": 180, "D": 60},
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
        "candidate_bad_clause": refine["bad_clause"],
        "candidate_underspecified": refine["underspecified"],
        "candidate_multi_answer": refine["multi_conflict"],
        "candidate_merged": refine["merged"],
        "candidate_deduped": refine["deduped"],
        "candidate_a": sum(1 for i in facts if i["tier"] == "A"),
        "candidate_b": sum(1 for i in facts if i["tier"] == "B"),
        "candidate_s": sum(1 for i in facts if i["tier"] == "S"),
        "candidate_timeline": len(timeline),
        "candidate_multihop": len(multihop),
        "candidate_multiturn": len(multiturn),
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
