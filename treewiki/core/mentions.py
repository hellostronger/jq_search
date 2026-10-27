from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .entities import METRIC_SUFFIXES, ORG_SUFFIXES, PRODUCT_SUFFIXES, normalize

CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-&']{1,}")
QUOTED = re.compile(
    r"《([^》\n]{2,30})》|「([^」\n]{2,30})」|“([^”\n]{2,30})”|‘([^’\n]{2,30})’"
)
ORG_ANCHOR = re.compile(r"(?:公司|集团|银行|证券|基金|大学|研究院|实验室|委员会|交易所|管理局)")

BOS = "<s>"
EOS = "</s>"

LATIN_STOP = frozenset(
    """
the and for with that this from are was were has have had not but you your our their its will
can may all any who what when where which while into than then there here they them we our
""".split()
)

# 这些字出现在词首/词尾时几乎总是句法成分而非实体边界
BAD_EDGE = frozenset("的了着过而且却则就都也很大太更最还又再并等被把将对从向于是和与及或会要能可需")
BAD_START = BAD_EDGE | frozenset("第每各某该本其此这那哪几多少")

CJK_STOPWORDS = frozenset(
    """
我们 你们 他们 她们 它们 自己 大家 其中 以及 因为 所以 但是 如果 虽然 由于 通过 根据 关于
对于 此外 同时 目前 已经 可以 可能 应该 需要 必须 这是 这些 那些 这个 那个 一个 一些 没有
不是 就是 还有 而且 或者 并且 然而 因此 以上 以下 之后 之前 方面 情况 问题 内容 相关 进行
表示 认为 指出 显示 出现 发生 成为 作为 包括 之一 同比 环比 增长 下降 上升 减少 增加 达到
超过 分别 主要 重要 其他 另外 具体 如下 上述 期间 以来 同期 去年 今年 明年 预计 左右
之间 本轮 本次 该行 该公司 有限公司 公司 集团 银行 证券 基金 大学 研究院 实验室 政府
委员会 机构 部门 手机 芯片 车型 汽车 平台 系统 模型 设备 指数 期货 合约 产品 服务
收入 营收 利润 市值 价格 成本 销量 规模 增长率 增速 占比 市场 行业 企业 用户 客户 数据
报告 消息 公告 记者 报道 分析 研究 投资 融资 业务 战略 计划 项目 原因 结果 影响 方式
售价 备货 产能 出货 订单 股价 业绩 毛利率 份额 目标 工厂 车型 首批 全年 预期 机构
调研 分析师 三季度 二季度 一季度 四季度 上半年 下半年 本月 上月 本周 上周 当日 同期
""".split()
)

BOUNDARY_WORDS = (
    "根据", "按照", "截至", "其中", "以及", "通过", "由于", "对于", "关于", "相比",
    "包括", "来自", "属于", "持有", "收购", "投资", "控股", "参股", "旗下",
)
# 句子层面的连接词：出现在名称前面时基本可以确定是分句边界
SEGMENT_WORDS = (
    "认为", "表示", "指出", "显示", "宣布", "披露", "公告", "签署", "签订", "发布",
    "报告", "预计", "分析", "消息", "根据", "同时", "此外", "截至", "其中", "以及",
    "通过", "由于", "对于", "关于", "相比", "来自", "包括", "旗下",
)
LEGAL_PREFIXES = ("集团", "股份", "有限", "责任", "控股", "实业")
BOUNDARY_CHARS = "的了着过和与及或对从向于是在为把被将由并等"
STRONG_SEP = frozenset("的了着过和与及或对从向把被将并而则就都也还很更最等吗呢吧")
_SUFFIX_MAX = 20

_SPAN_PRIORITY = {"quote": 3, "suffix": 2, "latin": 1, "ngram": 0}


@dataclass(frozen=True)
class Span:
    surface: str
    start: int
    end: int
    source: str = "rule"


def _prefix(flags: list[bool]) -> list[int]:
    out = [0] * (len(flags) + 1)
    for i, flag in enumerate(flags):
        out[i + 1] = out[i] + (1 if flag else 0)
    return out


def _blocked(prefix: list[int], start: int, end: int) -> bool:
    return prefix[end] - prefix[start] > 0


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    """按来源优先级保留不重叠的片段；引用 > 机构后缀 > 拉丁词 > n-gram。"""
    ordered = sorted(
        spans, key=lambda s: (-_SPAN_PRIORITY.get(s.source, 0), s.start, -(s.end - s.start))
    )
    taken = [False] * max((s.end for s in spans), default=0)
    out: list[Span] = []
    for span in ordered:
        if span.end <= span.start:
            continue
        if any(taken[span.start : span.end]):
            continue
        for i in range(span.start, span.end):
            taken[i] = True
        out.append(span)
    return sorted(out, key=lambda s: s.start)


def _clean_latin(raw: str) -> str:
    return raw.rstrip(".-_+&'")


def _suffix_span(text: str, anchor_end: int) -> tuple[int, str] | None:
    """从机构后缀向前回溯，截出「名称+后缀」；最长 20 字。"""
    start = anchor_end
    limit = max(0, anchor_end - _SUFFIX_MAX)
    while start > limit and (text[start - 1].isalnum() or text[start - 1] in "·&"):
        start -= 1
    surface = text[start:anchor_end]

    cut = 0
    for word in SEGMENT_WORDS:
        pos = surface.rfind(word)
        if pos >= 0:
            cut = max(cut, pos + len(word))
    surface = surface[cut:]

    for idx in range(len(surface) - 1, 0, -1):
        if surface[idx - 1] in STRONG_SEP:
            surface = surface[idx:]
            break

    for _ in range(2):
        stripped = surface
        for word in BOUNDARY_WORDS:
            if stripped.startswith(word):
                stripped = stripped[len(word) :]
                break
        else:
            stripped = stripped.lstrip(BOUNDARY_CHARS)
        if stripped == surface:
            break
        surface = stripped

    if len(surface) < 3 or not any("\u4e00" <= ch <= "\u9fff" for ch in surface):
        return None
    if surface in CJK_STOPWORDS or surface.startswith(LEGAL_PREFIXES):
        return None
    return anchor_end - len(surface), surface


def rule_spans(text: str) -> list[Span]:
    """高精度片段：引号内名称、拉丁词、带机构后缀的中文名。"""
    spans: list[Span] = []
    for m in QUOTED.finditer(text):
        value = next((g for g in m.groups() if g), "").strip()
        if len(normalize(value)) >= 2:
            spans.append(Span(value, m.start() + 1, m.end() - 1, "quote"))

    for m in LATIN.finditer(text):
        value = _clean_latin(m.group())
        if len(value) < 2 or value.lower() in LATIN_STOP:
            continue
        spans.append(Span(value, m.start(), m.start() + len(value), "latin"))

    for m in ORG_ANCHOR.finditer(text):
        hit = _suffix_span(text, m.end())
        if hit:
            start, surface = hit
            spans.append(Span(surface, start, m.end(), "suffix"))
    return _dedupe_spans(spans)


class NgramMiner:
    """中文新词发现：词频 + 左右邻接熵，两遍扫描（先统计，再抽取）。"""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.max_len = max(2, int(cfg.get("ngram_max", 6)))
        self.min_freq = max(2, int(cfg.get("ngram_min_frequency", 3)))
        self.min_entropy = float(cfg.get("ngram_min_entropy", 0.5))
        self._counts: Counter[str] = Counter()
        self._left: dict[str, Counter[str]] = {}
        self._right: dict[str, Counter[str]] = {}
        self._selected: dict[str, float] = {}

    def observe(self, text: str) -> None:
        if not text:
            return
        for m in CJK_RUN.finditer(text):
            run, base = m.group(), m.start()
            for i in range(len(run)):
                for length in range(1, self.max_len + 1):
                    if i + length > len(run):
                        break
                    start = base + i
                    end = start + length
                    gram = run[i : i + length]
                    self._counts[gram] += 1
                    if length == 1:
                        continue
                    left = text[start - 1] if start > 0 else BOS
                    right = text[end] if end < len(text) else EOS
                    self._left.setdefault(gram, Counter())[left] += 1
                    self._right.setdefault(gram, Counter())[right] += 1

    def finalize(self) -> dict[str, float]:
        selected: dict[str, float] = {}
        for gram, freq in self._counts.items():
            if len(gram) < 2 or freq < self.min_freq:
                continue
            if not self._acceptable(gram):
                continue
            score = self._boundary_score(gram)
            if score is None:
                continue
            selected[gram] = score
        self._selected = selected
        return selected

    def _acceptable(self, gram: str) -> bool:
        if gram in CJK_STOPWORDS:
            return False
        if gram[0] in BAD_START or gram[-1] in BAD_EDGE:
            return False
        return len(set(gram)) > 1

    def _boundary_score(self, gram: str) -> float | None:
        left = _entropy(self._left.get(gram, Counter()))
        right = _entropy(self._right.get(gram, Counter()))
        score = min(left, right)
        if score >= self.min_entropy:
            return round(score, 4)
        # 机构/产品/指标后缀是强信号：允许左右熵偏低（报告体里常固定搭配）
        if gram.endswith(ORG_SUFFIXES + PRODUCT_SUFFIXES + METRIC_SUFFIXES):
            return round(max(score, self.min_entropy), 4)
        return None

    @property
    def candidates(self) -> dict[str, float]:
        return dict(self._selected)

    def extract(self, text: str, extra: Iterable[Span] = ()) -> list[Span]:
        """最长匹配抽取；与高精度片段重叠的位置跳过。"""
        spans = list(extra)
        if not self._selected or not text:
            return _dedupe_spans(spans)

        flags = [False] * len(text)
        for span in spans:
            for i in range(max(0, span.start), min(len(text), span.end)):
                flags[i] = True
        prefix = _prefix(flags)

        for m in CJK_RUN.finditer(text):
            run, base = m.group(), m.start()
            i = 0
            while i < len(run):
                if flags[base + i]:
                    i += 1
                    continue
                picked = 0
                upper = min(self.max_len, len(run) - i)
                for length in range(upper, 1, -1):
                    if _blocked(prefix, base + i, base + i + length):
                        continue
                    gram = run[i : i + length]
                    if gram in self._selected:
                        picked = length
                        break
                if picked:
                    spans.append(Span(run[i : i + picked], base + i, base + i + picked, "ngram"))
                    i += picked
                else:
                    i += 1
        return _dedupe_spans(spans)


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 1:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())
