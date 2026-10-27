from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

# 同一产品的型号后缀：出现这些后缀时禁止把短名并入长名，
# 否则 iPhone 与 iPhone Pro、Mate 与 Mate Pro 会被误合并。
MODEL_MARKERS = frozenset(
    {"pro", "max", "plus", "ultra", "mini", "air", "se", "duo", "turbo", "lite", "neo"}
)

ORG_SUFFIXES = ("公司", "集团", "银行", "证券", "基金", "大学", "研究院", "实验室", "政府", "委员会", "机构", "部门")
PRODUCT_SUFFIXES = ("手机", "芯片", "车型", "汽车", "平台", "系统", "模型", "设备", "指数", "期货", "合约", "产品", "服务")
METRIC_SUFFIXES = ("收入", "营收", "利润", "市值", "价格", "成本", "销量", "规模", "增长率", "增速", "占比")

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_DIGIT_RE = re.compile(r"^\d+$")


class Mention(BaseModel):
    doc_id: str
    page: int
    count: int = 1


class Entity(BaseModel):
    id: str
    canonical: str
    type: str = "OTHER"
    aliases: list[str] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    frequency: int = 0
    degree: float = 0.0
    sources: list[str] = Field(default_factory=list)


class EntityIndex(BaseModel):
    built_at: str = ""
    docs: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    alias_index: dict[str, str] = Field(default_factory=dict)
    stats: dict[str, int] = Field(default_factory=dict)

    def by_id(self, entity_id: str) -> Entity | None:
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def resolve(self, mention: str) -> Entity | None:
        entity_id = self.alias_index.get(normalize(mention))
        return self.by_id(entity_id) if entity_id else None

    def expand(self, mention: str, limit: int = 8) -> list[str]:
        """返回该 mention 对应的 canonical + 所有别名（去重、保持顺序）。"""
        entity = self.resolve(mention)
        if entity is None:
            return []
        names = [entity.canonical, *entity.aliases]
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            key = normalize(name)
            if key and key not in seen:
                seen.add(key)
                out.append(name)
        return out[:limit]

    def summary(self) -> dict:
        return {
            "built_at": self.built_at,
            "docs": len(self.docs),
            "entities": len(self.entities),
            "aliases": sum(len(e.aliases) for e in self.entities),
            "mentions": sum(len(e.mentions) for e in self.entities),
            "stats": dict(self.stats),
        }


@dataclass
class Variant:
    """一个表面形式（归一化后唯一）及其统计。图节点就是它。"""

    surface: str
    norm: str
    freq: int = 0
    rule_hits: int = 0
    pages: Counter = field(default_factory=Counter)  # (doc_id, page) -> count
    contexts: Counter = field(default_factory=Counter)  # 上下文字符 bigram
    types: Counter = field(default_factory=Counter)
    sources: set[str] = field(default_factory=set)


def normalize(text: str) -> str:
    """NFKC + 小写 + 去标点/空白/下划线；用于比较与索引，不用于展示。"""
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    return _PUNCT_RE.sub("", text)


def infer_type(surface: str) -> str:
    if not surface:
        return "OTHER"
    if surface.endswith(ORG_SUFFIXES):
        return "ORG"
    if surface.endswith(PRODUCT_SUFFIXES):
        return "PRODUCT"
    if surface.endswith(METRIC_SUFFIXES):
        return "METRIC"
    return "OTHER"


def add_mention(
    variants: dict[str, Variant],
    surface: str,
    doc_id: str,
    page: int,
    context: str = "",
    source: str = "rule",
) -> None:
    norm = normalize(surface)
    if len(norm) < 2 or _DIGIT_RE.match(norm):
        return
    variant = variants.get(norm)
    if variant is None:
        variant = Variant(surface=surface.strip(), norm=norm)
        variants[norm] = variant
    variant.freq += 1
    variant.sources.add(source)
    if source != "ngram":
        variant.rule_hits += 1
    variant.pages[(doc_id, page)] += 1
    variant.types[infer_type(norm)] += 1
    for gram in _char_bigrams(context):
        variant.contexts[gram] += 1


def _char_bigrams(text: str) -> list[str]:
    text = normalize(text)
    if len(text) < 2:
        return []
    return [text[i : i + 2] for i in range(len(text) - 1)]


def _context_set(counter: Counter, limit: int = 300) -> set[str]:
    if len(counter) <= limit:
        return set(counter)
    return {gram for gram, _ in counter.most_common(limit)}


def _jaccard(a: Counter, b: Counter) -> float:
    sa, sb = _context_set(a), _context_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cooccurrence(a: Variant, b: Variant) -> float:
    shared_pages = set(a.pages) & set(b.pages)
    if not shared_pages:
        return 0.0
    shared = sum(min(a.pages[p], b.pages[p]) for p in shared_pages)
    smaller = min(sum(a.pages.values()), sum(b.pages.values()))
    return shared / max(1, smaller)


def _dominant_type(variant: Variant) -> str:
    if not variant.types:
        return "OTHER"
    return variant.types.most_common(1)[0][0]


def _type_factor(a: Variant, b: Variant) -> float:
    ta, tb = _dominant_type(a), _dominant_type(b)
    if ta == tb:
        return 1.0
    if ta == "OTHER" or tb == "OTHER":
        return 0.9
    return 0.75


def _cannot_link(a: str, b: str) -> bool:
    """短名是长名前缀且剩余部分是型号后缀时，视为不同实体。"""
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short or not long.startswith(short) or short == long:
        return False
    rest = long[len(short) :].strip("-_ .")
    if not rest:
        return False
    tokens = [t for t in re.split(r"[\s\-_·]+", rest) if t]
    return bool(tokens) and all(t in MODEL_MARKERS for t in tokens)


def _short_alnum_block(a: str, b: str) -> bool:
    """型号/代码类短串（≤8 字符且含数字）只允许完全一致。

    DDR4/DDR5、BU2701/FU2701、ROIC11.15/ROIC 字面相近但指代不同；一旦并簇，
    查询扩展就会把错误别名带进证据页。数字序列不同直接拒绝，数字相同但
    字面差异超过 10% 也拒绝（如 E5 与 E7、4090 与 4090D）。
    """
    if a == b or min(len(a), len(b)) > 8:
        return False
    if not any(ch.isdigit() for ch in a) and not any(ch.isdigit() for ch in b):
        return False
    digits_a = "".join(ch for ch in a if ch.isdigit())
    digits_b = "".join(ch for ch in b if ch.isdigit())
    if digits_a != digits_b:
        return True
    return SequenceMatcher(None, a, b).ratio() < 0.9


def _block_keys(norm: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = {("first", norm[:1]), ("suffix", norm[-2:])}
    if norm.isascii():
        keys.add(("first2", norm[:2]))
    return keys


def _candidate_pairs(nodes: dict[str, Variant], max_block_size: int) -> set[tuple[str, str]]:
    """分块 + 包含关系生成候选对，避免全量 O(n²)。"""
    blocks: dict[tuple[str, str], list[Variant]] = defaultdict(list)
    for variant in nodes.values():
        for key in _block_keys(variant.norm):
            blocks[key].append(variant)

    pairs: set[tuple[str, str]] = set()
    for group in blocks.values():
        group = sorted(group, key=lambda v: (-v.freq, v.norm))[:max_block_size]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.add(tuple(sorted((group[i].norm, group[j].norm))))

    for norm in nodes:
        for length in range(2, len(norm)):
            for start in range(0, len(norm) - length + 1):
                sub = norm[start : start + length]
                if sub != norm and sub in nodes:
                    pairs.add(tuple(sorted((norm, sub))))
    return pairs


def edge_weight(a: Variant, b: Variant) -> float:
    if _cannot_link(a.norm, b.norm) or _short_alnum_block(a.norm, b.norm):
        return 0.0
    string = SequenceMatcher(None, a.norm, b.norm).ratio()
    containment = 0.0
    prefix = 0.0
    if a.norm in b.norm or b.norm in a.norm:
        containment = min(len(a.norm), len(b.norm)) / max(len(a.norm), len(b.norm))
    if a.norm.startswith(b.norm) or b.norm.startswith(a.norm):
        # 中文实体简称通常就是全称的前缀（宁德时代 / 宁德时代新能源…），
        # 这条信号比字面相似度更可靠，单独给权重。
        prefix = 1.0
    context = _jaccard(a.contexts, b.contexts)
    cooccur = _cooccurrence(a, b)
    return (
        0.30 * string
        + 0.10 * context
        + 0.20 * cooccur
        + 0.15 * containment
        + 0.25 * prefix
    ) * _type_factor(a, b)


def _label_propagation(
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    freq: dict[str, int],
    iterations: int = 20,
    self_bias: float = 0.15,
) -> dict[str, str]:
    """确定性标签传播：按频次排序、同分优先当前标签、再按字典序。"""
    labels = {node: node for node in nodes}
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), weight in edges.items():
        adjacency[a][b] = weight
        adjacency[b][a] = weight

    order = sorted(nodes, key=lambda n: (-freq.get(n, 0), n))
    for _ in range(iterations):
        changed = False
        for node in order:
            scores: dict[str, float] = defaultdict(float)
            scores[labels[node]] += self_bias
            for neighbor, weight in adjacency[node].items():
                scores[labels[neighbor]] += weight
            best = sorted(
                scores.items(),
                key=lambda kv: (-kv[1], kv[0] != labels[node], kv[0]),
            )[0][0]
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    return labels


def _pick_canonical(members: list[Variant]) -> Variant:
    """优先完整名称：最长成员频次不低于最频繁成员的 30% 时选最长名。"""
    by_freq = max(members, key=lambda v: v.freq)
    longest = max(members, key=lambda v: (len(v.norm), v.freq))
    if longest.freq >= 0.3 * max(1, by_freq.freq):
        return longest
    return by_freq


def align_variants(
    variants: dict[str, Variant],
    docs: list[str],
    cfg: dict | None = None,
) -> EntityIndex:
    """把表面形式图切成实体簇，产出别名索引。纯确定性，不调用 LLM。"""
    cfg = cfg or {}
    min_frequency = int(cfg.get("min_frequency", 2))
    threshold = float(cfg.get("similarity_threshold", 0.55))
    max_block_size = int(cfg.get("max_block_size", 200))

    nodes = {
        norm: variant
        for norm, variant in variants.items()
        if variant.freq >= min_frequency or variant.rule_hits > 0
    }
    if not nodes:
        return EntityIndex(
            built_at=_now(),
            docs=list(docs),
            stats={"variants": len(variants), "kept": 0, "edges": 0},
        )

    edges: dict[tuple[str, str], float] = {}
    for a, b in _candidate_pairs(nodes, max_block_size):
        weight = edge_weight(nodes[a], nodes[b])
        if weight >= threshold:
            edges[(a, b)] = weight

    freq = {norm: variant.freq for norm, variant in nodes.items()}
    labels = _label_propagation(sorted(nodes), edges, freq)

    communities: dict[str, list[str]] = defaultdict(list)
    for norm, label in labels.items():
        communities[label].append(norm)

    # 簇内边权重和 = 该实体聚类的紧密程度；跨簇边不计入
    cohesion: dict[str, float] = defaultdict(float)
    for (a, b), weight in edges.items():
        if labels[a] == labels[b]:
            cohesion[labels[a]] += weight

    rows: list[tuple[list[Variant], float]] = []
    for label, members_norm in communities.items():
        members = [nodes[n] for n in members_norm]
        total_freq = sum(v.freq for v in members)
        if total_freq < min_frequency:
            continue
        rows.append((members, cohesion.get(label, 0.0)))

    rows.sort(key=lambda item: (-sum(v.freq for v in item[0]), _pick_canonical(item[0]).norm))

    entities: list[Entity] = []
    alias_index: dict[str, str] = {}
    for position, (members, degree) in enumerate(rows, start=1):
        canonical = _pick_canonical(members)
        entity_id = f"ent_{position:04d}"
        aliases = sorted(
            (v for v in members if v.norm != canonical.norm),
            key=lambda v: (-v.freq, -len(v.norm), v.norm),
        )
        page_counts: Counter = Counter()
        sources: set[str] = set()
        for variant in members:
            page_counts.update(variant.pages)
            sources.update(variant.sources)
        mentions = [
            Mention(doc_id=doc_id, page=page, count=count)
            for (doc_id, page), count in sorted(page_counts.items())
        ]
        entity = Entity(
            id=entity_id,
            canonical=canonical.surface,
            type=_dominant_type(canonical),
            aliases=[v.surface for v in aliases],
            mentions=mentions,
            frequency=sum(v.freq for v in members),
            degree=round(degree, 4),
            sources=sorted(sources),
        )
        entities.append(entity)
        for variant in members:
            alias_index[variant.norm] = entity_id

    return EntityIndex(
        built_at=_now(),
        docs=list(docs),
        entities=entities,
        alias_index=alias_index,
        stats={
            "variants": len(variants),
            "kept": len(nodes),
            "edges": len(edges),
            "entities": len(entities),
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
