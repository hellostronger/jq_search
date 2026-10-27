"""用脱敏验证集重建可重放的页语料。

原始语料丢失时（只留下 `eval_out/dataset_public.jsonl`），仍可以用验证集里
保留的三个字段还原一部分原文：

  question    literal 题的问句左半句逐字来自原句
  gold_match  原句里被替换成疑问词的那个数字（含单位）
  gold_pages  事实句所在的页

拼接即可得到事实句本身，例如
  "dLLM可将端侧Agent的运行速度提升多少？" + "5倍" → "dLLM可将端侧Agent的运行速度提升5倍。"

**这不是原始语料的等价替代**，它有三个无法消除的偏差：

1. 页密度：原页面通常 1000~3000 字，重放页平均只有几十字。字面锚词的
   区分度被显著放大，任何 2~3 字词都只会命中一两页。
2. 覆盖面：脱敏集只保留了被抽中的事实句，C 档（时序）和 D 档（未收录）
   没有可还原的页，A/B/S 三档约 90% 的 gold 页可还原。
3. 语序：一页内的多句话按 gold_pages 归并，丢失了原文的段落与上下文。

因此重放语料只适合做**管线机制验证**和**新旧策略的相对对比**，不能替代
原语料上的绝对指标。`--mode entity` 探针会把"短锚词收益"和"别名扩展收益"
分开统计，正是为了避免把机制验证误读成效果提升。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..core.store import KB

FACTUAL_TIERS = ("A", "B", "S")


@dataclass
class ReplayCorpus:
    pages: dict[int, list[str]] = field(default_factory=dict)
    total: int = 0
    stats: dict = field(default_factory=dict)

    def render(self) -> list[str]:
        return ["\n".join(self.pages.get(i, [])) for i in range(1, self.total + 1)]


def _left_of(question: str) -> str:
    q = (question or "").rstrip("？?")
    for suffix in ("多少", "是在什么时候"):
        if q.endswith(suffix):
            return q[: -len(suffix)]
    return q


def reconstruct(items: Iterable[dict]) -> ReplayCorpus:
    """literal 题直接还原；para 题借同一 figure 的 literal 兄弟还原，避免把改写句塞回页里。"""
    items = list(items)
    literal = [it for it in items if it["kind"] == "literal"]
    para = [it for it in items if it["kind"] == "para"]

    def key(it: dict) -> tuple:
        return (it.get("article"), it.get("gold_match"), tuple(it.get("gold_figures") or []))

    siblings: dict[tuple, list[dict]] = defaultdict(list)
    for it in literal:
        siblings[key(it)].append(it)

    pages: dict[int, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    used: set[str] = set()

    def add(source: dict, item: dict) -> None:
        used.add(item["id"])
        sentence = _left_of(source["question"]) + (source.get("gold_match") or "") + "。"
        dedupe = (source.get("article") or "", sentence)
        if dedupe in seen or len(sentence) < 8:
            return
        seen.add(dedupe)
        for page in item.get("gold_pages") or []:
            pages[int(page)].append(sentence)

    for it in literal:
        add(it, it)
    for it in para:
        for source in siblings.get(key(it), []):
            add(source, it)

    total = 0
    for it in items:
        for page in it.get("gold_pages") or []:
            total = max(total, int(page))
        span = it.get("gold_span") or []
        if len(span) == 2:
            total = max(total, int(span[1]))
    by_tier: dict[str, int] = defaultdict(int)
    for it in items:
        if it.get("id") in used:
            by_tier[str(it.get("tier"))] += 1
    filled = [p for p in pages if pages[p]]
    chars = sum(len(s) for p in filled for s in pages[p])
    stats = {
        "items": len(items),
        "reconstructed_items": len(used),
        "by_tier": dict(sorted(by_tier.items())),
        "pages": total,
        "filled_pages": len(filled),
        "sentences": sum(len(pages[p]) for p in filled),
        "avg_chars_per_filled_page": round(chars / len(filled), 1) if filled else 0.0,
    }
    return ReplayCorpus(pages=dict(pages), total=total, stats=stats)


def build_replay_kb(
    dataset: Path,
    home: Path,
    doc_id: str = "src_0001",
) -> tuple[KB, ReplayCorpus]:
    items = [json.loads(line) for line in Path(dataset).open(encoding="utf-8") if line.strip()]
    corpus = reconstruct(items)
    kb = KB(home)
    kb.write_pages(doc_id, corpus.render())
    kb.register(doc_id, Path(dataset), "-", corpus.total, "重放语料（由脱敏验证集还原）")
    return kb, corpus


def _print_stats(corpus: ReplayCorpus, home: Path) -> None:
    s = corpus.stats
    print(
        f"页 {s['pages']} · 有文本 {s['filled_pages']} 页 · 事实句 {s['sentences']} 条 · "
        f"平均 {s['avg_chars_per_filled_page']} 字/页 · 还原题 {s['reconstructed_items']}"
    )
    print("按档还原：" + ", ".join(f"{k} {v}" for k, v in s["by_tier"].items()))
    print(f"KB → {home}")
    print("注意：重放页只有事实句，密度远低于原文，绝对命中率不可与原语料比较。")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用脱敏验证集重建可重放页语料")
    parser.add_argument("--dataset", default="eval_out/dataset_public.jsonl")
    parser.add_argument("--out", default="eval_out/replay_kb")
    parser.add_argument("--doc-id", default="src_0001")
    args = parser.parse_args(argv)
    kb, corpus = build_replay_kb(Path(args.dataset), Path(args.out), args.doc_id)
    _print_stats(corpus, Path(args.out))


if __name__ == "__main__":
    main()
