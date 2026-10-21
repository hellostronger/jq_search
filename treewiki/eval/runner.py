"""评测执行与判分：retrieval 模式零成本全量跑，full 模式并发调用真实模型抽样跑。"""

from __future__ import annotations

import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import load_config
from ..core.llm import LLM
from ..core.store import KB
from ..pipelines.query import (
    _dedupe,
    _lexical_pages,
    _nodes_for_pages,
    _pages_of,
    anchor_terms,
    iter_nodes,
    query_raw,
)
from .dataset import read_jsonl, write_jsonl

_REFUSE = re.compile(
    r"(未提及|未收录|未明确|未给出|未披露|未出现|未找到|未见|不包含|没有.*找到|没有.*相关"
    r"|找不到|证据不足|无法回答|文档中没)"
)
_NUM = re.compile(r"\d[\d,，.]*")


def _norm(text: str) -> str:
    return re.sub(r"[,，.．\s]", "", text).lower()


def cited_pages(result: dict) -> list[int]:
    out = []
    for c in result.get("citations") or []:
        out.extend(int(p) for p in c.get("pages") or [])
    return sorted(set(out))


def grade(item: dict, result: dict, lows: list[str] | None = None) -> dict:
    gold = set(item["gold_pages"])
    got = set(cited_pages(result))
    answer = _norm(result.get("answer") or "")
    figures = item.get("gold_figures") or []
    hit_all = [f for f in figures if _norm(f) in answer]
    page_hit = bool(gold & got)
    near_hit = bool(gold & {p + d for p in gold for d in (-1, 0, 1)} & got)
    span = item.get("gold_span") or []
    article_hit = bool(span) and any(span[0] <= p <= span[1] for p in got)
    # 引用页原文里确实出现这个数字，说明引用撑得住，即使不是 gold 那一页
    verifiable = bool(lows) and bool(figures) and any(
        all(_norm(f) in _norm(lows[p - 1]) for f in figures) for p in got if 0 < p <= len(lows)
    )
    # 诚实拒答允许顺带引用邻近页（"没找到 X，只找到 Y [p234]"），不能当成幻觉
    refused = bool(_REFUSE.search(result.get("answer") or ""))
    tier = item["tier"]
    recall = len(gold & got) / len(gold) if gold else 0.0
    spans = item.get("gold_spans") or []
    coverage = (
        sum(1 for a, b in spans if any(a <= p <= b for p in got)) / len(spans) if spans else 0.0
    )
    figure_ok = bool(hit_all) or not figures
    grounded = page_hit or article_hit or verifiable
    if tier == "D":
        ok = refused
    elif tier == "C":
        ok = coverage >= 0.5
    else:
        ok = figure_ok and grounded
    return {
        "id": item["id"],
        "tier": tier,
        "question": item["question"],
        "ok": ok,
        "page_hit": page_hit,
        "near_page_hit": near_hit,
        "article_hit": article_hit,
        "citation_verifiable": verifiable,
        "unsupported_citation": bool(got) and not grounded and not refused,
        "gold_page_recall": round(recall, 3),
        "article_coverage": round(coverage, 3),
        "figure_hit": len(hit_all),
        "figure_total": len(figures),
        "refused": refused,
        "grader": 2,
        "gold_pages": sorted(gold),
        "cited_pages": sorted(got),
        "read_pages": result.get("read_pages") or [],
        "rounds": len(result.get("rounds") or []),
        "usage": result.get("usage") or {},
        "answer_head": (result.get("answer") or "")[:240],
        "answer_status": result.get("answer_status"),
        "lexical": result.get("lexical"),
        "fallback": result.get("fallback"),
    }


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


def _ngram_terms(question: str) -> list[str]:
    """近似分词：把中文串切成 4 字窗口，取最长的几个，替代整句当一个锚词。"""
    grams = []
    for run in _CJK_RUN.findall(question):
        run = run.rstrip("多少是在什么时候？?")
        grams += [run[i : i + 4] for i in range(0, max(1, len(run) - 3))][:3]
    return [g for g in dict.fromkeys(grams) if len(g) == 4][:4]


def _pages_with(lows: list[str], terms: list[str], mode: str, cap: int) -> list[int]:
    if not terms:
        return []
    scored = []
    for i, t in enumerate(lows, start=1):
        present = [term in t for term in terms]
        n = sum(present)
        if (mode == "and" and n == len(terms)) or (mode == "or" and n):
            scored.append((i, n, t.count(terms[0])))
    ranked = sorted(scored, key=lambda x: (-x[1], -x[2], x[0]))
    return [p for p, _, _ in ranked[:cap]]


def retrieval_probe(kb: KB, doc_id: str, items: list[dict], cap: int, read_cap: int = 8) -> list[dict]:
    """零 LLM：只测字面证据这一层，顺带对比三种锚词策略能救回多少标准引用。"""
    total = kb.page_count(doc_id)
    lows = [t.lower() for t in kb.read_pages(doc_id, 1, total)]
    tree = kb.load_tree(doc_id)
    leaves = [n for n, _ in iter_nodes(tree.structure) if not n.nodes]
    out = []
    for it in items:
        terms = anchor_terms(it["question"])
        grams = _ngram_terms(it["question"])
        wide = _lexical_pages(lows, terms, 10_000)
        capped = _lexical_pages(lows, terms, cap)
        nodes = _nodes_for_pages(leaves, capped, read_cap)
        covered = sorted({p for n in nodes for p in range(n.start_index, n.end_index + 1)})
        gold = set(it["gold_pages"])
        out.append(
            {
                "id": it["id"],
                "tier": it["tier"],
                "terms": terms,
                "ngram_terms": grams,
                "lex_wide_hit": bool(gold & set(wide)),
                "lex_capped_hit": bool(gold & set(capped)),
                "lex_wide_pages": len(wide),
                "or_capped_hit": bool(gold & set(_pages_with(lows, terms, "or", cap))),
                "ngram_hit": bool(gold & set(_pages_with(lows, grams, "and", cap))),
                "node_cover_hit": bool(gold & set(covered)),
                "gold_pages": it["gold_pages"],
            }
        )
    return out


def floor_probe(
    kb: KB, doc_id: str, items: list[dict], cap: int, max_pages: int = 30, read_cap: int = 8
) -> list[dict]:
    """零 LLM：复刻 query_raw 里字面证据那一段，看证据页在哪一层被砍掉。

    lex 命中 → 页映射成节点（_spread_sample 只留 8 个）→ 节点展开成页（max_pages 截断）。
    """
    total = kb.page_count(doc_id)
    lows = [t.lower() for t in kb.read_pages(doc_id, 1, total)]
    leaves = [n for n, _ in iter_nodes(kb.load_tree(doc_id).structure) if not n.nodes]
    rows = []
    for it in items:
        gold = set(it["gold_pages"])
        terms = anchor_terms(it["question"])
        lex_pages = _lexical_pages(lows, terms, cap)
        lex_nodes = _nodes_for_pages(leaves, lex_pages, read_cap)
        floor = set(_pages_of(_dedupe(lex_nodes), total, max_pages))
        node_pages = {p for n in lex_nodes for p in range(n.start_index, n.end_index + 1)}
        lex_hit = bool(gold & set(lex_pages))
        node_hit = bool(gold & node_pages)
        rows.append(
            {
                "id": it["id"],
                "tier": it["tier"],
                "terms": terms,
                "lex_hit": lex_hit,
                "node_hit": node_hit,
                "floor_hit": bool(gold & floor),
                "lost_at_node": lex_hit and not node_hit,
                "lost_at_page_cap": node_hit and not (gold & floor),
                "n_nodes": len(lex_nodes),
                "n_floor_pages": len(floor),
                "gold_pages": it["gold_pages"],
            }
        )
    return rows


def _summarize_floor(rows: list[dict]) -> dict:
    tiers = sorted({r["tier"] for r in rows})
    out = {}
    for t in tiers:
        sub = [r for r in rows if r["tier"] == t]
        n = len(sub)
        out[t] = {
            "n": n,
            "lex_hit": round(sum(r["lex_hit"] for r in sub) / n, 3),
            "floor_hit": round(sum(r["floor_hit"] for r in sub) / n, 3),
            "lost_at_node": sum(r["lost_at_node"] for r in sub),
            "lost_at_page_cap": sum(r["lost_at_page_cap"] for r in sub),
        }
    out["all"] = {
        "n": len(rows),
        "lex_hit": round(sum(r["lex_hit"] for r in rows) / len(rows), 3),
        "floor_hit": round(sum(r["floor_hit"] for r in rows) / len(rows), 3),
        "lost_at_node": sum(r["lost_at_node"] for r in rows),
        "lost_at_page_cap": sum(r["lost_at_page_cap"] for r in rows),
    }
    return out


def tree_probe(kb: KB, doc_id: str, items: list[dict]) -> list[dict]:
    """零 LLM 的目录信息量探针：标准页所属叶节点的摘要里，有没有关键词和那个数字。

    分清楚两类失败：摘要里根本没有线索 = 索引层缺陷；摘要有线索但模型没读到 = 遍历缺陷。
    """
    leaves = [n for n, _ in iter_nodes(kb.load_tree(doc_id).structure) if not n.nodes]
    rows = []
    for it in items:
        gold = set(it["gold_pages"])
        inside = [n for n in leaves if any(n.start_index <= p <= n.end_index for p in gold)]
        key = (it.get("subject") or it.get("entity") or it.get("key_term") or "").lower()
        figs = [_norm(f) for f in it.get("gold_figures") or [] if f]
        rows.append(
            {
                "id": it["id"],
                "tier": it["tier"],
                "page_in_tree": bool(inside),
                "leaf_has_key": bool(key) and any(key in (n.summary or "").lower() for n in inside),
                "leaf_has_figure": bool(figs)
                and any(f in _norm(n.summary or "") for n in inside for f in figs),
                "key": key,
                "n_leaves": len(inside),
            }
        )
    return rows


def _summarize_tree(rows: list[dict]) -> dict:
    out = {}
    for tier in sorted({r["tier"] for r in rows}):
        g = [r for r in rows if r["tier"] == tier]
        n = len(g)
        out[tier] = {
            "n": n,
            "page_in_tree": round(sum(r["page_in_tree"] for r in g) / n, 3),
            "leaf_has_key": round(sum(r["leaf_has_key"] for r in g) / n, 3),
            "leaf_has_figure": round(sum(r["leaf_has_figure"] for r in g) / n, 3),
        }
    return out


def run_full(
    items: list[dict],
    doc_id: str,
    out_path: Path,
    workers: int = 4,
    home: Path | None = None,
) -> list[dict]:
    cfg = load_config(home)
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [i for i in items if i["id"] not in done]
    local = threading.local()

    def deps():
        if not hasattr(local, "llm"):
            local.kb = KB(kb_home_of(home))
            local.llm = LLM(cfg)
            local.lows = [t.lower() for t in local.kb.read_pages(doc_id, 1, local.kb.page_count(doc_id))]
        return local.kb, local.llm, local.lows

    def one(item: dict) -> dict:
        kb, llm, lows = deps()
        t0 = time.time()
        try:
            result = query_raw(item.get("doc_id") or doc_id, item["question"], kb, llm, cfg)
            row = grade(item, result, lows)
        except Exception as e:  # 评测里一条失败不该终止整轮
            row = {
                **grade(item, {"answer": f"异常：{e}", "citations": []}, lows),
                "error": str(e)[:200],
            }
        row["seconds"] = round(time.time() - t0, 1)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, todo):
            rows.append(row)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{len(rows)}/{len(todo)}] {row['tier']} {'ok ' if row['ok'] else 'FAIL'} "
                f"{row['seconds']}s r{row['rounds']} 引用{len(row['cited_pages'])}页 {row['question'][:32]}",
                flush=True,
            )
    return rows


def kb_home_of(home: Path | None) -> Path:
    from ..config import kb_home

    return home or kb_home()


def _regrade(row: dict) -> dict:
    """旧结果文件按新口径回填两个派生标记：诚实拒答允许顺带引用邻近页。"""
    if row.get("grader") == 2:
        return row
    refused = bool(_REFUSE.search(row.get("answer_head") or ""))
    grounded = row["page_hit"] or row.get("article_hit") or row.get("citation_verifiable")
    row = {
        **row,
        "refused": refused,
        "unsupported_citation": bool(row.get("cited_pages")) and not grounded and not refused,
    }
    if row["tier"] == "D":
        row["ok"] = refused
    return row


def summarize(rows: list[dict]) -> dict[str, Any]:
    rows = [_regrade(r) for r in rows]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["tier"], []).append(r)
    report = {}
    for tier, group in sorted(by.items()):
        secs = [g.get("seconds") for g in group if g.get("seconds")]
        report[tier] = {
            "n": len(group),
            "errors": sum(1 for g in group if g.get("error")),
            "ok": sum(1 for g in group if g["ok"]),
            "ok_rate": round(sum(1 for g in group if g["ok"]) / len(group), 3),
            "page_hit_rate": round(sum(1 for g in group if g["page_hit"]) / len(group), 3),
            "article_hit_rate": round(sum(1 for g in group if g.get("article_hit")) / len(group), 3),
            "citation_verifiable_rate": round(
                sum(1 for g in group if g.get("citation_verifiable")) / len(group), 3
            ),
            "unsupported_citation_rate": round(
                sum(1 for g in group if g.get("unsupported_citation")) / len(group), 3
            ),
            "near_page_hit_rate": round(sum(1 for g in group if g["near_page_hit"]) / len(group), 3),
            "refused_rate": round(sum(1 for g in group if g["refused"]) / len(group), 3),
            "median_article_coverage": (
                statistics.median(g.get("article_coverage") or 0 for g in group)
                if tier == "C"
                else None
            ),
            "median_rounds": statistics.median(g["rounds"] for g in group),
            "median_seconds": round(statistics.median(secs), 1) if secs else None,
            "median_calls": statistics.median(g["usage"].get("calls", 0) for g in group),
        }
    return report


def show_failures(path: Path, per_tier: int = 4) -> None:
    for tier in ("A", "B", "S", "C", "D"):
        g = [r for r in map(_regrade, read_jsonl(path)) if r["tier"] == tier and not r["ok"]]
        if not g:
            continue
        print(f"\n=== {tier} 失败 {len(g)} 条，示例 {min(per_tier, len(g))} 条")
        for r in g[:per_tier]:
            print(
                f"  Q: {r['question'][:44]}\n"
                f"     gold={r['gold_pages'][:6]} cited={r['cited_pages'][:6]} "
                f"fig={r['figure_hit']}/{r['figure_total']} rounds={r['rounds']} "
                f"lex_pages={(r.get('lexical') or {}).get('pages', [])[:6]}\n"
                f"     ans: {r['answer_head'][:120]}"
            )


def _cause(row: dict) -> str:
    """把一条失败归到链路上的某一环：锚词→证据页→截断→引用→作答。"""
    row = _regrade(row)
    if row.get("error"):
        return "调用失败（429 / 超时）"
    lex = set((row.get("lexical") or {}).get("pages") or [])
    gold = set(row.get("gold_pages") or [])
    read = set(row.get("read_pages") or [])
    cited = set(row.get("cited_pages") or [])
    if row["tier"] == "D":
        return "正确拒答" if row["ok"] else (
            "把邻近事实当成答案" if cited else "编造了数字"
        )
    if row["ok"] and row["page_hit"]:
        return "命中 gold 页"
    if row["ok"]:
        return "引用别的页但撑得住（重复报道）"
    if not gold & lex:
        return "锚词没落到证据页"
    if not gold & read:
        return "证据页在节点取样/页数截断里丢了"
    if not gold & cited:
        return "读到了原文，引用没落到 gold 页"
    return "读到了也引对了，答案内容不对"


def cause_table(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        c = _cause(r)
        out.setdefault(r["tier"], {}).setdefault(c, 0)
        out[r["tier"]][c] += 1
    return out


def main(
    dataset_path: Path,
    out_path: Path,
    mode: str,
    workers: int = 4,
    limit: int = 0,
    tiers: str = "A,B,S,C,D",
    home: Path | None = None,
) -> None:
    items = read_jsonl(dataset_path)
    want = sorted({t.strip() for t in tiers.split(",") if t.strip()})
    items = [i for i in items if i["tier"] in want]
    if limit:
        per = max(1, limit // len(want))
        picked = []
        for t in want:
            picked.extend([i for i in items if i["tier"] == t][:per])
        items = picked
    cfg = load_config(home)
    doc_id = items[0].get("doc_id") or "src_0001"
    if mode == "report":
        rows = read_jsonl(out_path)
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
        print("\n失败归因：")
        print(json.dumps(cause_table(rows), ensure_ascii=False, indent=2))
        show_failures(out_path)
        return
    if mode == "retrieval":
        kb = KB(home or kb_home_of(home))
        rows = retrieval_probe(
            kb, doc_id, items, int(cfg["query"].get("lexical_pages", 12))
        )
        write_jsonl(out_path, rows)
        print(json.dumps(_summarize_probe(rows), ensure_ascii=False, indent=2))
        return
    if mode == "floor":
        kb = KB(kb_home_of(home))
        rows = floor_probe(
            kb, doc_id, items, int(cfg["query"].get("lexical_pages", 12)),
            int(cfg["query"].get("max_pages", 30)),
        )
        write_jsonl(out_path, rows)
        print(json.dumps(_summarize_floor(rows), ensure_ascii=False, indent=2))
        return
    if mode == "tree":
        kb = KB(kb_home_of(home))
        rows = tree_probe(kb, doc_id, items)
        write_jsonl(out_path, rows)
        print(json.dumps(_summarize_tree(rows), ensure_ascii=False, indent=2))
        return
    rows = run_full(items, doc_id, out_path, workers, home)
    all_rows = read_jsonl(out_path)
    print(json.dumps(summarize(all_rows), ensure_ascii=False, indent=2))
    print("\n失败归因：")
    print(json.dumps(cause_table(all_rows), ensure_ascii=False, indent=2))


def _summarize_probe(rows: list[dict]) -> dict:
    out = {}
    for tier in sorted({r["tier"] for r in rows}):
        g = [r for r in rows if r["tier"] == tier]
        n = len(g)
        out[tier] = {
            "n": n,
            "and_wide_hit": round(sum(r["lex_wide_hit"] for r in g) / n, 3),
            "and_capped_hit": round(sum(r["lex_capped_hit"] for r in g) / n, 3),
            "or_capped_hit": round(sum(r["or_capped_hit"] for r in g) / n, 3),
            "ngram_hit": round(sum(r["ngram_hit"] for r in g) / n, 3),
            "node_cover_hit": round(sum(r["node_cover_hit"] for r in g) / n, 3),
            "median_lex_wide_pages": statistics.median(r["lex_wide_pages"] for r in g),
        }
    return out
