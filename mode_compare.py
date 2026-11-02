"""Compare eval result files across query modes: ok rate, latency, model calls, citations.

用法：python mode_compare.py eval_out/m_raw.jsonl eval_out/m_wiki.jsonl ...
"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def block(rows):
    n = len(rows)
    secs = [r.get("seconds") or 0 for r in rows]
    calls = [ (r.get("usage") or {}).get("calls", 0) for r in rows]
    ptoks = [(r.get("usage") or {}).get("prompt_tokens", 0) for r in rows]
    ctoks = [(r.get("usage") or {}).get("completion_tokens", 0) for r in rows]
    return {
        "n": n,
        "ok": sum(1 for r in rows if r.get("ok")),
        "median_s": round(statistics.median(secs), 1),
        "median_calls": statistics.median(calls) if calls else None,
        "median_prompt_tokens": statistics.median(ptoks) if ptoks else None,
        "median_cites": statistics.median([len(r.get("cited_pages") or []) for r in rows]),
        "refused": sum(1 for r in rows if r.get("refused")),
    }


files = sys.argv[1:]
data = {f: load(f) for f in files}
common = set.intersection(*[set(v) for v in data.values()]) if data else set()
print(f"共同题 {len(common)} 条\n")
print(f"{'file':<28} {'ok':>6} {'中位秒':>8} {'中位调用':>8} {'中位引用页':>9} {'拒答':>5}")
for f in files:
    b = block([data[f][i] for i in sorted(common)])
    print(f"{f:<28} {b['ok']:>3}/{b['n']:<2} {b['median_s']:>8} {str(b['median_calls']):>8} {str(b['median_cites']):>9} {b['refused']:>5}")

print("\n逐题（ok 标记）")
tiers = defaultdict(list)
for i in sorted(common):
    tier = data[files[0]][i]["tier"]
    tiers[tier].append(i)
for tier, ids in sorted(tiers.items()):
    for i in ids:
        marks = " ".join(
            f"{f.split('/')[-1].replace('m_','').replace('.jsonl','')}={'ok' if data[f][i].get('ok') else '--'}"
            f"({data[f][i].get('seconds')}s)"
            for f in files
        )
        q = (data[files[0]][i].get("question") or "")[:34]
        print(f"  [{tier}] {i:<18} {q:<36} {marks}")
