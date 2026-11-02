"""Diagnose entity-anchor regressions: why A/C lost gold pages that the old anchors found."""
import json
import sys
from collections import Counter

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
reg = [r for r in rows if r["regress"]]
print("regressions by tier:", dict(Counter(r["tier"] for r in reg)))

for tier in ("C", "A"):
    print(f"\n########## {tier} 回退样本")
    for r in [x for x in reg if x["tier"] == tier][:3]:
        print(f"  id={r['id']} gold={r['gold_pages']}")
        print(f"    old_terms   ={r['old_terms']}  -> old_pages={r['old_pages'][:10]}")
        print(f"    entity_terms={r['entity_terms']}")
        print(f"    groups      ={[len(g) for g in r['groups']]} -> new_pages={r['new_pages'][:10]}")

print("\n=== 别名组大小分布 ===")
mx = [max((len(g) for g in r["groups"]), default=0) for r in rows]
print("max group size:", dict(Counter(mx).most_common(8)))
print("items whose gold page count > 1:", sum(1 for r in rows if len(r["gold_pages"]) > 1))
