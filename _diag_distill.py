"""诊断 distill 的 JSON 解析失败：是被 max_tokens 截断，还是模型输出结构不对。

只走 _request，一次调用一个结果，不触发 chat_json 的二次重问，也不触发退避重试风暴。
"""
from pathlib import Path

from treewiki.config import load_config
from treewiki.core.llm import LLM, extract_json
from treewiki.core.store import KB
from treewiki.pipelines.distill import EXTRACT_SYS, EXTRACT_USER, _batch_leaves, _leaves

kb = KB(Path("kb"))
cfg = load_config(Path("kb"))
llm = LLM(cfg)
tree = kb.load_tree("src_0001")
pages = kb.read_pages("src_0001", 1, kb.page_count("src_0001"))

batch = next(iter(_batch_leaves(_leaves(tree), pages, 3000, 9000)))
body = "\n\n".join(
    f"【片段 {i}】{it['title']}（第 {it['start']}-{it['end']} 页）\n{it['text']}"
    for i, it in enumerate(batch, 1)
)
lo = min(p for it in batch for p in it["pages"])
hi = max(p for it in batch for p in it["pages"])
print(f"批内叶节点 {len(batch)} 个，覆盖第 {lo}-{hi} 页，输入 {len(body)} 字符")

messages = [
    {"role": "system", "content": EXTRACT_SYS},
    {
        "role": "user",
        "content": EXTRACT_USER.format(
            doc_name=tree.doc_name, hint=tree.user_description or "（无）", body=body
        ),
    },
]

for mt in (4000, 12000):
    kwargs = dict(
        model=cfg["llm"]["index_model"],
        messages=messages,
        temperature=0.2,
        max_tokens=mt,
        response_format={"type": "json_object"},
    )
    try:
        resp = llm._request(kwargs)
    except Exception as e:
        print(f"max_tokens={mt}: 调用异常 {type(e).__name__}: {str(e)[:120]}")
        continue
    llm._record(resp)
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    print(
        f"max_tokens={mt}: finish={resp.choices[0].finish_reason} "
        f"补全={getattr(u, 'completion_tokens', None)} 输出={len(text)} 字符 "
        f"extract_json={'OK' if extract_json(text) is not None else 'None'}"
    )
    print(f"  头部: {text[:80]!r}")
    print(f"  尾部: {text[-120:]!r}")
    data = extract_json(text)
    if isinstance(data, dict):
        print(f"  键: {sorted(data)} 实体 {len(data.get('entities') or [])} 主题 {len(data.get('topics') or [])}")
