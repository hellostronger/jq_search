import json

import pytest

from treewiki.config import DEFAULTS
from treewiki.core.store import KB
from treewiki.core.tree import Node, TreeDoc, assign_ids, flash_skeleton
from treewiki.pipelines.query import (
    _parse_citations,
    _spread_sample,
    query_raw,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.calls = []

    def chat_json(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        item = self.responses[self.i]
        self.i += 1
        if isinstance(item, Exception):
            raise item
        return item

    def usage(self):
        return {"calls": len(self.calls), "prompt_tokens": 0, "completion_tokens": 0}


def _make_kb(tmp_path, pages, build_tree=None):
    kb = KB(tmp_path / "kb")
    doc_id = kb.new_doc_id()
    kb.write_pages(doc_id, pages)
    structure = build_tree() if build_tree else flash_skeleton(pages)
    assert structure
    assign_ids(structure)
    kb.save_tree(
        TreeDoc(
            doc_id=doc_id,
            doc_name="report.pdf",
            doc_description="一份财报",
            structure=structure,
        )
    )
    (kb.home / "index" / "sources.json").write_text(
        json.dumps(
            {
                doc_id: {
                    "path": "raw/x.pdf",
                    "sha256": "",
                    "pages": len(pages),
                    "status": "active",
                }
            }
        ),
        encoding="utf-8",
    )
    return kb, doc_id


@pytest.fixture()
def kb(tmp_path):
    pages = [
        "# 财务摘要\n\n收入 100 亿",
        "# 风险因素\n\n存在市场风险",
        "# 展望\n\n前景良好",
    ]
    kb, _ = _make_kb(tmp_path, pages)
    return kb


def _cfg(**overrides):
    query = dict(DEFAULTS["query"])
    query.update(overrides)
    return {"query": query, "llm": {**DEFAULTS["llm"]}}


def _answer(pages, text=None, sufficient=True):
    text = text if text is not None else "答案 " + " ".join(f"[p{p}]" for p in pages)
    return {"answer": text, "pages": list(pages), "sufficient": sufficient}


def _last_user(llm):
    return llm.calls[-1][-1]["content"]


def test_query_raw_flow(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险问题对应风险因素章节"},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    result = query_raw("src_0001", "有哪些风险？", kb, llm, _cfg())
    assert "市场风险" in result["answer"]
    assert result["answer_status"] == "ok"
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [2]}]
    assert result["citation_source"] == "model"
    assert result["read_pages"] == [2]
    assert result["fallback"] == "none"
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["read"] == ["0002"]


def test_query_raw_expand_leaf_reads_it(kb):
    llm = FakeLLM(
        [
            {"read": [], "expand": ["0003"], "reason": "进入展望章节（叶子）"},
            _answer([3], "前景良好 [p3]"),
        ]
    )
    result = query_raw("src_0001", "未来展望如何？", kb, llm, _cfg())
    assert result["read_pages"] == [3]
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["expand"] == ["0003"]


def test_query_raw_invalid_id_retries_then_spread_fallback(kb):
    llm = FakeLLM(
        [
            {"read": ["9999"], "expand": [], "reason": "幻觉编号"},
            {"read": ["9999"], "expand": [], "reason": "仍然幻觉"},
            _answer([1], "兜底答案 [p1]"),
        ]
    )
    result = query_raw("src_0001", "随便什么", kb, llm, _cfg())
    assert result["fallback"] == "spread_sample"
    assert result["read_pages"] == [1, 2, 3]
    assert result["rounds"][0]["attempts"] == 2
    assert "忽略无效编号：9999" in result["rounds"][0]["reason"]
    assert "只能从这些编号中选择" in llm.calls[1][-1]["content"]


def test_query_raw_retry_recovers_with_valid_id(kb):
    llm = FakeLLM(
        [
            {"read": ["8888"], "expand": [], "reason": "幻觉"},
            {"read": ["0002"], "expand": [], "reason": "重试命中"},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert result["fallback"] == "none"
    assert result["read_pages"] == [2]
    assert result["rounds"][0]["attempts"] == 2
    assert result["rounds"][0]["read"] == ["0002"]


def test_query_raw_empty_action_retries(kb):
    llm = FakeLLM(
        [
            {"reason": "没有给出动作", "expand": [], "read": []},
            {"read": ["0002"], "expand": [], "reason": "第二次给出动作"},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert result["rounds"][0]["attempts"] == 2
    assert result["rounds"][0]["read"] == ["0002"]
    assert result["read_pages"] == [2]


def test_query_raw_localizer_exception_retries(kb):
    llm = FakeLLM(
        [
            RuntimeError("网关抖动"),
            {"read": ["0002"], "expand": [], "reason": "重试命中"},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert result["rounds"][0]["attempts"] == 2
    assert result["read_pages"] == [2]
    assert "上一次调用失败" in llm.calls[1][-1]["content"]


@pytest.mark.parametrize(
    "payload, expect_expand, expect_read, expect_pages",
    [
        ({"expand": {"0003": "展望"}}, ["0003"], [], [3]),
        ({"read": 2}, [], ["0002"], [2]),
        ({"read": "0002, 0003"}, [], ["0002", "0003"], [2, 3]),
    ],
)
def test_query_raw_normalizes_localizer_shapes(
    kb, payload, expect_expand, expect_read, expect_pages
):
    llm = FakeLLM([{**payload, "reason": "x"}, _answer(expect_pages)])
    result = query_raw("src_0001", "问题", kb, llm, _cfg())
    assert result["rounds"][0]["expand"] == expect_expand
    assert result["rounds"][0]["read"] == expect_read
    assert result["read_pages"] == expect_pages


def test_query_raw_non_contiguous_pages_labelled_correctly(kb):
    llm = FakeLLM(
        [
            {"read": ["0001", "0003"], "expand": [], "reason": "跨章节"},
            _answer([1, 3], "收入 100 亿 [p1]，前景良好 [p3]"),
        ]
    )
    result = query_raw("src_0001", "财务与展望？", kb, llm, _cfg())
    assert result["read_pages"] == [1, 3]
    user = _last_user(llm)
    assert "【第 1 页】\n# 财务摘要\n\n收入 100 亿" in user
    assert "【第 3 页】\n# 展望\n\n前景良好" in user
    assert "【第 2 页】" not in user
    assert "存在市场风险" not in user
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [1, 3]}]


def test_query_raw_injects_root_summary(tmp_path):
    def build_tree():
        return [
            Node(
                title="总览",
                start_index=1,
                end_index=2,
                summary="全局摘要：收入增长且风险可控",
                nodes=[
                    Node(title="财务", start_index=1, end_index=1),
                    Node(title="风险", start_index=2, end_index=2),
                ],
            )
        ]

    kb, doc_id = _make_kb(tmp_path, ["收入 100 亿", "存在市场风险"], build_tree)
    llm = FakeLLM(
        [
            {"read": ["0003"], "expand": [], "reason": "风险分支"},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    query_raw(doc_id, "整体怎么样？", kb, llm, _cfg())
    user = _last_user(llm)
    assert "节点摘要（用于定位与全局概览，不能作为页码引用依据）" in user
    assert "全局摘要：收入增长且风险可控" in user


def test_query_raw_empty_answer_is_retried(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险"},
            {"answer": "", "pages": [], "sufficient": True},
            _answer([2], "存在市场风险 [p2]"),
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert len(llm.calls) == 3
    assert result["answer_status"] == "ok"
    assert "存在市场风险" in result["answer"]
    assert "上一次 answer 为空" in _last_user(llm)


def test_query_raw_empty_answer_status_after_exhausted(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险"},
            {"answer": "", "pages": [], "sufficient": True},
            {"answer": "  ", "pages": [2], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg(answer_attempts=2))
    assert len(llm.calls) == 3
    assert result["answer"] == ""
    assert result["answer_status"] == "empty"
    assert result["citations"] == [{"doc_id": "src_0001", "pages": []}]
    assert result["citation_source"] == "none"


def test_query_raw_citation_parsed_from_answer_text(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险"},
            {"answer": "存在市场风险 [p2]", "pages": [], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [2]}]
    assert result["citation_source"] == "text"


def test_query_raw_citations_never_fall_back_to_all_read_pages(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险"},
            {"answer": "存在市场风险", "pages": [3], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert result["read_pages"] == [2]
    assert result["citations"] == [{"doc_id": "src_0001", "pages": []}]
    assert result["citation_source"] == "none"


@pytest.mark.parametrize(
    "raw, answer, read_pages, expected, source",
    [
        ([2], "", [2, 3], [2], "model"),
        (["p2", "3"], "", [2, 3], [2, 3], "model"),
        ([], "见 [p2-3]", [1, 2, 3], [2, 3], "text"),
        ([], "见 [p2, p3]", [1, 2, 3], [2, 3], "text"),
        ([], "", [1, 2], [], "none"),
        ([9], "", [1, 2], [], "none"),
    ],
)
def test_parse_citations(raw, answer, read_pages, expected, source):
    pages, got_source = _parse_citations(raw, answer, read_pages)
    assert pages == expected
    assert got_source == source


def test_spread_sample_hits_head_and_tail():
    nodes = [
        Node(title=str(i), start_index=i, end_index=i, node_id=f"{i:04d}")
        for i in range(1, 12)
    ]
    out = _spread_sample(nodes, 8)
    assert len(out) == 8
    assert out[0].start_index == 1
    assert out[-1].start_index == 11


def test_query_raw_fallback_is_not_head_only(tmp_path):
    pages = [f"第 {i} 页正文" for i in range(1, 12)]
    kb, doc_id = _make_kb(
        tmp_path,
        pages,
        lambda: [Node(title=f"章{i}", start_index=i, end_index=i) for i in range(1, 12)],
    )
    llm = FakeLLM(
        [
            {"read": ["9999"], "expand": [], "reason": "幻觉"},
            {"read": ["9999"], "expand": [], "reason": "还是幻觉"},
            _answer([11]),
        ]
    )
    result = query_raw(doc_id, "全局问题", kb, llm, _cfg())
    assert result["fallback"] == "spread_sample"
    assert len(result["read_pages"]) == 8
    assert 1 in result["read_pages"]
    assert 11 in result["read_pages"]
    assert result["read_pages"] != list(range(1, 9))


def test_query_raw_internal_read_spreads_representative_leaves(tmp_path):
    pages = [f"第 {i} 页正文" for i in range(1, 13)]

    def build_tree():
        return [
            Node(
                title="全文",
                start_index=1,
                end_index=12,
                nodes=[Node(title=f"节{i}", start_index=i, end_index=i) for i in range(1, 13)],
            )
        ]

    kb, doc_id = _make_kb(tmp_path, pages, build_tree)
    llm = FakeLLM(
        [
            {"read": ["0001"], "expand": [], "reason": "读全篇"},
            _answer([12]),
        ]
    )
    result = query_raw(doc_id, "汇总", kb, llm, _cfg())
    assert len(result["read_pages"]) == 8
    assert 1 in result["read_pages"]
    assert 12 in result["read_pages"]


def test_query_raw_insufficient_retry_keeps_read_pages(tmp_path):
    pages = [f"第 {i} 页正文" for i in range(1, 21)]
    kb, doc_id = _make_kb(
        tmp_path,
        pages,
        lambda: [Node(title=f"章{i}", start_index=i, end_index=i) for i in range(1, 21)],
    )
    llm = FakeLLM(
        [
            {"read": ["0020"], "expand": [], "reason": "末页"},
            {"answer": "证据不足", "pages": [], "sufficient": False},
            _answer([20]),
        ]
    )
    result = query_raw(doc_id, "末页说了什么？", kb, llm, _cfg())
    assert len(llm.calls) == 3
    assert result["read_nodes"][0]["id"] == "0020"
    assert 20 in result["read_pages"]
    assert result["citations"] == [{"doc_id": doc_id, "pages": [20]}]
    assert "上一次判定证据不足" in _last_user(llm)
