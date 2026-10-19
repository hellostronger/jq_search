import pytest

from treewiki.config import DEFAULTS
from treewiki.core.store import KB
from treewiki.core.tree import TreeDoc, assign_ids, flash_skeleton
from treewiki.pipelines.query import query_raw


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.calls = []

    def chat_json(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        item = self.responses[self.i]
        self.i += 1
        return item

    def usage(self):
        return {"calls": len(self.calls), "prompt_tokens": 0, "completion_tokens": 0}


@pytest.fixture()
def kb(tmp_path):
    kb = KB(tmp_path / "kb")
    pages = [
        "# 财务摘要\n\n收入 100 亿",
        "# 风险因素\n\n存在市场风险",
        "# 展望\n\n前景良好",
    ]
    doc_id = kb.new_doc_id()
    assert doc_id == "src_0001"
    kb.write_pages(doc_id, pages)
    structure = flash_skeleton(pages)
    assign_ids(structure)
    kb.save_tree(
        TreeDoc(
            doc_id=doc_id,
            doc_name="report.pdf",
            doc_description="一份财报",
            structure=structure,
        )
    )
    import json

    (kb.home / "index" / "sources.json").write_text(
        json.dumps(
            {doc_id: {"path": "raw/x.pdf", "sha256": "", "pages": 3, "status": "active"}}
        ),
        encoding="utf-8",
    )
    return kb


def _cfg():
    return {"query": DEFAULTS["query"], "llm": {**DEFAULTS["llm"]}}


def test_query_raw_flow(kb):
    llm = FakeLLM(
        [
            {"read": ["0002"], "expand": [], "reason": "风险问题对应风险因素章节"},
            {"answer": "存在市场风险 [p2]", "pages": [2], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "有哪些风险？", kb, llm, _cfg())
    assert "市场风险" in result["answer"]
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [2]}]
    assert result["read_pages"] == [2]
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["read"] == ["0002"]


def test_query_raw_expand_leaf_reads_it(kb):
    llm = FakeLLM(
        [
            {"read": [], "expand": ["0003"], "reason": "进入展望章节（叶子）"},
            {"answer": "前景良好 [p3]", "pages": [3], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "未来展望如何？", kb, llm, _cfg())
    assert result["read_pages"] == [3]
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["expand"] == ["0003"]


def test_query_raw_fallback_reads_frontier(kb):
    llm = FakeLLM(
        [
            {"read": ["9999"], "expand": [], "reason": "无效编号"},  # 找不到节点 → 兜底读当前层
            {"answer": "兜底答案", "pages": [1], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "随便什么", kb, llm, _cfg())
    assert result["read_pages"] == [1, 2, 3]


def test_query_raw_insufficient_retry(kb):
    llm = FakeLLM(
        [
            {"read": ["0001"], "expand": [], "reason": "财务"},
            {"answer": "原文未找到", "pages": [], "sufficient": False},
            {"answer": "扩大范围后的答案 [p2]", "pages": [2], "sufficient": True},
        ]
    )
    result = query_raw("src_0001", "风险？", kb, llm, _cfg())
    assert len(llm.calls) == 3
    assert result["read_pages"] == [1, 2, 3]
