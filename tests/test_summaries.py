from __future__ import annotations

from treewiki.core.tree import Node, add_summaries, assign_ids, describe_doc


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        item = self.responses[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def _two_leaves():
    structure = [
        Node(
            title="根",
            start_index=1,
            end_index=2,
            nodes=[
                Node(title="甲", start_index=1, end_index=1),
                Node(title="乙", start_index=2, end_index=2),
            ],
        )
    ]
    assign_ids(structure)
    return structure


def _retry_message(messages):
    return messages[-1]["content"]


def test_add_summaries_retries_empty_leaf_summary():
    structure = _two_leaves()
    llm = SeqLLM(
        [
            {"summary": ""},
            {"summary": "甲摘要"},
            {"summary": "乙摘要"},
            {"summary": "根摘要"},
        ]
    )
    report = add_summaries(
        structure, ["a", "b"], llm, "m", workers=1, retries=1
    )
    root = structure[0]
    assert root.nodes[0].summary == "甲摘要"
    assert root.nodes[1].summary == "乙摘要"
    assert root.summary == "根摘要"
    assert report == {"leaves": 2, "leaf_failures": [], "rollup_failures": []}
    assert len(llm.calls) == 4
    assert "上一次没有给出有效的 summary" in _retry_message(llm.calls[1])


def test_add_summaries_retries_leaf_exception():
    structure = _two_leaves()
    llm = SeqLLM(
        [
            RuntimeError("网关抖动"),
            {"summary": "甲摘要"},
            {"summary": "乙摘要"},
            {"summary": "根摘要"},
        ]
    )
    report = add_summaries(
        structure, ["a", "b"], llm, "m", workers=1, retries=1
    )
    assert structure[0].nodes[0].summary == "甲摘要"
    assert report["leaf_failures"] == []
    assert len(llm.calls) == 4


def test_add_summaries_reports_persistent_leaf_failure():
    structure = [Node(title="甲", start_index=1, end_index=1)]
    assign_ids(structure)
    llm = SeqLLM([{"summary": ""}, {"summary": "  "}])
    report = add_summaries(
        structure, ["a"], llm, "m", workers=1, retries=1
    )
    assert structure[0].summary == ""
    assert report["leaves"] == 1
    assert report["leaf_failures"] == [
        {
            "id": "0001",
            "title": "甲",
            "pages": [1, 1],
            "error": "摘要为空",
        }
    ]
    assert len(llm.calls) == 2


def test_add_summaries_rollup_fallback_from_children():
    structure = _two_leaves()
    llm = SeqLLM(
        [
            {"summary": "甲摘要"},
            {"summary": "乙摘要"},
            {"summary": ""},
            {"summary": ""},
        ]
    )
    report = add_summaries(
        structure, ["a", "b"], llm, "m", workers=1, retries=1
    )
    root = structure[0]
    assert root.summary == "甲：甲摘要；乙：乙摘要"
    assert report["rollup_failures"] == [
        {
            "id": "0001",
            "title": "根",
            "error": "摘要为空",
            "fallback": True,
        }
    ]
    assert len(llm.calls) == 4


def test_add_summaries_stage_reports_failures():
    structure = [Node(title="甲", start_index=1, end_index=1)]
    assign_ids(structure)
    llm = SeqLLM([{"summary": ""}, {"summary": ""}])
    stages = []
    add_summaries(
        structure,
        ["a"],
        llm,
        "m",
        workers=1,
        retries=1,
        stage=stages.append,
    )
    assert any("1 个叶节点摘要失败" in s for s in stages)


def test_describe_doc_retries_empty_then_succeeds():
    structure = [Node(title="甲", start_index=1, end_index=1)]
    assign_ids(structure)
    llm = SeqLLM([{"description": ""}, {"description": "文档描述"}])
    desc = describe_doc(["正文"], structure, llm, "m", retries=1)
    assert desc == "文档描述"
    assert len(llm.calls) == 2
    assert "上一次没有给出有效的 description" in _retry_message(llm.calls[1])


def test_describe_doc_retries_exception_then_succeeds():
    structure = [Node(title="甲", start_index=1, end_index=1)]
    llm = SeqLLM([RuntimeError("boom"), {"description": "文档描述"}])
    desc = describe_doc(["正文"], structure, llm, "m", retries=1)
    assert desc == "文档描述"
    assert len(llm.calls) == 2


def test_describe_doc_returns_empty_when_all_attempts_fail():
    structure = [Node(title="甲", start_index=1, end_index=1)]
    llm = SeqLLM([{"description": " "}, {"description": ""}])
    desc = describe_doc(["正文"], structure, llm, "m", retries=1)
    assert desc == ""
    assert len(llm.calls) == 2
