from __future__ import annotations

import json

import pytest

from treewiki.config import DEFAULTS
from treewiki.core.store import KB
from treewiki.core.tree import TreeDoc, assign_ids, flash_skeleton
from treewiki.core.wiki import (
    WikiPage,
    WikiStore,
    build_wiki_tree,
    merge_into,
    page_id_for,
    parse_page,
    dump_page,
)
from treewiki.pipelines.distill import distill_source
from treewiki.pipelines.query import query_cascade, query_parallel, query_wiki, run_query

PAGES = [
    "# 财务摘要\n\n收入 100 亿，公司为 Acme。",
    "# 风险因素\n\n存在市场风险。",
    "# 展望\n\n前景良好。",
]


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
    doc_id = kb.new_doc_id()
    kb.write_pages(doc_id, PAGES)
    structure = flash_skeleton(PAGES)
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
            {doc_id: {"path": "raw/x.pdf", "sha256": "", "pages": 3, "status": "active"}}
        ),
        encoding="utf-8",
    )
    return kb


def _cfg(**query):
    cfg = {
        "llm": {**DEFAULTS["llm"]},
        "wiki": {**DEFAULTS["wiki"]},
        "query": {**DEFAULTS["query"], **query},
    }
    return cfg


def _seed_wiki(kb, doc_id="src_0001") -> WikiStore:
    store = WikiStore(kb)
    store.save(
        WikiPage(
            id=doc_id,
            title="report.pdf",
            type="source",
            description="一份财报",
            sources=[doc_id],
            key_points=["收入 100 亿", "存在市场风险"],
        )
    )
    store.save(
        WikiPage(
            id="ent_acme",
            title="Acme",
            type="entity",
            description="一家公司",
            sources=[doc_id],
            citations=[{"doc": doc_id, "pages": [1], "claim": "Acme 收入 100 亿"}],
        )
    )
    return store


# ------------------------------------------------------------------ 页面模型

def test_page_roundtrip(tmp_path):
    page = WikiPage(
        id="ent_acme",
        title="Acme",
        type="entity",
        description="一家公司",
        tags=["org"],
        sources=["src_0001"],
        citations=[{"doc": "src_0001", "pages": [1], "claim": "收入 100 亿"}],
        relations=[{"predicate": "面临", "target": "ent_risk", "doc": "src_0001", "pages": [2]}],
    )
    back = parse_page(dump_page(page))
    assert back.id == "ent_acme"
    assert back.type == "entity"
    assert back.citations[0]["pages"] == [1]
    assert back.relations[0]["target"] == "ent_risk"
    assert "关键事实" in back.body and "关系" in back.body


def test_merge_dedupes_and_unions_pages():
    page = WikiPage(id="ent_acme", title="Acme", type="entity")
    merge_into(
        page,
        description="一家公司",
        tags=["org"],
        sources=["src_0001"],
        citations=[{"doc": "src_0001", "pages": [1], "claim": "收入 100 亿"}],
        relations=[{"predicate": "面临", "target": "ent_risk", "doc": "src_0001", "pages": [2]}],
    )
    merge_into(
        page,
        description="一家公司",
        tags=["org"],
        sources=["src_0001", "src_0002"],
        citations=[{"doc": "src_0001", "pages": [1], "claim": "收入 100 亿"}],
        relations=[{"predicate": "面临", "target": "ent_risk", "doc": "src_0001", "pages": [3]}],
    )
    assert page.tags == ["org"]
    assert page.sources == ["src_0001", "src_0002"]
    assert len(page.citations) == 1
    assert page.relations[0]["pages"] == [2, 3]


def test_wiki_store_index_log_and_tree(kb):
    store = _seed_wiki(kb)
    store.write_index()
    store.append_log("src_0001", "report.pdf", {"entities": 1, "topics": 0, "relations": 0})
    tree = build_wiki_tree(store, kb)

    assert (store.root / "index.md").read_text(encoding="utf-8").count("[[ent_acme]]") == 1
    log = (store.root / "log.md").read_text(encoding="utf-8")
    assert "src_0001" in log and "实体 1" in log
    assert tree["page_count"] == 2
    assert store.tree_path.is_file()
    assert tree["root"]["children"][0]["children"][0]["pages"] == ["src_0001"]


def test_page_id_for_is_stable():
    assert page_id_for("Acme Inc.", "entity").startswith("ent_acme")
    assert page_id_for("风险因素", "topic") == "top_风险因素"


# -------------------------------------------------------------------- 蒸馏

def test_distill_creates_pages_relations_and_stubs(kb):
    llm = FakeLLM(
        [
            {
                "summary": "财报显示公司收入 100 亿，存在市场风险。",
                "entities": [
                    {
                        "name": "Acme",
                        "type": "org",
                        "aliases": ["Acme Inc"],
                        "description": "一家公司",
                        "claims": [{"claim": "收入 100 亿", "pages": [1]}],
                    }
                ],
                "topics": [
                    {
                        "name": "市场风险",
                        "description": "外部市场波动风险",
                        "claims": [{"claim": "存在市场风险", "pages": [2]}],
                    }
                ],
                "relations": [
                    {"from": "Acme", "predicate": "面临", "to": "市场风险", "pages": [2]}
                ],
            }
        ]
    )
    stats = distill_source("src_0001", kb, llm, _cfg())
    store = WikiStore(kb)

    assert stats["entities"] == 1
    assert stats["topics"] == 1
    assert stats["relations"] == 1
    assert stats["batches"] == 1

    acme = store.load("ent_acme")
    assert acme.citations[0] == {"doc": "src_0001", "pages": [1], "claim": "收入 100 亿"}
    assert acme.relations[0]["target"] == "top_市场风险"
    assert acme.relations[0]["pages"] == [2]
    topic = store.load("top_市场风险")
    assert topic.citations[0]["claim"] == "存在市场风险"
    source = store.load("src_0001")
    assert source.type == "source" and "ent_acme" in source.links
    assert (store.root / "index.md").is_file() and store.tree_path.is_file()


def test_distill_reuses_cached_extracts_without_calling_llm(kb):
    payload = {
        "summary": "财报摘要。",
        "entities": [
            {"name": "Acme", "type": "org", "claims": [{"claim": "收入 100 亿", "pages": [1]}]}
        ],
        "topics": [],
        "relations": [],
    }
    assert distill_source("src_0001", kb, FakeLLM([payload]), _cfg())["batches"] == 1

    again = FakeLLM([])  # 任何新的抽取调用都会在这里抛 IndexError
    stats = distill_source("src_0001", kb, again, _cfg())
    assert stats["reused"] == 1
    assert again.calls == []
    assert WikiStore(kb).load("ent_acme").citations[0]["claim"] == "收入 100 亿"


def test_distill_is_idempotent(kb):
    payload = {
        "summary": "收入 100 亿。",
        "entities": [
            {"name": "Acme", "type": "org", "description": "一家公司",
             "claims": [{"claim": "收入 100 亿", "pages": [1]}]}
        ],
        "topics": [],
        "relations": [
            {"from": "Acme", "predicate": "面临", "to": "市场风险", "pages": [2]}
        ],
    }
    distill_source("src_0001", kb, FakeLLM([payload]), _cfg())
    stats = distill_source("src_0001", kb, FakeLLM([payload]), _cfg())
    store = WikiStore(kb)
    assert stats["pages"] == 3  # source + Acme + 市场风险桩
    acme = store.load("ent_acme")
    assert len(acme.citations) == 1
    assert len(acme.relations) == 1


def test_distill_all_batches_failed_raises_and_writes_nothing(kb):
    class Boom:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("429")

        def usage(self):
            return {"calls": 1}

    with pytest.raises(RuntimeError, match="429"):
        distill_source("src_0001", kb, Boom(), _cfg())
    store = WikiStore(kb)
    assert store.list() == []
    assert not (store.root / "index.md").exists()
    assert not store.tree_path.exists()


# -------------------------------------------------------------------- 查询

def test_query_wiki_answers_with_source_citations(kb):
    _seed_wiki(kb)
    llm = FakeLLM(
        [
            {"pages": ["ent_acme"], "sufficient": True, "reason": "实体页直接相关"},
            {
                "answer": "Acme 收入 100 亿 [src_0001:p1]",
                "citations": [{"doc": "src_0001", "pages": [1]}],
                "sufficient": True,
            },
        ]
    )
    result = query_wiki("Acme 收入多少？", kb, llm, _cfg())

    assert result["sufficient"] is True
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [1]}]
    assert result["wiki_pages"][0]["id"] == "ent_acme"
    assert result["rounds"][0]["select"] == ["ent_acme"]


def test_query_wiki_rejects_fabricated_pages(kb):
    _seed_wiki(kb)
    llm = FakeLLM(
        [
            {"pages": ["ent_acme"], "sufficient": True, "reason": "相关"},
            {
                "answer": "答案 [src_0001:p99]",
                "citations": [{"doc": "src_0001", "pages": [99]}],
                "sufficient": True,
            },
        ]
    )
    result = query_wiki("问题", kb, llm, _cfg())
    # 越界页码被丢弃，回退到 wiki 页自身的引用池
    assert result["citations"] == [{"doc_id": "src_0001", "pages": [1]}]


def test_cascade_falls_back_to_source_tree(kb):
    _seed_wiki(kb)
    llm = FakeLLM(
        [
            {"pages": [], "sufficient": False, "reason": "wiki 无相关页"},
            {"read": ["0002"], "expand": [], "reason": "风险章节"},
            {"answer": "存在市场风险 [p2]", "pages": [2], "sufficient": True},
        ]
    )
    result = query_cascade("有哪些风险？", kb, llm, _cfg())

    assert result["mode"] == "cascade"
    assert result["path"] == "raw"
    assert result["doc_id"] == "src_0001"
    assert "市场风险" in result["answer"]
    assert result["wiki_attempt"]["sufficient"] is False


def test_cascade_stays_on_wiki_when_sufficient(kb):
    _seed_wiki(kb)
    llm = FakeLLM(
        [
            {"pages": ["ent_acme"], "sufficient": True, "reason": "相关"},
            {"answer": "收入 100 亿 [src_0001:p1]",
             "citations": [{"doc": "src_0001", "pages": [1]}], "sufficient": True},
        ]
    )
    result = query_cascade("收入多少？", kb, llm, _cfg())
    assert result["path"] == "wiki"
    assert len(llm.calls) == 2


class RoutingLLM:
    """按 system prompt 路由响应，供并行模式（调用顺序不确定）使用。"""

    def __init__(self):
        self.calls = []

    def chat_json(self, messages, model=None, **kwargs):
        self.calls.append(messages)
        system = messages[0]["content"]
        if "知识库检索专家" in system:
            return {"pages": ["ent_acme"], "sufficient": True, "reason": "相关"}
        if "核对两条问答结果" in system:
            return {"consistent": True, "note": "一致"}
        if "严谨的知识库问答助手" in system:
            return {
                "answer": "收入 100 亿 [src_0001:p1]",
                "citations": [{"doc": "src_0001", "pages": [1]}],
                "sufficient": True,
            }
        if "定位答案位置" in system:
            return {"read": ["0002"], "expand": [], "reason": "风险章节"}
        if "严谨的问答助手" in system:
            return {"answer": "存在市场风险 [p2]", "pages": [2], "sufficient": True}
        raise AssertionError(f"未预期 prompt: {system[:40]}")

    def usage(self):
        return {"calls": len(self.calls)}


def test_parallel_compares_both_paths(kb):
    _seed_wiki(kb)
    llm = RoutingLLM()
    result = query_parallel("有哪些风险？", kb, llm, _cfg(), doc_id="src_0001")

    assert result["mode"] == "parallel"
    assert result["verification"]["consistent"] is True
    assert "双路验证一致" in result["answer"]
    assert "存在市场风险" in result["answer"]
    assert "收入 100 亿 [src_0001:p1]" in result["wiki_answer"]
    cited = {c["doc_id"]: set(c["pages"]) for c in result["citations"]}
    assert cited["src_0001"] == {1, 2}
    assert len(llm.calls) == 5  # wiki 选页+作答、raw 定位+作答、对比


def test_run_query_dispatch_and_validation(kb):
    _seed_wiki(kb)
    llm = FakeLLM(
        [
            {"pages": ["ent_acme"], "sufficient": True, "reason": "相关"},
            {"answer": "ok [src_0001:p1]",
             "citations": [{"doc": "src_0001", "pages": [1]}], "sufficient": True},
        ]
    )
    result = run_query("收入?", kb, llm, _cfg(mode="wiki"), mode="wiki")
    assert result["mode"] == "wiki"

    with pytest.raises(ValueError):
        run_query("收入?", kb, llm, _cfg(), mode="raw")
    with pytest.raises(ValueError):
        run_query("收入?", kb, llm, _cfg(), mode="nope")
