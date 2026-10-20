from __future__ import annotations

from treewiki.config import load_config
from treewiki.core.store import KB
from treewiki.pipelines import ingest


def test_ingest_markdown_bypasses_mineru(tmp_path, monkeypatch):
    kb = KB(tmp_path / "kb")
    src = tmp_path / "note.md"
    src.write_text(
        "# 第一章\n\n甲。\n\n# 第二章\n\n乙。\n\n# 第三章\n\n丙。",
        encoding="utf-8",
    )
    cfg = load_config(kb.home)
    cfg["index"]["summaries"] = False

    def boom(*args, **kwargs):
        raise AssertionError("md/txt 不应调用 MinerU")

    monkeypatch.setattr(ingest.mineru, "parse_cloud", boom)
    monkeypatch.setattr(ingest, "LLM", lambda cfg: object())
    seen = {}

    def fake_describe(pages, structure, llm, model, hint=""):
        seen["hint"] = hint
        return "测试文档"

    monkeypatch.setattr(ingest, "describe_doc", fake_describe)

    doc_id = ingest.ingest_file(
        src, kb, cfg, stage=lambda _: None, description="v1.0 平台部署文档"
    )

    assert doc_id == "src_0001"
    assert kb.page_count(doc_id) == 3
    assert (kb.raw_dir / "src_0001_note.md").is_file()
    assert (kb.parse_dir / doc_id / "document.md").is_file()
    tree = kb.load_tree(doc_id)
    assert tree.doc_description == "测试文档"
    assert tree.user_description == "v1.0 平台部署文档"
    assert seen["hint"] == "v1.0 平台部署文档"
    assert kb.sources()[doc_id]["description"] == "v1.0 平台部署文档"
    assert len(tree.structure) == 3


def test_ingest_warns_on_summary_and_description_failures(tmp_path, monkeypatch):
    kb = KB(tmp_path / "kb")
    src = tmp_path / "note.md"
    src.write_text(
        "# 第一章\n\n甲。\n\n# 第二章\n\n乙。\n\n# 第三章\n\n丙。",
        encoding="utf-8",
    )
    cfg = load_config(kb.home)
    cfg["index"]["summaries"] = True

    monkeypatch.setattr(ingest, "LLM", lambda cfg: object())
    monkeypatch.setattr(
        ingest,
        "add_summaries",
        lambda *args, **kwargs: {
            "leaves": 3,
            "leaf_failures": [{"id": "0001"}],
            "rollup_failures": [{"id": "0002"}],
        },
    )
    monkeypatch.setattr(ingest, "describe_doc", lambda *args, **kwargs: "")

    stages = []
    ingest.ingest_file(src, kb, cfg, stage=stages.append)

    assert any("⚠ 摘要失败 2 个节点" in s for s in stages)
    assert any("⚠ 文档描述为空" in s for s in stages)
