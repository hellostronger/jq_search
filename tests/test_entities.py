from __future__ import annotations

from pathlib import Path

import pytest

from treewiki.config import DEFAULTS
from treewiki.core.entities import (
    Entity,
    EntityIndex,
    Mention,
    add_mention,
    align_variants,
    normalize,
)
from treewiki.core.mentions import NgramMiner, rule_spans
from treewiki.core.store import KB
from treewiki.pipelines.entities import build_entities
from treewiki.pipelines.query import _lexical_pages, anchor_groups


def _cfg(**overrides) -> dict:
    return {"entities": {**DEFAULTS["entities"], **overrides}}


def _variants(rows: list[tuple[str, int, str]], source: str = "ngram") -> dict:
    """rows: (surface, page, context)"""
    variants: dict = {}
    for surface, page, context in rows:
        add_mention(variants, surface, "src_0001", page, context=context, source=source)
    return variants


def test_normalize_folds_case_punct_and_width():
    assert normalize(" iPhone　15-Pro ") == "iphone15pro"
    assert normalize("宁德时代，") == "宁德时代"
    assert normalize("  ") == ""


def test_short_alnum_codes_do_not_merge():
    """DDR4/DDR5 这类只差数字的短串是不同型号，并簇会让查询扩展引用错页。"""
    variants = _variants(
        [
            ("DDR4", 1, "内存 DDR4 出货提升"),
            ("DDR4", 2, "DDR4 报价回落"),
            ("DDR5", 3, "内存 DDR5 出货提升"),
            ("DDR5", 4, "DDR5 报价回落"),
        ]
    )
    index = align_variants(variants, ["src_0001"], _cfg())
    ddr4 = index.resolve("DDR4")
    ddr5 = index.resolve("DDR5")
    assert ddr4 and ddr5
    assert ddr4.id != ddr5.id


def test_rule_spans_extracts_quote_latin_and_org_name():
    text = "《宁德时代》与 iPhone 发布，宁德时代新能源科技股份有限公司披露业绩。"
    spans = {s.surface: s.source for s in rule_spans(text)}
    assert spans["宁德时代"] == "quote"
    assert spans["iPhone"] == "latin"
    assert spans["宁德时代新能源科技股份有限公司"] == "suffix"


def test_rule_spans_rejects_sentence_fragments():
    text = "与宁德时代签署供货协议的公司包括比亚迪股份有限公司。"
    surfaces = [s.surface for s in rule_spans(text)]
    assert surfaces == ["比亚迪股份有限公司"]


def test_ngram_miner_discovers_repeated_name_and_drops_stopwords():
    pages = [
        "宁德时代今天发布公告。",
        "报告显示，宁德时代在电池市场的份额提升。",
        "与宁德时代签署供货协议。",
        "宁德时代与比亚迪的竞争加剧。",
        "比亚迪披露了三季度业绩。",
        "比亚迪的毛利率同比回升。",
        "分析师认为比亚迪的估值合理。",
    ]
    miner = NgramMiner(_cfg(ngram_min_frequency=3)["entities"])
    for page in pages:
        miner.observe(page)
    selected = miner.finalize()
    assert "宁德时代" in selected
    assert "比亚迪" in selected
    assert "同比" not in selected
    assert "报告" not in selected


def test_extract_prefers_rule_spans_over_ngrams():
    text = "《宁德时代》宁德时代"
    miner = NgramMiner(_cfg(ngram_min_frequency=3)["entities"])
    for page in ("今日宁德时代公告", "另外宁德时代扩产", "市场称宁德时代签约"):
        miner.observe(page)
    miner.finalize()
    spans = miner.extract(text, rule_spans(text))
    assert [(s.surface, s.source) for s in spans] == [
        ("宁德时代", "quote"),
        ("宁德时代", "ngram"),
    ]


def test_align_merges_short_and_full_name():
    variants = _variants(
        [
            ("宁德时代", 1, "宁德时代在动力电池市场的份额提升"),
            ("宁德时代", 2, "宁德时代与整车厂签订供货协议"),
            ("宁德时代", 3, "宁德时代的产能继续扩张"),
            ("宁德时代", 4, "宁德时代发布新一代电池"),
            ("宁德时代新能源科技股份有限公司", 1, "宁德时代新能源科技股份有限公司披露业绩"),
            ("宁德时代新能源科技股份有限公司", 2, "宁德时代新能源科技股份有限公司扩产"),
            ("宁德时代新能源科技股份有限公司", 3, "宁德时代新能源科技股份有限公司公告"),
            ("宁德时代新能源科技股份有限公司", 4, "宁德时代新能源科技股份有限公司签约"),
        ],
        source="suffix",
    )
    index = align_variants(variants, ["src_0001"], _cfg()["entities"])
    entity = index.resolve("宁德时代")
    assert entity is not None
    names = {entity.canonical, *entity.aliases}
    assert "宁德时代新能源科技股份有限公司" in names
    assert len(entity.mentions) == 4


def test_align_keeps_model_variants_apart():
    variants = _variants(
        [
            ("iPhone 15", 1, "iPhone 15 起售价公布"),
            ("iPhone 15", 2, "iPhone 15 销量领先"),
            ("iPhone 15 Pro", 1, "iPhone 15 Pro 起售价公布"),
            ("iPhone 15 Pro", 2, "iPhone 15 Pro 销量领先"),
        ],
        source="latin",
    )
    index = align_variants(variants, ["src_0001"], _cfg()["entities"])
    base = index.resolve("iPhone 15")
    pro = index.resolve("iPhone 15 Pro")
    assert base is not None and pro is not None
    assert base.id != pro.id


def test_align_drops_variants_below_min_frequency():
    variants = _variants(
        [("宁德时代", 1, "宁德时代公告"), ("宁德时代", 2, "宁德时代扩产"), ("一次性词", 1, "一次性词")]
    )
    index = align_variants(variants, ["src_0001"], _cfg(min_frequency=2)["entities"])
    assert index.resolve("一次性词") is None
    assert index.stats["kept"] == 1


def test_entity_index_roundtrip(tmp_path):
    kb = KB(tmp_path / "kb")
    original = EntityIndex(
        built_at="2026-10-24T00:00:00+00:00",
        docs=["src_0001"],
        entities=[
            Entity(
                id="ent_0001",
                canonical="宁德时代",
                type="ORG",
                aliases=["宁德时代新能源科技股份有限公司"],
                mentions=[Mention(doc_id="src_0001", page=1, count=2)],
                frequency=6,
            )
        ],
        alias_index={"宁德时代": "ent_0001"},
        stats={"entities": 1},
    )
    kb.save_entities(original)
    loaded = kb.load_entities()
    assert loaded.entities[0].canonical == "宁德时代"
    assert loaded.resolve("宁德时代").id == "ent_0001"
    assert loaded.expand("宁德时代") == ["宁德时代", "宁德时代新能源科技股份有限公司"]


def _seed_kb(tmp_path) -> KB:
    kb = KB(tmp_path / "kb")
    pages = [
        "宁德时代今天发布公告，宁德时代新能源科技股份有限公司董事会决定扩产。",
        "报告显示，宁德时代在动力电池市场的份额继续提升。",
        "与宁德时代签署供货协议的公司包括比亚迪股份有限公司。",
        "比亚迪股份有限公司披露三季度业绩，比亚迪的毛利率回升。",
        "分析师认为宁德时代的估值仍具吸引力，比亚迪估值更高。",
        "比亚迪与宁德时代的竞争格局正在变化。",
    ]
    doc_id = kb.new_doc_id()
    kb.write_pages(doc_id, pages)
    kb.register(doc_id, Path("raw/a.md"), "sha", len(pages))
    return kb


def test_build_entities_end_to_end(tmp_path):
    kb = _seed_kb(tmp_path)
    cfg = _cfg(ngram_min_frequency=3, min_frequency=2)
    index = build_entities(kb, cfg, stage=lambda s: None)
    assert index.entities
    assert kb.entities_path.is_file()

    entity = index.resolve("宁德时代")
    assert entity is not None
    names = {entity.canonical, *entity.aliases}
    assert "宁德时代新能源科技股份有限公司" in names
    assert any(m.page == 2 for m in entity.mentions)

    byd = index.resolve("比亚迪")
    assert byd is not None and byd.id != entity.id


def test_build_entities_disabled(tmp_path):
    kb = _seed_kb(tmp_path)
    with pytest.raises(RuntimeError):
        build_entities(kb, _cfg(enabled=False), stage=lambda s: None)


def test_lexical_pages_or_within_group_and_across_groups():
    texts = [
        "无关内容",
        "宁德时代发布新品",
        "宁德时代新能源科技股份有限公司发布新品",
        "比亚迪发布新品",
        "宁德时代与比亚迪同时出现",
    ]
    groups = [["宁德时代", "宁德时代新能源科技股份有限公司"], ["比亚迪"]]
    assert _lexical_pages(texts, groups, 5) == [5]
    assert _lexical_pages(texts, ["宁德时代"], 5) == [2, 3, 5]


def test_anchor_groups_expands_aliases_and_keeps_unknown_terms():
    index = EntityIndex(
        entities=[
            Entity(
                id="ent_0001",
                canonical="宁德时代",
                aliases=["CATL", "宁德时代新能源科技股份有限公司"],
                frequency=10,
            )
        ],
        alias_index={
            "宁德时代": "ent_0001",
            "catl": "ent_0001",
            "宁德时代新能源科技股份有限公司": "ent_0001",
        },
    )
    groups = anchor_groups(["宁德时代", "毛利率"], index, limit=8)
    assert "catl" in groups[0]
    assert "宁德时代新能源科技股份有限公司" in groups[0]
    assert groups[1] == ["毛利率"]
    assert anchor_groups(["宁德时代"], None) == [["宁德时代"]]
