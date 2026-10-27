import random
from pathlib import Path

from treewiki.core.entities import Entity, EntityIndex, normalize
from treewiki.core.store import KB
from treewiki.eval.dataset import (
    Article,
    _cloze,
    derive_multihop,
    derive_multiturn,
    _figures,
    _left_context,
    _longest_run,
    _COUNT,
    _MONEY,
    _PCT,
    _paraphrase,
    _subjectless,
    _underspecified,
    classify_by_lexicon,
    build_factual_items,
    build_timeline_items,
    expand_duplicate_gold,
    refine_questions,
    quota,
    split_articles,
)
from treewiki.eval.runner import (
    _ngram_terms,
    _pages_with,
    cited_pages,
    entity_probe,
    grade,
    grade_multiturn,
)
from treewiki.eval.replay import build_replay_kb, reconstruct
from treewiki.pipelines.query import resolve_followup

P1 = (
    "**【文章 A0001】苹果上调 iPhone 17 备货**\n\n"
    "来源：财闻 ｜ 发布时间：2026-10-15 10:21（北京时间）\n\n"
    "苹果将 iPhone 17 的备货量上调至 1200 万台，较此前预期增长 40%。\n\n"
    "机构预估其在中国市场的起售价为 5999 元，共 256GB 版本。\n"
)
P2 = "**【文章 A0002】宁德时代发布新电池**\n\n钠电池能量密度达到 175Wh/kg。\n"


def _pages():
    return [(1, P1), (2, P2)]


def test_split_articles_keeps_page_attribution_across_pages():
    arts = split_articles(_pages())
    assert [a.aid for a in arts] == ["A0001", "A0002"]
    assert arts[0].date == "2026-10-15"
    assert arts[0].source == "财闻"
    assert [p for p, _ in arts[0].frags] == [1]
    assert [p for p, _ in arts[1].frags] == [2]


def test_cloze_question_keeps_original_clause_wording():
    art = Article(aid="A0001", title="苹果上调 iPhone 17 备货")
    sent = "机构预估其在中国市场的起售价为 5999 元。"
    m = _MONEY.search(sent)
    assert m
    assert _cloze(sent, m, "money", art) == "机构预估其在中国市场的起售价为多少？"


def test_cloze_rejects_clause_without_cue_word():
    art = Article(aid="A0001", title="随便一篇文章")
    m = _MONEY.search("会场面积 3000 元 的布置很罕见")
    assert m
    assert _cloze("3000 元 的布置很罕见。", m, "money", art) is None


def test_cloze_rejects_deictic_clause_that_cannot_stand_alone():
    art = Article(aid="A0001", title="某公司被罚")
    sent = "上述处罚将导致公司 2026 年度净利润减少 3.5 亿元。"
    m = _MONEY.search(sent)
    assert m
    assert _cloze(sent, m, "money", art) is None


def test_paraphrase_keeps_latin_entity_when_stripping_would_lose_the_subject():
    q = "iPhone 17 的同比增长是多少？"
    out = _paraphrase(q)
    assert out and "iPhone" in out and "去年同期" in out


def test_subjectless_question_is_rejected_even_if_clause_reads_smoothly():
    for q in (
        "9月环比走高多少？",
        "最新市值升至多少？",
        "第三季度营业收入为多少？",
        "2027年3月升至多少？",
        "各治疗组中超过多少？",
        "较去年同期增长多少？",
    ):
        assert _subjectless(q), q
    for q in (
        "赛力斯汽车9月批发销量为多少？",
        "富祥股份市值多少？",
        "台积电预计2026年营收增长多少？",
        "三星电子第三季度营收同比增长多少？",
    ):
        assert not _subjectless(q), q


def test_cloze_drops_sentence_whose_subject_only_lives_in_the_article():
    art = Article(aid="A0001", title="月度数据")
    sent = "9月环比走高 1.2 个百分点。"
    m = _PCT.search(sent)
    assert m
    assert _cloze(sent, m, "pct", art) is None


def test_tier_uses_production_anchor_behaviour_not_template_guess():
    items = [
        {"id": "a", "question": "iPhone 17 的备货量是多少？", "gold_pages": [1], "subject": "iphone 17"},
        {"id": "b", "question": "iPhone 17 与 Mate 90 的备货量是多少？", "gold_pages": [1], "subject": "iphone 17"},
        {"id": "c", "question": "备货量上调多少？", "gold_pages": [1], "subject": "iphone 17"},
    ]
    lows = [P1.lower(), P2.lower()]
    classify_by_lexicon(items, lows)
    assert items[0]["tier"] == "A"
    assert items[1]["tier"] == "B"
    assert items[1]["term_hits"] == "1/2"
    assert items[2]["tier"] == "S"


def test_ambiguous_guard_drops_question_whose_key_phrase_spreads_across_pages():
    items = [{"id": "a", "question": "市场起售价为多少？", "gold_pages": [1], "subject": "x"}]
    lows = [f"市场起售价为 {i} 元" for i in range(6)]
    classify_by_lexicon(items, lows)
    assert items[0]["key_pages"] == 3
    assert items[0]["ambiguous"] is True


def test_longest_run_strips_the_question_word():
    assert _longest_run("机构预估其在中国市场的起售价为多少？") == "机构预估其在中国市场的起售价为"


def test_quota_tops_up_when_a_tier_pool_runs_short():
    items = [{"id": str(i), "tier": t} for i, t in enumerate(["A", "B", "B", "B", "C"])]
    out = quota(items, {"A": 3, "B": 1, "C": 3}, random.Random(1), total=5)
    assert len(out) == 5
    assert sum(1 for i in out if i["tier"] == "A") == 1


def test_grade_requires_figure_and_citation():
    item = {"id": "1", "tier": "A", "question": "q", "gold_pages": [12], "gold_figures": ["5999"]}
    good = {"answer": "起售价 5,999 元", "citations": [{"doc_id": "d", "pages": [12]}]}
    bad_page = {"answer": "起售价 5999 元", "citations": [{"doc_id": "d", "pages": [99]}]}
    bad_fig = {"answer": "起售价 6999 元", "citations": [{"doc_id": "d", "pages": [12]}]}
    assert grade(item, good)["ok"] is True
    assert grade(item, bad_page)["ok"] is False
    assert grade(item, bad_fig)["ok"] is False


def test_grade_honest_hedge_with_adjacent_citation_is_not_called_hallucination():
    item = {
        "id": "1",
        "tier": "A",
        "question": "5年期以上LPR为多少？",
        "gold_pages": [12],
        "gold_span": [12, 12],
        "gold_figures": ["3.5%"],
    }
    hedge = {
        "answer": "文档未明确给出 5 年期以上 LPR，只提到 1 年期 LPR 为 3.00%",
        "citations": [{"doc_id": "d", "pages": [3]}],
    }
    lows = ["x", "y", "目前 1 年期 LPR 为 3.00%"]
    row = grade(item, hedge, lows)
    assert row["refused"] is True
    assert row["unsupported_citation"] is False
    assert row["ok"] is False


def test_grade_refusal_quoting_the_gold_figure_is_not_success():
    item = {
        "id": "1",
        "tier": "B",
        "question": "台积电预计2026年营收增长多少？",
        "gold_pages": [12],
        "gold_span": [12, 12],
        "gold_figures": ["30%"],
    }
    refusal = {
        "answer": "文档未包含台积电官方的全年营收增速，只提到另一项业务增长 30%",
        "citations": [{"doc_id": "d", "pages": [12]}],
    }
    row = grade(item, refusal)
    assert row["refused"] is True
    assert row["figure_hit"] == 1
    assert row["ok"] is False


def test_sanitize_items_keeps_grading_fields_and_drops_prose():
    from treewiki.eval.dataset import sanitize_items

    items = [
        {
            "id": "1",
            "tier": "A",
            "question": "公募FOF成立门槛降至多少？",
            "gold_pages": [12],
            "gold_span": [12, 13],
            "gold_figures": ["5000"],
            "gold_answer": "公募FOF成立门槛降至 5000 万元，规模压力显著缓解",
            "article_title": "公募FOF新规落地：一批产品宣布延长募集",
            "subject": "批产品宣布延长",
        }
    ]
    out = sanitize_items(items)[0]
    assert "gold_answer" not in out and "article_title" not in out and "subject" not in out
    assert out["gold_pages"] == [12] and out["gold_figures"] == ["5000"]
    assert grade(out, {"answer": "门槛降至 5,000 万元", "citations": [{"pages": [12]}]})["ok"]


def test_grade_counts_hallucinated_citation_as_refusal_failure():
    item = {"id": "1", "tier": "D", "question": "不存在的东西？", "gold_pages": [], "gold_figures": []}
    refuse = {"answer": "文档中未提及该产品", "citations": []}
    halluc = {"answer": "该设备售价约 5999 元", "citations": [{"doc_id": "d", "pages": [7]}]}
    assert grade(item, refuse)["ok"] is True
    assert grade(item, halluc)["ok"] is False
    assert grade(item, halluc)["refused"] is False


def test_grade_accepts_alternative_page_whose_text_actually_contains_the_figure():
    item = {
        "id": "1",
        "tier": "A",
        "question": "q",
        "gold_pages": [12],
        "gold_span": [12, 12],
        "gold_figures": ["5999"],
    }
    alt = {"answer": "起售价 5999 元", "citations": [{"doc_id": "d", "pages": [3]}]}
    lows = ["x" * 29, "别的页", "苹果宣布起售价为 5,999 元"]
    row = grade(item, alt, lows)
    assert row["page_hit"] is False
    assert row["citation_verifiable"] is True
    assert row["ok"] is True


def test_grade_flags_citation_that_does_not_support_the_number():
    item = {
        "id": "1",
        "tier": "A",
        "question": "q",
        "gold_pages": [12],
        "gold_span": [12, 12],
        "gold_figures": ["5999"],
    }
    fake = {"answer": "起售价 5999 元", "citations": [{"doc_id": "d", "pages": [3]}]}
    lows = ["x", "y", "苹果宣布起售价为 4999 元"]
    row = grade(item, fake, lows)
    assert row["unsupported_citation"] is True
    assert row["ok"] is False


def test_cited_pages_merges_multiple_sources():
    res = {"citations": [{"pages": [3, 5]}, {"pages": [5, 9]}]}
    assert cited_pages(res) == [3, 5, 9]


def test_ngram_segmentation_recovers_a_gold_page_that_and_over_clause_terms_misses():
    lows = ["机构预估其在中国市场的起售价为 5999 元，共 256GB 版本。", "别的页面"]
    q = "机构预估其在中国市场的起售价为多少？"
    from treewiki.pipelines.query import anchor_terms

    assert _pages_with(lows, anchor_terms(q), "and", 12) != [1]
    assert _pages_with(lows, _ngram_terms(q), "and", 12) == [1]


def test_figures_keeps_single_digit_percentages():
    sent = "油价盘中涨幅一度超过5%，随后回落。"
    assert _figures(sent, _PCT.search(sent)) == ["5"]


def test_figures_keeps_single_digit_with_count_units():
    sent = "今年年底地面算力部署规模将超过2GW。"
    assert _figures(sent, _COUNT.search(sent)) == ["2"]


def test_left_context_uses_unit_match_to_skip_date_digits():
    answer = "2026年2月，公司营收增长2%。"
    assert _left_context(answer, "2%") == "2026年2月，公司营收增长"


def _dup_item(**kw):
    base = {
        "id": "x",
        "tier": "A",
        "kind": "literal",
        "question": "德国DAX指数均涨多少？",
        "gold_pages": [2],
        "gold_span": [2, 2],
        "gold_answer": "欧洲主要股指集体上涨，德国DAX指数均涨1.4%，法国CAC40指数涨1.1%。",
        "gold_figures": ["1.4"],
    }
    base.update(kw)
    return base


def test_duplicate_fact_on_another_page_joins_the_gold_set():
    lows = [
        "无关页面",
        "欧洲主要股指集体上涨，德国DAX指数均涨1.4%，法国CAC40指数涨1.1%。",
        "美股收涨，欧洲主要股指集体上涨，德国DAX指数均涨1.4%。",
    ]
    items = [_dup_item()]
    expanded, conflicted = expand_duplicate_gold(items, [t.lower() for t in lows])
    assert (expanded, conflicted) == (1, 0)
    assert items[0]["gold_pages"] == [2, 3]
    assert items[0]["gold_spans"] == [[2, 2], [3, 3]]


def test_same_wording_with_a_different_number_is_dropped_as_ambiguous():
    lows = [
        "无关页面",
        "欧洲主要股指集体上涨，德国DAX指数均涨1.4%。",
        "欧洲主要股指集体上涨，德国DAX指数均涨2.2%。",
    ]
    items = [_dup_item()]
    expanded, conflicted = expand_duplicate_gold(items, [t.lower() for t in lows])
    assert (expanded, conflicted) == (0, 1)
    assert items[0]["ambiguous"] is True


def test_truncated_number_is_not_mistaken_for_the_same_value():
    lows = [
        "无关页面",
        "贵金属上涨，现货黄金盘中突破4220美元/盎司。",
        "贵金属上涨，现货黄金盘中突破42\n\n00美元/盎司。",
    ]
    item = {
        "id": "1", "kind": "literal", "tier": "S", "question": "现货黄金盘中突破多少？",
        "gold_pages": [2], "gold_span": [2, 2], "gold_figures": ["4220"],
        "gold_answer": "贵金属上涨，现货黄金盘中突破4220美元/盎司。",
    }
    expanded, conflicted = expand_duplicate_gold([item], [t.lower() for t in lows])
    assert (expanded, conflicted) == (0, 1)
    assert item["ambiguous"] is True


def test_grade_credits_any_of_multiple_gold_pages():
    item = {
        "id": "1",
        "tier": "A",
        "question": "q",
        "gold_pages": [2, 9],
        "gold_span": [2, 2],
        "gold_spans": [[2, 2], [9, 9]],
        "gold_figures": ["1.4"],
    }
    alt = {"answer": "均涨 1.4%", "citations": [{"doc_id": "d", "pages": [9]}]}
    assert grade(item, alt)["ok"] is True


def test_refine_drops_literal_clause_that_is_not_in_its_answer():
    items = [
        {
            "id": "1",
            "kind": "literal",
            "tier": "S",
            "question": "中国高校培养了美国多少？",
            "gold_pages": [1],
            "gold_answer": "2022 年时，美国为 46%，中国为 27%。",
        },
        {
            "id": "2",
            "kind": "literal",
            "tier": "A",
            "question": "美国为多少？",
            "gold_pages": [1],
            "gold_answer": "2022 年时，美国为 46%，中国为 27%。",
        },
    ]
    out, stats = refine_questions(items)
    assert stats["bad_clause"] == 1
    assert [i["id"] for i in out] == ["2"]


def test_refine_drops_paraphrase_that_lost_its_subject():
    same_q = "机构预估前三季度净利表现为多少？"
    items = [
        {"id": "1", "kind": "para", "tier": "S", "question": same_q, "gold_pages": [10], "gold_answer": ""},
        {"id": "2", "kind": "para", "tier": "S", "question": same_q, "gold_pages": [20], "gold_answer": ""},
    ]
    out, stats = refine_questions(items)
    assert stats["multi_conflict"] == 2 and out == []


def test_refine_dedupes_identical_question_with_identical_gold():
    q = "Thrust版本的预订意向金则达到多少？"
    items = [
        {"id": "1", "kind": "literal", "tier": "A", "question": q, "gold_pages": [10, 20], "gold_answer": q.replace("多少", "3 万元")},
        {"id": "2", "kind": "literal", "tier": "A", "question": q, "gold_pages": [10, 20], "gold_answer": q.replace("多少", "3 万元")},
    ]
    out, stats = refine_questions(items)
    assert stats["deduped"] == 1 and len(out) == 1


def test_refine_merges_same_value_reports_from_different_pages():
    q = "日经225指数收盘上涨多少？"
    items = [
        {
            "id": "1", "kind": "literal", "tier": "A", "question": q,
            "gold_pages": [10], "gold_span": [10, 11], "gold_figures": ["2.40"],
            "gold_answer": "日经225指数收盘上涨2.40%，报69946.86点。",
        },
        {
            "id": "2", "kind": "literal", "tier": "A", "question": q,
            "gold_pages": [20], "gold_span": [20, 20], "gold_figures": ["2.40"],
            "gold_answer": "日经225指数收盘上涨2.40%，报69946.86点。",
        },
    ]
    out, stats = refine_questions(items)
    assert stats["merged"] == 1 and len(out) == 1
    assert out[0]["gold_pages"] == [10, 20]
    assert out[0]["gold_spans"] == [[10, 11], [20, 20]]


def test_refine_drops_question_that_lost_its_subject():
    items = [
        {
            "id": "1", "kind": "literal", "tier": "S", "question": "总市值突破多少？",
            "gold_pages": [1], "gold_answer": "涨停板上封单超31万手，总市值突破200亿元。",
            "gold_figures": ["200"],
        },
        {
            "id": "2", "kind": "literal", "tier": "A", "question": "万达商管2025年营收多少？",
            "gold_pages": [2], "gold_answer": "万达商管2025年营收284.9亿元。",
            "gold_figures": ["284.9"],
        },
    ]
    out, stats = refine_questions(items)
    assert stats["underspecified"] == 1
    assert [i["id"] for i in out] == ["2"]


def test_timeline_gold_articles_align_with_gold_spans():
    arts = [
        Article(aid="A0001", title="智元机器人：发布新品", date="2026-10-01", frags=[(40, "x")]),
        Article(aid="A0002", title="智元机器人，再获融资", date="2026-10-02", frags=[(30, "x")]),
        Article(aid="A0003", title="智元机器人！出货量大增", date="2026-10-03", frags=[(50, "x")]),
    ]
    items = build_timeline_items(arts)
    it = next(i for i in items if i["entity"] == "智元机器人")
    assert it["gold_articles"] == ["A0001", "A0002", "A0003"]
    assert it["gold_spans"] == [[40, 40], [30, 30], [50, 50]]


def test_underspecified_catches_relative_and_deictic_heads():
    bad = [
        "较冲突爆发前的1710万桶下降多少？",
        "相较该合约今年5月中旬超过多少？",
        "则提出未来3至5年数据中心AI业务营收增长多少？",
        "刚结束的2026财年营收同比增长多少？",
        "配售价较前一交易日收市价折让多少？",
        "业绩变动主要系2026年1—9月公司营收同比增长多少？",
        "2028财年营收同比增长多少？",
        "2028年度及2029年度实际实现的净利润分别不低于多少？",
        "占全市住房总成交比重突破多少？",
        "相关内容还提到一期规划年产AI服务器多少？",
        "客户转换成本极高：动态PE大约多少？",
        "就有一块是它造的——份额多少？",
        "美联储10月加息的概率从公布前的多少？",
        "今年1-8月营业收入多少？",
        "过去10年股东总回报突破多少？",
        "北美收入下滑超过多少？",
        "交易价值约多少？",
        "他管理这三只基金的任职回报均突破多少？",
    ]
    for q in bad:
        assert _underspecified(q), q
    good = [
        "壁仞科技配售价较前一交易日收市价折让多少？",
        "贵州茅台2026年三季度净利润多少？",
        "比亚迪9月销量多少？",
    ]
    for q in good:
        assert not _underspecified(q), q


def test_refine_drops_para_question_that_became_deictic():
    items = [
        {
            "id": "1", "kind": "para", "tier": "B",
            "question": "则提出未来3至5年数据中心AI业务营收增长多少？",
            "gold_pages": [1],
            "gold_answer": "AMD则提出未来3至5年数据中心AI业务营收增长80%的目标。",
            "gold_figures": ["80"],
        },
    ]
    out, stats = refine_questions(items)
    assert out == [] and stats["underspecified"] == 1


def test_timeline_skips_unit_event_and_metric_fragments():
    groups = ["同比下降", "万亿韩元", "被提起公诉", "分红率", "贵州茅台", "机器人"]
    arts = []
    n = 1
    for term in groups:
        for date in ("2026-10-01", "2026-10-02", "2026-10-03"):
            arts.append(Article(aid=f"A{n:04d}", title=term, date=date, frags=[(n, "x")]))
            n += 1
    items = build_timeline_items(arts)
    ents = {i["entity"] for i in items}
    assert {"贵州茅台", "机器人"} <= ents
    assert not ents & {"同比下降", "万亿韩元", "被提起公诉", "分红率"}


def test_factual_items_take_three_sentences_per_article():
    art = Article(
        aid="A0001",
        title="测试公司：业绩发布",
        date="2026-10-15",
        frags=[
            (1, "测试公司2026年一季度营收达到100亿元。"),
            (2, "测试公司2026年二季度营收达到200亿元。"),
            (3, "测试公司2026年三季度营收达到300亿元。"),
        ],
    )
    items = build_factual_items([art], random.Random(7))
    literal = [i for i in items if i["kind"] == "literal"]
    assert len(literal) == 3


def test_replay_reconstructs_literal_and_borrows_literal_for_para():
    items = [
        {
            "id": "1", "tier": "A", "kind": "literal", "article": "A0001",
            "question": "测试公司营收为多少？", "gold_match": "100亿元",
            "gold_figures": ["100"], "gold_pages": [2], "gold_span": [2, 2],
        },
        {
            "id": "2", "tier": "S", "kind": "para", "article": "A0001",
            "question": "测试公司收入体量为多少？", "gold_match": "100亿元",
            "gold_figures": ["100"], "gold_pages": [2], "gold_span": [2, 2],
        },
        {
            "id": "3", "tier": "C", "kind": "timeline",
            "question": "测试公司 最近的报道里有哪些数字？",
            "gold_pages": [5], "gold_span": [5, 6],
        },
    ]
    corpus = reconstruct(items)
    assert corpus.total == 6
    assert corpus.pages[2] == ["测试公司营收为100亿元。"]
    assert corpus.pages.get(5, []) == []
    assert corpus.stats["by_tier"] == {"A": 1, "S": 1}


def test_build_replay_kb_preserves_original_page_numbers(tmp_path):
    rows = [
        {
            "id": "1", "tier": "A", "kind": "literal", "article": "A0001",
            "question": "测试公司营收为多少？", "gold_match": "100亿元",
            "gold_figures": ["100"], "gold_pages": [2], "gold_span": [2, 3],
        }
    ]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    kb, corpus = build_replay_kb(dataset, tmp_path / "kb")
    assert kb.page_count("src_0001") == 3
    assert kb.read_pages("src_0001", 2, 2) == ["测试公司营收为100亿元。"]
    assert kb.read_pages("src_0001", 1, 1) == [""]


def _probe_index() -> EntityIndex:
    full = "宁德时代新能源科技股份有限公司"
    short = "宁德时代"
    return EntityIndex(
        docs=["src_0001"],
        entities=[
            Entity(
                id="ent_0001",
                canonical=full,
                type="ORG",
                aliases=[short],
                frequency=6,
                sources=["suffix"],
            )
        ],
        alias_index={normalize(full): "ent_0001", normalize(short): "ent_0001"},
    )


def test_entity_probe_separates_alias_gain_from_short_anchor_gain(tmp_path):
    kb = KB(tmp_path / "kb")
    kb.write_pages(
        "src_0001",
        ["无关内容。", "宁德时代营收为100亿元。", "宁德时代新能源科技股份有限公司披露业绩。"],
    )
    kb.register("src_0001", Path("dataset.jsonl"), "-", 3)
    kb.save_entities(_probe_index())
    items = [
        {
            "id": "short-page", "tier": "S", "kind": "literal",
            "question": "宁德时代新能源科技股份有限公司营收为多少？",
            "gold_pages": [2], "gold_figures": ["100"],
        },
        {
            "id": "full-page", "tier": "S", "kind": "literal",
            "question": "宁德时代新能源科技股份有限公司营收为多少？",
            "gold_pages": [3], "gold_figures": ["100"],
        },
    ]
    rows = {r["id"]: r for r in entity_probe(kb, "src_0001", items, 12)}
    assert rows["short-page"]["old_hit"] is False
    assert rows["short-page"]["alias_gain"] is True
    assert rows["short-page"]["anchor_gain"] is False
    assert rows["full-page"]["old_hit"] is False
    assert rows["full-page"]["anchor_gain"] is True


def _hop_sources():
    return [
        {
            "id": "h1", "tier": "A", "kind": "literal", "article": "A1",
            "question": "赛力斯汽车9月批发销量为多少？",
            "gold_pages": [1], "gold_span": [1, 1], "gold_figures": ["1.2"],
        },
        {
            "id": "h2", "tier": "S", "kind": "literal", "article": "A1",
            "question": "赛力斯汽车9月零售销量是多少？",
            "gold_pages": [2], "gold_span": [2, 2], "gold_figures": ["2.4"],
        },
    ]


def test_derive_multihop_combines_two_facts_and_keeps_hop_metadata():
    items = derive_multihop(_hop_sources())
    assert len(items) == 1
    it = items[0]
    assert it["tier"] == "H" and it["kind"] == "multihop"
    assert "批发销量" in it["question"] and "零售销量" in it["question"]
    assert it["question"].endswith("分别是多少？")
    assert it["gold_pages"] == [1, 2]
    assert [h["figures"] for h in it["hops"]] == [["1.2"], ["2.4"]]


def test_derive_multihop_preserves_motion_predicate():
    src = [
        {**_hop_sources()[0], "question": "重要堤防达标率升至多少？", "gold_pages": [1]},
        {**_hop_sources()[1], "question": "国家水网覆盖率升至多少？", "gold_pages": [2]},
    ]
    it = derive_multihop(src)[0]
    assert it["question"].endswith("分别升至多少？")
    assert "分别是多少" not in it["question"]


def test_derive_multihop_rejects_change_question_pairs():
    # “股价下跌多少” + “股价上涨多少” 不能合并成 “股价分别是多少”，那是两个不同的量
    src = [
        {**_hop_sources()[0], "question": "德州仪器股价下跌多少？", "gold_pages": [1]},
        {**_hop_sources()[1], "question": "美光科技股价上涨多少？", "gold_pages": [2]},
    ]
    assert derive_multihop(src) == []


def test_derive_multihop_rejects_fragment_left_and_same_figure():
    bad = [
        {**_hop_sources()[0], "id": "f1", "question": "ROIC连续两年回多少？"},
        {**_hop_sources()[1], "id": "f2"},
    ]
    assert derive_multihop(bad) == []
    same = [
        {**_hop_sources()[0], "id": "s1"},
        {**_hop_sources()[1], "id": "s2", "gold_figures": ["1.20"]},
    ]
    assert derive_multihop(same) == []


def test_derive_multihop_marks_other_pages_restating_both_facts():
    src = [
        {**_hop_sources()[0], "gold_pages": [1]},
        {**_hop_sources()[1], "gold_pages": [2]},
    ]
    # lows 是 0 基：第 1 页 = lows[0]
    lows = [
        "赛力斯汽车9月批发销量 1.2 万辆",
        "赛力斯汽车9月零售销量 2.4 万辆",
        "行情：赛力斯汽车9月批发销量 1.2 万辆，零售销量 2.4 万辆",
    ]
    it = derive_multihop(src, lows=lows)[0]
    # 第 3 页复述了两处事实，也要标成 gold；判分不能只认来源页
    assert it["gold_pages"] == [1, 2, 3]


def test_grade_multihop_accepts_another_page_with_same_figures():
    it = derive_multihop(_hop_sources())[0]
    lows = [
        "赛力斯汽车9月批发销量 1.2 万辆",
        "赛力斯汽车9月零售销量 2.4 万辆",
        "行情：批发销量 1.2 万辆，零售销量 2.4 万辆",
    ]
    alt = {
        "answer": "批发 1.2 万辆，零售 2.4 万辆 [p3]",
        "citations": [{"pages": [3]}],
        "read_pages": [3],
    }
    row = grade(it, alt, lows)
    assert row["ok"] and row["hop_ok"] == [True, True]
    assert row["page_hit"] is False


def test_grade_multihop_needs_both_hops():
    it = derive_multihop(_hop_sources())[0]
    lows = ["", "赛力斯汽车9月批发销量 1.2 万辆", "赛力斯汽车9月零售销量 2.4 万辆"]
    both = {
        "answer": "批发 1.2 万辆，零售 2.4 万辆 [p1][p2]",
        "citations": [{"pages": [1, 2]}],
        "read_pages": [1, 2],
    }
    row = grade(it, both, lows)
    assert row["ok"] and row["hop_ok"] == [True, True]
    one = {"answer": "批发 1.2 万辆 [p1]", "citations": [{"pages": [1]}], "read_pages": [1]}
    row = grade(it, one, lows)
    assert not row["ok"] and row["hop_ok"] == [True, False]


def test_derive_multiturn_elides_subject_and_grade_needs_each_turn():
    items = derive_multiturn(_hop_sources())
    assert len(items) == 1
    it = items[0]
    q1, q2 = it["turns"][0]["question"], it["turns"][1]["question"]
    assert q2.startswith("那") and "赛力斯汽车" not in q2
    history = [{"role": "user", "content": q1}]
    assert resolve_followup(q2, history) == "赛力斯汽车9月零售销量是多少？"
    lows = ["赛力斯汽车9月批发销量 1.2 万辆", "赛力斯汽车9月零售销量 2.4 万辆"]
    results = [
        {"answer": "批发 1.2 万辆 [p1]", "citations": [{"pages": [1]}], "read_pages": [1],
         "resolved_question": q1},
        {"answer": "零售 2.4 万辆 [p2]", "citations": [{"pages": [2]}], "read_pages": [2],
         "resolved_question": "赛力斯汽车9月零售销量是多少？"},
    ]
    row = grade_multiturn(it, results, lows)
    assert row["ok"] and row["turn_ok"] == [True, True]
    row = grade_multiturn(it, [results[0], {**results[1], "answer": "不确定"}], lows)
    assert not row["ok"] and row["turn_ok"] == [True, False]


def test_derive_multiturn_skips_multi_value_followup():
    src = [
        {
            "id": "m1", "tier": "A", "kind": "literal", "article": "A1",
            "question": "皇冠新材的发行价为多少？", "gold_pages": [1], "gold_figures": ["16.98"],
        },
        {
            "id": "m2", "tier": "S", "kind": "literal", "article": "A1",
            "question": "皇冠新材的营业收入分别为多少？", "gold_pages": [2], "gold_figures": ["28.95"],
        },
    ]
    assert derive_multiturn(src) == []
