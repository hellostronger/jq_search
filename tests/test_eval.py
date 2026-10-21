import random

from treewiki.eval.dataset import (
    Article,
    _cloze,
    _longest_run,
    _MONEY,
    _paraphrase,
    classify_by_lexicon,
    quota,
    split_articles,
)
from treewiki.eval.runner import _ngram_terms, _pages_with, cited_pages, grade

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


def test_paraphrase_drops_latin_entity_and_stays_answerable():
    q = "iPhone 17 的同比增长是多少？"
    out = _paraphrase(q)
    assert out and "iPhone" not in out and "去年同期" in out


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
