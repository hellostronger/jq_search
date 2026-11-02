from __future__ import annotations

from treewiki.pipelines.neo4j_hybrid import (
    lucene_query,
    page_id,
    parse_page_id,
    rrf_fuse,
)


def test_page_id_roundtrip():
    pid = page_id("src_0001", 7)
    assert pid == "src_0001:p0007"
    assert parse_page_id(pid) == ("src_0001", 7)


def test_rrf_prefers_consensus():
    fused, scores, ranks = rrf_fuse(
        {
            "vector": ["a", "b", "c"],
            "fulltext": ["b", "c", "a"],
            "graph": ["c", "d", "b"],
        },
        k=60,
    )
    assert fused[0] in {"b", "c"}
    assert ranks[fused[0]]
    assert scores[fused[0]] > scores["d"]


def test_rrf_weights_change_order():
    ranked = {"vector": ["a", "b"], "fulltext": ["b", "a"]}
    plain, _, _ = rrf_fuse(ranked, k=60)
    weighted, _, _ = rrf_fuse(ranked, k=60, weights={"vector": 5.0})
    assert plain[0] == "a"
    assert weighted[0] == "a"

    ranked = {"vector": ["b", "a"], "fulltext": ["a", "b"]}
    weighted, _, _ = rrf_fuse(ranked, k=60, weights={"fulltext": 5.0})
    assert weighted[0] == "a"


def test_rrf_ignores_duplicate_ids_in_one_channel():
    fused, scores, _ = rrf_fuse({"vector": ["a", "a", "b"]})
    assert abs(scores["a"] - 1 / 61) < 1e-9
    assert abs(scores["b"] - 1 / 63) < 1e-9
    assert fused == ["a", "b"]


def test_lucene_query_escapes_specials():
    query = lucene_query('9月CPI同比涨幅是多少？"quoted"')
    assert query
    assert '"' not in query
    assert "cpi" in query


def test_lucene_query_empty():
    assert lucene_query("  ") == ""
