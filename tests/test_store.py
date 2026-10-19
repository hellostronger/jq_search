from treewiki.core.store import KB, sha256_file
from treewiki.core.tree import Node, TreeDoc


def test_kb_roundtrip(tmp_path):
    kb = KB(tmp_path / "kb")
    doc_id = kb.new_doc_id()
    assert doc_id == "src_0001"

    src = tmp_path / "doc.pdf"
    src.write_bytes(b"fake pdf bytes")
    sha = sha256_file(src)
    kb.register(doc_id, src, sha, pages=3)
    assert kb.sources()[doc_id]["pages"] == 3
    assert kb.new_doc_id() == "src_0002"

    kb.write_pages(doc_id, ["p1 text", "p2 text", "p3 text"])
    assert kb.read_pages(doc_id, 1, 2) == ["p1 text", "p2 text"]
    assert kb.read_pages(doc_id, 1, 99) == ["p1 text", "p2 text", "p3 text"]
    assert kb.page_count(doc_id) == 3

    tree = TreeDoc(
        doc_id=doc_id,
        doc_name="doc.pdf",
        doc_description="测试文档",
        structure=[Node(title="章节", start_index=1, end_index=3, node_id="0001")],
    )
    kb.save_tree(tree)
    loaded = kb.load_tree(doc_id)
    assert loaded == tree


def test_kb_creates_layout(tmp_path):
    kb = KB(tmp_path / "kb")
    for d in (kb.raw_dir, kb.parse_dir, kb.pages_dir, kb.trees_dir, kb.wiki_dir, kb.runs_dir):
        assert d.is_dir()
