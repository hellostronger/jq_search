from treewiki.core.tree import (
    Node,
    assign_ids,
    find_node,
    flash_skeleton,
    iter_nodes,
    skeleton_text,
)


def _pages():
    return [
        "# 第一章 Alpha\n\nintro of alpha",
        "## 1.1 Beta\n\nbeta text",
        "## 1.2 Gamma\n\ngamma text",
        "# 第二章 Delta\n\ndelta text",
        "more delta text",
    ]


def test_flash_skeleton_structure():
    pages = _pages()
    structure = flash_skeleton(pages)
    assert structure is not None
    titles = [n.title for n in structure]
    assert titles == ["第一章 Alpha", "第二章 Delta"]

    ch1, ch2 = structure
    assert (ch1.start_index, ch1.end_index) == (1, 3)
    assert (ch2.start_index, ch2.end_index) == (4, 5)
    child_titles = [n.title for n in ch1.nodes]
    assert child_titles == ["1.1 Beta", "1.2 Gamma"]
    beta, gamma = ch1.nodes
    assert (beta.start_index, beta.end_index) == (2, 2)
    assert (gamma.start_index, gamma.end_index) == (3, 3)


def test_flash_skeleton_needs_headings():
    assert flash_skeleton(["纯文本页", "没有标题"]) is None


def test_repeated_headers_dropped():
    pages = [
        "# 报告\n\n## 真章节A\n\ncontent",
        "## 页眉重复\n\nx",
        "## 页眉重复\n\ny",
        "## 页眉重复\n\nz",
        "## 页眉重复\n\nw\n\n## 真章节B\n\ntail",
    ]
    structure = flash_skeleton(pages)
    assert structure is not None
    for n, _ in iter_nodes(structure):
        assert n.title != "页眉重复"
    titles = [n.title for n, _ in iter_nodes(structure)]
    assert "真章节A" in titles and "真章节B" in titles


def test_assign_ids_and_find():
    structure = flash_skeleton(_pages())
    assign_ids(structure)
    ids = [n.node_id for n, _ in iter_nodes(structure)]
    assert ids == ["0001", "0002", "0003", "0004"]
    assert find_node(structure, "0003").title == "1.2 Gamma"


def test_skeleton_text():
    structure = flash_skeleton(_pages())
    assign_ids(structure)
    text = skeleton_text(structure)
    assert "[0001] 第一章 Alpha (p1-3)" in text
    assert "[0003] 1.2 Gamma (p3-3)" in text


def test_node_model_roundtrip():
    n = Node(title="t", start_index=1, end_index=2, node_id="0001", summary="s")
    data = n.model_dump()
    restored = Node.model_validate(data)
    assert restored == n
