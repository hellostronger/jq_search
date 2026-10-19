import pytest

from treewiki.core.mineru import MinerUError, normalize_pages


def test_normalize_pages_v1_list():
    cl = [
        {"type": "text", "text": "# Title", "page_idx": 0},
        {"type": "text", "text": "body1", "page_idx": 0},
        {"type": "table", "table_body": "<table><tr><td>1</td></tr></table>", "table_caption": ["表1"], "page_idx": 1},
        {"type": "image", "img_path": "images/a.jpg", "page_idx": 1},
        {"type": "equation", "text": "x^2", "page_idx": 2},
    ]
    pages = normalize_pages(cl)
    assert len(pages) == 3
    assert "# Title" in pages[0] and "body1" in pages[0]
    assert "表1" in pages[1] and "<table>" in pages[1] and "images/a.jpg" in pages[1]
    assert "$$x^2$$" in pages[2]


def test_normalize_pages_v2_grouped():
    cl = {
        "pages": [
            {"page_idx": 0, "blocks": [{"type": "text", "content": "hello"}]},
            {"page_idx": 1, "blocks": [{"type": "text", "content": [{"content": "world"}]}]},
            {"page_idx": 2, "blocks": []},
        ]
    }
    pages = normalize_pages(cl)
    assert pages == ["hello", "world", ""]


def test_normalize_pages_gaps():
    cl = [{"type": "text", "text": "p2 content", "page_idx": 1}]
    pages = normalize_pages(cl)
    assert len(pages) == 2
    assert pages[0] == ""
    assert "p2 content" in pages[1]


def test_normalize_pages_rejects_garbage():
    with pytest.raises(MinerUError):
        normalize_pages("not a list")
