from __future__ import annotations

import pytest

from treewiki.core import native


def test_h1_starts_new_page(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 第一章\n\n内容 A\n\n# 第二章\n\n内容 B", encoding="utf-8")
    pages = native.parse_local(p, page_chars=1000)["pages"]
    assert len(pages) == 2
    assert pages[0].startswith("# 第一章")
    assert pages[1].startswith("# 第二章")


def test_h3_stays_in_page(tmp_path):
    p = tmp_path / "b.md"
    p.write_text("# 章\n\n### 小节\n\n内容", encoding="utf-8")
    pages = native.parse_local(p, page_chars=1000)["pages"]
    assert len(pages) == 1
    assert "### 小节" in pages[0]


def test_long_text_chunked(tmp_path):
    p = tmp_path / "c.txt"
    body = "\n\n".join(f"段落 {i} " + "x" * 240 for i in range(20))
    p.write_text(body, encoding="utf-8")
    pages = native.parse_local(p, page_chars=1000)["pages"]
    assert len(pages) >= 5
    assert all(len(page) <= 1000 for page in pages)
    assert pages[0].startswith("段落 0")


def test_gb18030_fallback(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("中文内容，GBK 编码。", encoding="gb18030")
    assert "GBK 编码" in native.parse_local(p)["md"]


def test_empty_file_raises(tmp_path):
    p = tmp_path / "e.md"
    p.write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(ValueError):
        native.parse_local(p)
