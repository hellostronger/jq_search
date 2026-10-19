from __future__ import annotations

import re
from pathlib import Path

PAGE_CHARS = 2500
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文本编码（支持 UTF-8 / GB18030）: {path.name}")


def split_pages(text: str, page_chars: int = PAGE_CHARS) -> list[str]:
    """把 md/txt 切成合成页：一级/二级标题另起一页，超长段落按行边界切开。"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        raise ValueError("文件内容为空")
    pages: list[str] = []
    buf = ""
    for block in blocks:
        heading = _HEADING_RE.match(block)
        if buf and heading and (
            len(heading.group(1)) <= 2 or len(buf) >= page_chars // 2
        ):
            pages.append(buf)
            buf = ""
        for part in _split_long(block, page_chars):
            if buf and len(buf) + 2 + len(part) > page_chars:
                pages.append(buf)
                buf = ""
            buf = f"{buf}\n\n{part}" if buf else part
            if len(buf) >= page_chars:
                pages.append(buf)
                buf = ""
    if buf:
        pages.append(buf)
    return pages


def _split_long(block: str, limit: int) -> list[str]:
    parts = []
    while len(block) > limit:
        window = block[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit
        parts.append(block[:cut].rstrip())
        block = block[cut:].lstrip()
    if block:
        parts.append(block)
    return parts


def parse_local(path: Path, page_chars: int = PAGE_CHARS) -> dict:
    """本地解析 md/txt（不经过 MinerU），返回与 parse_cloud 相同的结构。"""
    text = read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    pages = split_pages(text, page_chars)
    content_list = {
        "pages": [
            {"page_idx": i, "blocks": [{"type": "text", "text": page}]}
            for i, page in enumerate(pages)
        ]
    }
    return {"md": text, "content_list": content_list, "pages": pages}
