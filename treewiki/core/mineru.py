from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx

BASE = "https://mineru.net"
POLL_INTERVAL = 3.0
POLL_TIMEOUT = 900.0


class MinerUError(RuntimeError):
    pass


def parse_cloud(
    path: Path,
    token: str,
    model_version: str = "pipeline",
    language: str = "ch",
    is_ocr: bool = False,
    enable_formula: bool = True,
    stage_cb: Callable[[str], None] | None = None,
) -> dict:
    """调用 mineru.net 云 API 解析文档，返回 {"md": str, "content_list": list|dict}。"""
    if not token:
        raise MinerUError("未配置 MinerU token（kb/config.yaml 的 mineru.token 或环境变量 MINERU_API_TOKEN）")
    if path.stat().st_size > 200 * 1024 * 1024:
        raise MinerUError("文件超过 MinerU 单文件 200MB 限制")

    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{BASE}/api/v4/file-urls/batch",
            headers=headers,
            json={
                "files": [{"name": path.name, "data_id": path.name}],
                "model_version": model_version,
                "language": language,
                "is_ocr": is_ocr,
                "enable_formula": enable_formula,
                "enable_table": True,
            },
        )
        data = _check(r)
        batch_id = data["batch_id"]
        upload_url = data["file_urls"][0]
        if stage_cb:
            stage_cb(f"MinerU 任务已创建（batch={batch_id}），上传文件…")

        r2 = c.put(upload_url, content=path.read_bytes())
        if r2.status_code >= 400:
            raise MinerUError(f"文件上传失败 HTTP {r2.status_code}: {r2.text[:200]}")
        if stage_cb:
            stage_cb("文件已上传，等待 MinerU 解析（最长 15 分钟）…")

    zip_url = _poll_batch(batch_id, headers, stage_cb)
    if stage_cb:
        stage_cb("解析完成，下载结果…")
    with httpx.Client(timeout=300, follow_redirects=True) as c:
        r = c.get(zip_url)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))

    md_text: str | None = None
    content_list: Any = None
    for name in zf.namelist():
        base = Path(name).name
        if base.endswith(".md") and md_text is None and not base.startswith("."):
            md_text = zf.read(name).decode("utf-8", "ignore")
        elif "content_list" in base and base.endswith(".json") and content_list is None:
            raw = zf.read(name).decode("utf-8", "ignore")
            content_list = json.loads(raw) if raw.strip() else None
    if md_text is None or content_list is None:
        raise MinerUError(f"MinerU 结果缺少 md/content_list: {zf.namelist()}")
    return {"md": md_text, "content_list": content_list}


def _check(r: httpx.Response) -> dict:
    try:
        data = r.json()
    except Exception:
        raise MinerUError(f"MinerU 接口异常 HTTP {r.status_code}: {r.text[:200]}")
    if data.get("code") != 0:
        raise MinerUError(f"MinerU 接口返回错误: {data}")
    return data["data"]


def _poll_batch(
    batch_id: str, headers: dict, stage_cb: Callable[[str], None] | None
) -> str:
    start = time.time()
    with httpx.Client(timeout=60) as c:
        while True:
            r = c.get(f"{BASE}/api/v4/extract-results/batch/{batch_id}", headers=headers)
            data = _check(r)
            item = data["extract_result"][0]
            state = item.get("state")
            if state == "done":
                return item["full_zip_url"]
            if state == "failed":
                raise MinerUError(f"MinerU 解析失败: {item.get('err_msg')}")
            if time.time() - start > POLL_TIMEOUT:
                raise MinerUError("MinerU 解析超时")
            if stage_cb:
                stage_cb(f"MinerU 状态：{state} …")
            time.sleep(POLL_INTERVAL)


def normalize_pages(content_list: Any) -> list[str]:
    """把 MinerU 的 content_list 归一化为页级 markdown 列表（下标 i 即第 i+1 页）。"""
    blocks: list[tuple[int, dict]] = []
    if isinstance(content_list, dict) and "pages" in content_list:
        for page in content_list["pages"]:
            pidx = int(page.get("page_idx", 0))
            for b in page.get("blocks", []):
                blocks.append((pidx, b))
    elif isinstance(content_list, list):
        for b in content_list:
            if isinstance(b, dict):
                blocks.append((int(b.get("page_idx", 0)), b))
    else:
        raise MinerUError("无法识别的 content_list 格式")

    pages: dict[int, list[str]] = {}
    declared = 0
    if isinstance(content_list, dict) and "pages" in content_list:
        declared = len(content_list["pages"])
    for pidx, b in blocks:
        text = _render_block(b)
        if text and text.strip():
            pages.setdefault(pidx, []).append(text.strip())
    if not pages:
        return [""] * declared if declared else []
    n = max(max(pages) + 1, declared)
    return ["\n\n".join(pages.get(i, [])) for i in range(n)]


def _render_block(b: dict) -> str:
    t = b.get("type")
    if t == "text":
        return b.get("text") or _content_text(b.get("content"))
    if t == "table":
        body = b.get("table_body") or b.get("html") or ""
        cap = " ".join(b.get("table_caption") or [])
        return f"{cap}\n\n{body}".strip()
    if t == "image":
        cap = " ".join(b.get("image_caption") or [])
        return f"{cap}\n[图片 {b.get('img_path', '')}]".strip()
    if t == "equation":
        return f"$${b.get('text', '')}$$"
    return b.get("text") or _content_text(b.get("content"))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            x.get("content", "") if isinstance(x, dict) else str(x) for x in content
        )
    return ""
