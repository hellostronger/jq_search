from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .tree import TreeDoc


class KB:
    """知识库目录布局：raw / cache / index / wiki / log（设计见 docs/DESIGN.md §2.2）"""

    def __init__(self, home: Path | str):
        self.home = Path(home)
        self.raw_dir = self.home / "raw"
        self.parse_dir = self.home / "cache" / "parse"
        self.pages_dir = self.home / "cache" / "pages"
        self.trees_dir = self.home / "index" / "trees"
        self.wiki_dir = self.home / "wiki"
        self.runs_dir = self.home / "log" / "agent_runs"
        for d in (
            self.home,
            self.raw_dir,
            self.parse_dir,
            self.pages_dir,
            self.trees_dir,
            self.wiki_dir,
            self.runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def sources_path(self) -> Path:
        return self.home / "index" / "sources.json"

    def sources(self) -> dict[str, dict[str, Any]]:
        if self.sources_path.is_file():
            return json.loads(self.sources_path.read_text(encoding="utf-8"))
        return {}

    def _save_sources(self, data: dict) -> None:
        self.sources_path.parent.mkdir(parents=True, exist_ok=True)
        self.sources_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def new_doc_id(self) -> str:
        nums = [
            int(k.rsplit("_", 1)[-1])
            for k in self.sources()
            if k.startswith("src_") and k.rsplit("_", 1)[-1].isdigit()
        ]
        return f"src_{(max(nums) + 1) if nums else 1:04d}"

    def register(
        self, doc_id: str, path: Path, sha256: str, pages: int, description: str = ""
    ) -> None:
        data = self.sources()
        data[doc_id] = {
            "path": str(path),
            "sha256": sha256,
            "pages": pages,
            "description": description,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "status": "active",
        }
        self._save_sources(data)

    def write_pages(self, doc_id: str, pages: list[str]) -> None:
        d = self.pages_dir / doc_id
        d.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(pages, start=1):
            (d / f"p{i:03d}.md").write_text(text, encoding="utf-8")

    def read_pages(self, doc_id: str, a: int, b: int) -> list[str]:
        d = self.pages_dir / doc_id
        out = []
        for i in range(a, min(b, self.page_count(doc_id)) + 1):
            p = d / f"p{i:03d}.md"
            out.append(p.read_text(encoding="utf-8") if p.is_file() else "")
        return out

    def page_count(self, doc_id: str) -> int:
        return int(self.sources().get(doc_id, {}).get("pages", 0))

    def save_tree(self, tree: TreeDoc) -> None:
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        (self.trees_dir / f"{tree.doc_id}.json").write_text(
            tree.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_tree(self, doc_id: str) -> TreeDoc:
        p = self.trees_dir / f"{doc_id}.json"
        if not p.is_file():
            raise FileNotFoundError(f"未找到 {doc_id} 的树索引")
        return TreeDoc.model_validate_json(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
