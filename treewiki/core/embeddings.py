from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx


class EmbeddingError(RuntimeError):
    pass


_TRANSIENT = {429, 500, 502, 503, 504}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OpenAIEmbedder:
    """OpenAI 兼容 /embeddings 客户端：批量请求、退避重试、按文本哈希落盘缓存。

    网关的 key 只允许一个在途请求，这里固定串行批处理，不靠并发换速度。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        batch_size: int = 16,
        timeout: float = 180.0,
        cache_dir: Path | str | None = None,
        trust_env: bool = False,
    ):
        if not base_url:
            raise EmbeddingError("未配置 embedding base_url")
        if not api_key:
            raise EmbeddingError(
                "未配置 embedding api_key：在 llm.embedding_api_key 或 EMBEDDING_API_KEY 里填"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, trust_env=trust_env)
        self._cache_path: Path | None = None
        self._cache: dict[str, list[float]] = {}
        if cache_dir:
            slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
            self._cache_path = Path(cache_dir) / f"embeddings_{slug}.jsonl"
            self._load_cache()

    def _load_cache(self) -> None:
        p = self._cache_path
        if not p or not p.is_file():
            return
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    h, v = row["h"], row["v"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if isinstance(v, list) and v:
                    self._cache[h] = v

    def _append_cache(self, rows: list[dict]) -> None:
        if not self._cache_path or not rows:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        delay = 5.0
        last = ""
        for attempt in range(1, 7):
            try:
                resp = self._client.post(
                    f"{self.base_url}/embeddings", json=payload, headers=headers
                )
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < 6:
                    time.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                raise EmbeddingError(f"embedding 请求失败：{last}") from exc
            if resp.status_code == 200:
                items = resp.json().get("data") or []
                if len(items) != len(texts):
                    raise EmbeddingError(
                        f"embedding 返回条数不符：{len(items)} != {len(texts)}"
                    )
                items = sorted(items, key=lambda x: int(x.get("index", 0)))
                return [item["embedding"] for item in items]
            last = resp.text[:300]
            if resp.status_code == 401:
                raise EmbeddingError(
                    "embedding 鉴权失败（401）：当前 key 没有 /embeddings 权限；"
                    "请在 llm.embedding_api_key 或 EMBEDDING_API_KEY 配置可用 key"
                )
            if resp.status_code in _TRANSIENT and attempt < 6:
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            raise EmbeddingError(f"embedding 失败 HTTP {resp.status_code}: {last}")
        raise EmbeddingError(f"embedding 重试 6 次仍失败：{last}")

    def encode(self, texts, progress=None) -> list[list[float]]:
        texts = list(texts)
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for i, text in enumerate(texts):
            h = _digest(text)
            if h in self._cache:
                out[i] = self._cache[h]
            else:
                missing.append((i, h, text))
        if progress and missing and len(missing) < len(texts):
            progress(f"embedding 缓存命中 {len(texts) - len(missing)}/{len(texts)}")
        done = 0
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vecs = self._request([x[2] for x in batch])
            fresh = []
            for (i, h, _text), vec in zip(batch, vecs):
                out[i] = vec
                self._cache[h] = vec
                fresh.append({"h": h, "v": vec})
            self._append_cache(fresh)
            done += len(batch)
            if progress:
                progress(f"embedding {done}/{len(missing)}")
        dim = len(out[0]) if out and out[0] else 0
        if not dim:
            raise EmbeddingError("embedding 结果为空")
        for vec in out:
            if not vec or len(vec) != dim:
                raise EmbeddingError("embedding 维度不一致，可能混用了不同模型")
        return out  # type: ignore[return-value]

    def close(self) -> None:
        self._client.close()
