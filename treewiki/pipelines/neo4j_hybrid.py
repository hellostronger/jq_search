from __future__ import annotations

import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from ..core.embeddings import OpenAIEmbedder
from ..core.entities import EntityIndex, normalize
from ..core.store import KB
from .query import _ANCHOR_STOP, _usable_anchor, question_anchors

try:
    import jieba
except ImportError:  # 中文分词可选；缺省退回字符 n-gram
    jieba = None

PAGE = "TWPage"
ENTITY = "TWEntity"
DOC = "TWDoc"
RRF_K = 60

_QUERY_STOP = frozenset(
    """
    什么 怎么 如何 为什么 多少 哪些 哪个 哪家 怎样 怎么样 是否 有没有 能不能 会不会
    以及 其中 请问 一下 具体 分别 目前 今年 去年 时候 情况 相关 方面 进行 可以 这个 那个
    """.split()
)
_BIGRAM_STOP = _QUERY_STOP | frozenset("是多 为多 是多 少？ 了吗 呢？ 什么".split())
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-_]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def page_id(doc_id: str, page: int) -> str:
    return f"{doc_id}:p{int(page):04d}"


def parse_page_id(value: str) -> tuple[str, int]:
    doc, sep, page = value.rpartition(":p")
    if not sep or not page.isdigit():
        raise ValueError(f"非法页 id: {value}")
    return doc, int(page)


def _tokens(text: str) -> list[str]:
    """中文二元组 + 拉丁词 + jieba 词。

    Neo4j 的 cjk analyzer 就是按二元组建索引，查询侧也用二元组才能对得上；
    jieba 补上“同比/涨幅”这类完整词，避免只靠字面二元组丢语义。
    """
    out: list[str] = []

    def add(token: str) -> None:
        token = token.strip().lower()
        if not token or token in _QUERY_STOP or token in out:
            return
        if len(token) == 1 and not token.isdigit():
            return
        out.append(token)

    for token in _TOKEN_RE.findall(text or ""):
        if _CJK_RUN_RE.fullmatch(token):
            for i in range(len(token) - 1):
                gram = token[i : i + 2]
                if gram not in _BIGRAM_STOP:
                    add(gram)
        else:
            add(token)
    if jieba is not None:
        for word in jieba.cut(text or ""):
            if len(word) >= 2:
                add(word)
    return out[:30]


def _escape_lucene(token: str) -> str:
    return "".join("\\" + ch if ch in _LUCENE_SPECIAL else ch for ch in token)


def lucene_query(text: str) -> str:
    tokens = _tokens(text)
    if not tokens:
        stripped = (text or "").strip()
        tokens = [stripped] if stripped else []
    return " OR ".join(_escape_lucene(t) for t in tokens)


def rrf_fuse(
    ranked: dict[str, list[str]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, int]]]:
    """倒数排名融合：每路按名次给 1/(k+rank)，同一路内重复页只算最高名次。"""
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for name, ids in ranked.items():
        weight = (weights or {}).get(name, 1.0)
        seen: set[str] = set()
        for rank, pid in enumerate(ids, start=1):
            if pid in seen:
                continue
            seen.add(pid)
            scores[pid] += weight / (k + rank)
            ranks[pid][name] = rank
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))
    if limit:
        ordered = ordered[:limit]
    return ordered, dict(scores), {pid: ranks[pid] for pid in ordered}


def iter_graph_entities(
    index: EntityIndex, total_pages: int, df_cap: int = 200
) -> Iterable[dict]:
    """只把有区分度的实体放进图：去掉泛词、超高频词和只出现一页的碎片。"""
    for entity in index.entities:
        if not _usable_anchor(entity):
            continue
        name = normalize(entity.canonical)
        if not (2 <= len(name) <= 20):
            continue
        if entity.frequency < 3:
            continue
        pages: dict[tuple[str, int], int] = {}
        for m in entity.mentions:
            doc_id = (m.doc_id or "").strip()
            page = int(m.page or 0)
            if doc_id and page > 0:
                pages[(doc_id, page)] = pages.get((doc_id, page), 0) + max(1, int(m.count or 1))
        df = len(pages)
        if df < 2 or df > df_cap:
            continue
        aliases: list[str] = []
        for alias in entity.aliases:
            norm = normalize(alias)
            if 2 <= len(norm) <= 20 and norm not in _ANCHOR_STOP and norm not in aliases:
                aliases.append(norm)
        names = " ".join(dict.fromkeys([name, *aliases[:12]]))
        idf = math.log((max(1, total_pages) + 1) / (df + 1)) + 1.0
        yield {
            "id": entity.id,
            "canonical": entity.canonical,
            "type": entity.type,
            "df": df,
            "freq": entity.frequency,
            "idf": round(idf, 4),
            "names": names,
            "mentions": [{"pid": page_id(d, p), "count": c} for (d, p), c in pages.items()],
        }


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Neo4jHybrid:
    """Neo4j 单实例三路检索：向量索引 + Lucene 全文 + Cypher 实体图遍历，RRF 合并。"""

    def __init__(
        self,
        cfg: dict,
        cache_dir: Path | str | None = None,
        kb: KB | None = None,
    ):
        ncfg = cfg.get("neo4j", {}) or {}
        self.cfg = cfg
        self.ncfg = ncfg
        self.uri = os.environ.get("NEO4J_URI") or ncfg.get("uri") or "bolt://127.0.0.1:7687"
        self.user = os.environ.get("NEO4J_USER") or ncfg.get("user") or "neo4j"
        password = os.environ.get("NEO4J_PASSWORD") or ncfg.get("password") or ""
        self.database = os.environ.get("NEO4J_DATABASE") or ncfg.get("database") or "neo4j"
        if not password:
            raise RuntimeError(
                "未配置 Neo4j 密码：设置 NEO4J_PASSWORD，或写进 kb/config.yaml 的 neo4j.password"
            )
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, password))
        self.driver.verify_connectivity()
        self.index: EntityIndex | None = kb.load_entities() if kb is not None else None

        llm = cfg.get("llm", {}) or {}
        model = ncfg.get("embedding_model") or llm.get("embedding_model") or "BAAI/bge-m3"
        api_key = (
            ncfg.get("embedding_api_key")
            or llm.get("embedding_api_key")
            or llm.get("api_key")
            or ""
        )
        self.embedder = OpenAIEmbedder(
            base_url=llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", ""),
            api_key=api_key,
            model=model,
            batch_size=int(ncfg.get("embedding_batch", 16)),
            timeout=float(ncfg.get("embedding_timeout", 180)),
            cache_dir=cache_dir,
        )

    # ---------------------------------------------------------------- 连接/工具

    def close(self) -> None:
        self.embedder.close()
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self, query: str, **params):
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, **params))

    def _index_of(self, kb: KB) -> EntityIndex:
        if self.index is None:
            self.index = kb.load_entities()
        return self.index

    # -------------------------------------------------------------------- 建库

    def build(
        self,
        kb: KB,
        doc_ids: list[str] | None = None,
        rebuild: bool = True,
        stage: Callable[[str], None] = print,
    ) -> dict:
        docs = doc_ids or sorted(kb.sources())
        pages: list[dict] = []
        for doc_id in docs:
            total = kb.page_count(doc_id)
            for page, text in enumerate(kb.read_pages(doc_id, 1, total), start=1):
                if text.strip():
                    pages.append(
                        {
                            "id": page_id(doc_id, page),
                            "doc_id": doc_id,
                            "page": page,
                            "text": text,
                        }
                    )
        if not pages:
            raise RuntimeError("没有可入库的页面")

        stage(f"页面 {len(pages)}，开始 embedding（{self.embedder.model}）")
        vectors = self.embedder.encode([p["text"] for p in pages], progress=stage)
        dim = len(vectors[0])
        stage(f"embedding 完成，维度 {dim}；准备 Neo4j schema")
        self._prepare_schema(dim)
        if rebuild:
            self._clear(stage)
        self._write_docs(kb, docs)
        self._write_pages(pages, vectors, stage)
        entity_count, rel_count = self._write_entities(kb, len(pages), stage)
        states = self._await_indexes(stage)
        stats = {
            "docs": len(docs),
            "pages": len(pages),
            "entities": entity_count,
            "mentions": rel_count,
            "dim": dim,
            "indexes": states,
            "embedding_model": self.embedder.model,
        }
        stage(
            f"完成：{stats['pages']} 页 / {stats['entities']} 实体 / "
            f"{stats['mentions']} 条提及，索引 {states}"
        )
        return stats

    def _prepare_schema(self, dim: int) -> None:
        self._run(
            f"CREATE CONSTRAINT tw_page_id IF NOT EXISTS FOR (p:{PAGE}) REQUIRE p.id IS UNIQUE"
        )
        self._run(
            f"CREATE CONSTRAINT tw_entity_id IF NOT EXISTS FOR (e:{ENTITY}) REQUIRE e.id IS UNIQUE"
        )
        self._run(
            f"CREATE CONSTRAINT tw_doc_id IF NOT EXISTS FOR (d:{DOC}) REQUIRE d.id IS UNIQUE"
        )
        # 全量重建时向量索引一并重建：既保证维度跟着 embedding 模型走，
        # 也避免旧维度索引让查询直接报错。
        self._run("DROP INDEX tw_page_vec IF EXISTS")
        self._run(
            f"CREATE VECTOR INDEX tw_page_vec IF NOT EXISTS FOR (p:{PAGE}) ON p.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: " + str(int(dim)) + ", "
            "`vector.similarity_function`: 'cosine'}}"
        )
        self._run(
            f"CREATE FULLTEXT INDEX tw_page_ft IF NOT EXISTS FOR (p:{PAGE}) ON EACH [p.text] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
        )
        self._run(
            f"CREATE FULLTEXT INDEX tw_entity_ft IF NOT EXISTS FOR (e:{ENTITY}) "
            "ON EACH [e.names] OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
        )

    def _clear(self, stage: Callable[[str], None]) -> None:
        for label in (PAGE, ENTITY, DOC):
            while True:
                with self.driver.session(database=self.database) as session:
                    summary = session.run(
                        f"MATCH (n:{label}) WITH n LIMIT 5000 DETACH DELETE n"
                    ).consume()
                if summary.counters.nodes_deleted < 5000:
                    break
        stage("已清空旧的 TWPage / TWEntity / TWDoc 节点")

    def _write_docs(self, kb: KB, doc_ids: list[str]) -> None:
        sources = kb.sources()
        rows = [
            {
                "id": doc_id,
                "pages": int(sources.get(doc_id, {}).get("pages", 0)),
                "description": sources.get(doc_id, {}).get("description", ""),
            }
            for doc_id in doc_ids
        ]
        self._run(
            f"UNWIND $rows AS row MERGE (d:{DOC} {{id: row.id}}) "
            "SET d.pages = row.pages, d.description = row.description",
            rows=rows,
        )

    def _write_pages(
        self, pages: list[dict], vectors: list[list[float]], stage: Callable[[str], None]
    ) -> None:
        query = (
            f"UNWIND $rows AS row MERGE (p:{PAGE} {{id: row.id}}) "
            "SET p.doc_id = row.doc_id, p.page = row.page, p.text = row.text, "
            "p.embedding = row.embedding"
        )
        done = 0
        for batch in _chunks(list(zip(pages, vectors)), 100):
            rows = [
                {
                    "id": page["id"],
                    "doc_id": page["doc_id"],
                    "page": page["page"],
                    "text": page["text"],
                    "embedding": vec,
                }
                for page, vec in batch
            ]
            self._run(query, rows=rows)
            done += len(rows)
            if done % 500 == 0 or done == len(pages):
                stage(f"写入页面 {done}/{len(pages)}")

    def _write_entities(
        self, kb: KB, total_pages: int, stage: Callable[[str], None]
    ) -> tuple[int, int]:
        index = self._index_of(kb)
        cap = int(self.ncfg.get("entity_df_cap", 200))
        entities = list(iter_graph_entities(index, total_pages, cap))
        stage(f"实体入图：{len(entities)} 个（df≤{cap}）")
        node_query = (
            f"UNWIND $rows AS row MERGE (e:{ENTITY} {{id: row.id}}) "
            "SET e.canonical = row.canonical, e.type = row.type, e.df = row.df, "
            "e.freq = row.freq, e.idf = row.idf, e.names = row.names"
        )
        rel_query = (
            "UNWIND $rows AS row "
            f"MATCH (e:{ENTITY} {{id: row.eid}}), (p:{PAGE} {{id: row.pid}}) "
            "MERGE (e)-[m:MENTIONS]->(p) SET m.count = row.count"
        )
        nodes = [{k: v for k, v in e.items() if k != "mentions"} for e in entities]
        for batch in _chunks(nodes, 1000):
            self._run(node_query, rows=batch)
        stage(f"实体节点写入完成 {len(nodes)}，开始写提及关系")
        rels = [
            {"eid": e["id"], "pid": m["pid"], "count": m["count"]}
            for e in entities
            for m in e["mentions"]
        ]
        done = 0
        for batch in _chunks(rels, 2000):
            self._run(rel_query, rows=batch)
            done += len(batch)
            if done % 20000 == 0 or done == len(rels):
                stage(f"提及关系 {done}/{len(rels)}")
        return len(nodes), len(rels)

    def _await_indexes(self, stage: Callable[[str], None]) -> dict[str, str]:
        names = ["tw_page_vec", "tw_page_ft", "tw_entity_ft"]
        deadline = time.time() + 300
        states: dict[str, str] = {}
        while time.time() < deadline:
            rows = self._run(
                "SHOW INDEXES YIELD name, state WHERE name IN $names RETURN name, state",
                names=names,
            )
            states = {r["name"]: r["state"] for r in rows}
            if all(states.get(n) == "ONLINE" for n in names):
                return states
            time.sleep(2)
        raise RuntimeError(f"索引未在 5 分钟内 ONLINE：{states}")

    # -------------------------------------------------------------------- 检索

    def search(self, question: str, k: int | None = None) -> dict:
        k = int(k or self.ncfg.get("top_k", 30))
        vector = self._vector_search(question, int(self.ncfg.get("vector_k", 40)))
        fulltext = self._fulltext_search(question, int(self.ncfg.get("fulltext_k", 40)))
        graph, graph_meta = self._graph_search(question, int(self.ncfg.get("graph_k", 40)))
        fused, scores, ranks = rrf_fuse(
            {"vector": vector, "fulltext": fulltext, "graph": graph},
            k=int(self.ncfg.get("rrf_k", RRF_K)),
            limit=k,
        )
        meta = self._page_meta(fused)
        return {
            "question": question,
            "terms": _tokens(question),
            "channels": {
                "vector": vector,
                "fulltext": fulltext,
                "graph": graph,
            },
            "fused": fused,
            "scores": {pid: round(scores.get(pid, 0.0), 6) for pid in fused},
            "ranks": {pid: ranks.get(pid, {}) for pid in fused},
            "pages": meta,
            "graph_entities": graph_meta,
        }

    def _vector_search(self, question: str, k: int) -> list[str]:
        vec = self.embedder.encode([question])[0]
        rows = self._run(
            "CALL db.index.vector.queryNodes('tw_page_vec', $k, $vec) "
            "YIELD node, score RETURN node.id AS id, score ORDER BY score DESC",
            k=int(k),
            vec=vec,
        )
        return [r["id"] for r in rows]

    def _fulltext_search(self, question: str, k: int) -> list[str]:
        query = lucene_query(question)
        if not query:
            return []
        rows = self._run(
            "CALL db.index.fulltext.queryNodes('tw_page_ft', $q, {limit: $k}) "
            "YIELD node, score RETURN node.id AS id, score ORDER BY score DESC",
            q=query,
            k=int(k),
        )
        return [r["id"] for r in rows]

    def _anchor_weights(self, question: str, entity_k: int) -> tuple[dict[str, float], list[str]]:
        exact: list[str] = []
        if self.index is not None:
            for term in question_anchors(question, self.index):
                entity = self.index.resolve(term)
                if entity is not None and entity.id not in exact:
                    exact.append(entity.id)
        weights: dict[str, float] = {}
        if exact:
            # 只保留真正入库的实体；没进图（df 超限）的泛实体不能当锚点
            rows = self._run(
                f"MATCH (e:{ENTITY}) WHERE e.id IN $ids RETURN e.id AS id", ids=exact
            )
            weights = {r["id"]: 1.0 for r in rows}
        query = lucene_query(question)
        if query and not weights:
            rows = self._run(
                "CALL db.index.fulltext.queryNodes('tw_entity_ft', $q, {limit: $k}) "
                "YIELD node, score RETURN node.id AS id, score ORDER BY score DESC",
                q=query,
                k=int(entity_k),
            )
            for row in rows:
                weights.setdefault(row["id"], 0.5)
        return weights, exact

    def _graph_search(self, question: str, k: int) -> tuple[list[str], list[str]]:
        weights, exact = self._anchor_weights(
            question, int(self.ncfg.get("graph_entity_k", 12))
        )
        if not weights:
            return [], []
        anchor_rows = self._run(
            "UNWIND $rows AS row "
            f"MATCH (e:{ENTITY} {{id: row.id}})-[m:MENTIONS]->(p:{PAGE}) "
            "WITH p, sum(row.w * coalesce(e.idf, 1.0) * log(1.0 + coalesce(m.count, 1))) AS s "
            "RETURN p.id AS id, s ORDER BY s DESC LIMIT $k",
            rows=[{"id": eid, "w": w} for eid, w in weights.items()],
            k=int(k),
        )
        anchor_ids = [r["id"] for r in anchor_rows]
        if not anchor_ids:
            return [], list(weights)
        anchor_score = {r["id"]: float(r["s"] or 0.0) for r in anchor_rows}

        # 第二跳：锚页上共现的其它实体，再回到这些实体出现的页
        neighbor_rows = self._run(
            "UNWIND $pids AS pid "
            f"MATCH (p:{PAGE} {{id: pid}})-[m:MENTIONS]->(e:{ENTITY}) "
            "WHERE NOT e.id IN $exclude "
            "RETURN e.id AS id, coalesce(e.idf, 1.0) AS idf, coalesce(m.count, 1) AS c",
            pids=anchor_ids[:30],
            exclude=list(weights),
        )
        support: dict[str, int] = defaultdict(int)
        strength: dict[str, float] = defaultdict(float)
        for row in neighbor_rows:
            support[row["id"]] += 1
            strength[row["id"]] += float(row["idf"]) * math.log1p(int(row["c"]))
        ranked = sorted(
            strength,
            key=lambda eid: -(strength[eid] * (1.0 + math.log1p(support[eid]))),
        )[:20]
        if ranked:
            page_rows = self._run(
                "UNWIND $rows AS row "
                f"MATCH (e:{ENTITY} {{id: row.id}})-[m:MENTIONS]->(p:{PAGE}) "
                "WHERE NOT p.id IN $exclude "
                "RETURN p.id AS id, sum(row.w * log(1.0 + coalesce(m.count, 1))) AS s "
                "ORDER BY s DESC LIMIT $k",
                rows=[{"id": eid, "w": strength[eid]} for eid in ranked],
                exclude=anchor_ids,
                k=int(k),
            )
            for row in page_rows:
                anchor_score[row["id"]] = anchor_score.get(row["id"], 0.0) + float(
                    row["s"] or 0.0
                )
        ordered = sorted(anchor_score, key=lambda pid: (-anchor_score[pid], pid))
        return ordered[: int(k)], list(weights)

    def _page_meta(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        rows = self._run(
            f"MATCH (p:{PAGE}) WHERE p.id IN $ids "
            "RETURN p.id AS id, p.doc_id AS doc_id, p.page AS page, "
            "substring(p.text, 0, 180) AS preview",
            ids=ids,
        )
        by_id = {r["id"]: r for r in rows}
        return [by_id.get(pid, {"id": pid}) for pid in ids]

    def stats(self) -> dict:
        rows = self._run(
            f"MATCH (p:{PAGE}) WITH count(p) AS pages "
            f"MATCH (e:{ENTITY}) WITH pages, count(e) AS entities "
            "MATCH ()-[m:MENTIONS]->() RETURN pages, entities, count(m) AS mentions"
        )
        return dict(rows[0]) if rows else {}
