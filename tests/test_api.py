from __future__ import annotations

import time

from fastapi.testclient import TestClient

from treewiki.core.store import KB
from treewiki.core.wiki import WikiPage, WikiStore
from treewiki.web import api


def test_ingest_form_description(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    home.mkdir()
    monkeypatch.setattr(api, "kb_home", lambda: home)
    captured = {}

    def fake_ingest(path, kb, cfg, stage, description=""):
        captured["description"] = description
        captured["filename"] = path.name
        stage("done")
        return "src_0001"

    monkeypatch.setattr(api, "ingest_file", fake_ingest)
    client = TestClient(api.app)

    r = client.post(
        "/api/ingest",
        files={"file": ("a.md", b"# A", "text/markdown")},
        data={"description": "v1.0 \u5e73\u53f0\u90e8\u7f72\u6587\u6863"},
    )

    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = api.JOBS[job_id]
        if job["status"] != "running":
            break
        time.sleep(0.02)
    api.JOBS.pop(job_id, None)
    assert job["status"] == "done"
    assert captured["description"] == "v1.0 \u5e73\u53f0\u90e8\u7f72\u6587\u6863"
    assert captured["filename"] == "a.md"


def test_wiki_overview_and_page(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    (home / "wiki" / "entities").mkdir(parents=True)
    (home / "wiki" / "entities" / "ent_demo.md").write_text(
        "---\n"
        "id: ent_demo\n"
        "title: \u793a\u4f8b\u5b9e\u4f53\n"
        "type: entity\n"
        "tags: [\u6d4b\u8bd5]\n"
        "sources: [src_0001]\n"
        "citations:\n"
        "  - doc: src_0001\n"
        "    pages: [2, 3]\n"
        "    claim: \u793a\u4f8b\u65ad\u8a00\n"
        "updated: 2026-10-20\n"
        "stale: false\n"
        "---\n"
        "\u6b63\u6587\u5f15\u7528 [[topic_demo]]\u3002\n",
        encoding="utf-8",
    )
    (home / "wiki" / "index.md").write_text("# \u76ee\u5f55\n", encoding="utf-8")
    monkeypatch.setattr(api, "kb_home", lambda: home)
    client = TestClient(api.app)

    overview = client.get("/api/wiki/overview").json()
    assert overview["status"] == "ready"
    assert overview["counts"]["pages"] == 1
    assert overview["pages"][0]["id"] == "ent_demo"
    assert overview["pages"][0]["links"] == ["topic_demo"]
    assert overview["special"]["index.md"]["exists"] is True

    page = client.get(
        "/api/wiki/page", params={"path": "entities/ent_demo.md"}
    ).json()
    assert page["meta"]["title"] == "\u793a\u4f8b\u5b9e\u4f53"
    assert page["links"] == ["topic_demo"]
    assert "[[topic_demo]]" in page["body"]


def test_wiki_page_rejects_path_traversal(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    (home / "wiki").mkdir(parents=True)
    (home / "secret.md").write_text("# secret", encoding="utf-8")
    monkeypatch.setattr(api, "kb_home", lambda: home)
    client = TestClient(api.app)

    r = client.get("/api/wiki/page", params={"path": "../secret.md"})
    assert r.status_code == 404


def test_pages_endpoint_caps_preview(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    kb = KB(home)
    kb.write_pages("src_0001", [f"p{i}" for i in range(1, 8)])
    kb.register("src_0001", home / "raw" / "a.md", "sha", 7)
    monkeypatch.setattr(api, "kb_home", lambda: home)
    client = TestClient(api.app)

    data = client.get(
        "/api/pages/src_0001", params={"start": 1, "end": 99}
    ).json()
    assert [p["page"] for p in data["pages"]] == [1, 2, 3, 4, 5]
    assert data["pages"][-1]["text"] == "p5"
    assert data["has_more"] is True
    assert client.get("/api/pages/src_9999").status_code == 404
def test_generated_wiki_visible_in_overview(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    home.mkdir()
    monkeypatch.setattr(api, "kb_home", lambda: home)
    kb = KB(home)
    WikiStore(kb).save(
        WikiPage(
            id="ent_acme",
            title="Acme",
            type="entity",
            description="一家公司",
            sources=["src_0001"],
            citations=[{"doc": "src_0001", "pages": [1], "claim": "收入 100 亿"}],
        )
    )
    client = TestClient(api.app)

    overview = client.get("/api/wiki/overview").json()
    assert overview["status"] == "ready"
    page = [p for p in overview["pages"] if p["id"] == "ent_acme"][0]
    assert page["type"] == "entity" and page["citations"] == 1
    detail = client.get(
        "/api/wiki/page", params={"path": page["path"]}
    ).json()
    assert detail["meta"]["title"] == "Acme"
    assert "关键事实" in detail["body"]


def test_query_dispatch_modes(tmp_path, monkeypatch):
    home = tmp_path / "kb"
    home.mkdir()
    monkeypatch.setattr(api, "kb_home", lambda: home)
    client = TestClient(api.app)

    assert client.post("/api/distill/src_9999").status_code == 404

    captured = {}

    def fake_query(question, kb, llm, cfg, mode=None, doc_id=None):
        captured.update(question=question, mode=mode, doc_id=doc_id)
        return {"answer": "ok", "citations": [], "usage": {}}

    monkeypatch.setattr(api, "run_query", fake_query)
    monkeypatch.setattr(api, "LLM", lambda cfg: object())

    r = client.post("/api/query", json={"question": "收入?"})
    assert r.status_code == 200 and captured == {"question": "收入?", "mode": None, "doc_id": None}
    r = client.post("/api/query", json={"question": "收入?", "doc_id": "src_0001", "mode": "raw"})
    assert r.status_code == 200
    assert captured == {"question": "收入?", "mode": "raw", "doc_id": "src_0001"}
