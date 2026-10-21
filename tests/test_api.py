from __future__ import annotations

import time

from fastapi.testclient import TestClient

from treewiki.core.store import KB
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
