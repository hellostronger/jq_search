from __future__ import annotations

import time

from fastapi.testclient import TestClient

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
