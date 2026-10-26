import json

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_draft_stream_run_and_zip(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = client.post("/api/draft", json={"generation": "tpu7x"})
    lines = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert lines[0]["type"] == "stage" and lines[-1]["type"] == "result"
    run = lines[-1]["run"]
    assert run["metrics"]["grammar_failures_reaching_writer"] == 0
    spec = next(t for t in run["topics"] if t["slot"] == "specifications")
    assert spec["compare"]["published_view"] and spec["compare"]["draft_view"] and spec["report"]
    assert client.get(f"/api/runs/{run['run']}").json()["run"] == run["run"]
    z = client.get(f"/api/runs/{run['run']}/zip")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    sc = client.get("/api/scorecard").json()
    assert sc["runs"] >= 1 and sc["runs_with_cost"] == sc["runs"]


def test_unknown_generation_and_health():
    assert client.post("/api/draft", json={"generation": "v9z"}).status_code == 404
    h = client.get("/api/health").json()
    assert h["generations"] == ["tpu7x"]
