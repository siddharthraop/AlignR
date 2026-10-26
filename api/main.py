"""AlignR API: draft a held-out generation's topic set on request, stream the stages, and keep the run log.

    POST /api/draft            {"generation": "tpu7x"} -> NDJSON stream of stage events, then the run
    GET  /api/runs/latest      the most recent web run
    GET  /api/runs/{id}        one run, with every draft beside the published topic
    GET  /api/runs/{id}/zip    the drafts and the reports as a .zip
    GET  /api/scorecard        the success metrics, measured over the run log
    GET  /api/health
"""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from alignr import __version__
from alignr.agent import draft_set, report_markdown, slim, zip_bytes
from alignr.corpus import ROOT, corpus_dir
from alignr.model import has_key, make_model

FACTS_DIR = Path(os.environ.get("ALIGNR_FACTS", ROOT / "facts"))
DAILY = int(os.environ.get("ALIGNR_DAILY_DRAFTS", "20"))   # each draft of a set is about 5 to 15 Gemini calls
origins = os.environ.get("ALIGNR_CORS_ORIGINS", "https://siddharthraop.github.io,http://localhost:8000").split(",")

app = FastAPI(title="AlignR", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class Store:
    """Run log and daily quota: Firestore on Cloud Run, memory locally and in tests."""

    def __init__(self):
        self.mem: dict[str, dict] = {}
        self.count: dict[str, int] = {}
        self.lock = threading.Lock()
        self.db = None
        if os.environ.get("ALIGNR_FIRESTORE") == "1":
            try:
                from google.cloud import firestore
                self.db = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
                self.fs = firestore
            except Exception as e:  # keep serving without the log rather than fail every request
                print(f"Firestore unavailable: {e}")
        self.backend = "firestore" if self.db else "memory"

    def take(self) -> int:
        """Count one draft against today's cap. Returns the count after this draft."""
        day = dt.date.today().isoformat()
        if self.db:
            ref = self.db.collection("alignr_quota").document(day)
            ref.set({"count": self.fs.Increment(1)}, merge=True)
            return int(ref.get().to_dict().get("count", 1))
        with self.lock:
            self.count[day] = self.count.get(day, 0) + 1
            return self.count[day]

    def used(self) -> int:
        day = dt.date.today().isoformat()
        if self.db:
            d = self.db.collection("alignr_quota").document(day).get()
            return int((d.to_dict() or {}).get("count", 0)) if d.exists else 0
        return self.count.get(day, 0)

    def save(self, run: dict):
        self.mem[run["run"]] = run
        if self.db:
            self.db.collection("alignr_runs").document(run["run"]).set(
                {"run": run["run"], "started": run["started"], "payload": json.dumps(run)})

    def get(self, rid: str) -> dict | None:
        if rid in self.mem:
            return self.mem[rid]
        if self.db:
            d = self.db.collection("alignr_runs").document(rid).get()
            if d.exists:
                run = json.loads(d.to_dict()["payload"])
                self.mem[rid] = run
                return run
        return None

    def recent(self, n: int = 50) -> list[dict]:
        if self.db:
            q = self.db.collection("alignr_runs").order_by("started", direction="DESCENDING").limit(n)
            return [json.loads(d.to_dict()["payload"]) for d in q.stream()]
        return sorted(self.mem.values(), key=lambda r: r["started"], reverse=True)[:n]


store = Store()


def generations() -> dict[str, Path]:
    return {p.stem: p for p in sorted(FACTS_DIR.glob("*.yaml"))}


def for_page(run: dict) -> dict:
    """The run as the page needs it: no JSON Schemas, plus each topic's report."""
    out = json.loads(json.dumps(run))
    for t in out["topics"]:
        t.pop("schema", None)
        t["report"] = report_markdown(run, t)
    out["report"] = report_markdown(run)
    return out


class DraftRequest(BaseModel):
    generation: str = Field(default="tpu7x", pattern=r"^[a-z0-9-]{2,20}$")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "model": os.environ.get("ALIGNR_MODEL", "gemini-3.8-flash") if has_key()
            else "baseline (no key)", "store": store.backend, "generations": list(generations()),
            "drafts_today": store.used(), "daily_cap": DAILY, "corpus": os.environ.get("ALIGNR_CORPUS_COMMIT", "local"),
            "grammar": (corpus_dir() / ".grammar").exists() or bool(os.environ.get("ALIGNR_GRAMMAR"))}


@app.post("/api/draft")
def draft(body: DraftRequest):
    gens = generations()
    if body.generation not in gens:
        raise HTTPException(404, f"No launch facts for {body.generation}. Available: {', '.join(gens)}")
    if store.used() >= DAILY:
        latest = store.recent(1)
        raise HTTPException(429, {"message": f"Today's {DAILY} public drafts are used up. Each draft calls Gemini. "
                                             "Here is the latest run instead; try again tomorrow.",
                                  "latest": latest[0]["run"] if latest else None})
    store.take()
    events: queue.Queue = queue.Queue()

    def work():
        try:
            model = make_model("auto")
            run = draft_set(gens[body.generation], model, emit=events.put, source="web")
            run = for_page(run)
            store.save(run)
            events.put({"type": "result", "run": run})
        except Exception as e:
            events.put({"type": "error", "message": str(e)[:500]})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            try:
                e = events.get(timeout=15)
            except queue.Empty:
                yield json.dumps({"type": "ping"}) + "\n"   # keeps proxies from closing a quiet stream
                continue
            if e is None:
                return
            yield json.dumps(e if "type" in e else {"type": "stage", **e}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/runs/latest")
def latest():
    r = store.recent(1)
    if not r:
        raise HTTPException(404, "No runs yet")
    return r[0]


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    r = store.get(rid)
    if not r:
        raise HTTPException(404, "No such run")
    return r


@app.get("/api/runs/{rid}/zip")
def get_zip(rid: str):
    r = store.get(rid)
    if not r:
        raise HTTPException(404, "No such run")
    name = f"alignr-{r['generation']}-{rid}.zip"
    return Response(zip_bytes(r), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/scorecard")
def scorecard():
    """The MVP metrics over the web run log. The Git-path metrics come from runs/latest-github.json."""
    runs = store.recent(50)
    model_runs = [r for r in runs if r.get("model_kind") == "gemini"]
    latest = model_runs[0] if model_runs else None   # the metrics are about model drafts, not the baseline
    return {
        "runs": len(runs), "runs_with_cost": sum(1 for r in runs if r.get("metrics", {}).get("cost_logged")
                                                and "cost_usd" in r.get("usage", {})),
        "grammar_failures_reaching_writer": sum(r["metrics"]["grammar_failures_reaching_writer"] for r in runs),
        "latest": slim(latest) if latest else None,
    }
