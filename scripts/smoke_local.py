r"""Start a real temporary Uvicorn app and exercise the offline demo pipeline.

Run from the project root: .venv\Scripts\python.exe scripts/smoke_local.py
Uses only synthetic input, a temporary database, and empty model settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import httpx
    import uvicorn

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=artifacts) as directory:
        temporary = Path(directory).resolve()
        assert temporary.is_relative_to(artifacts.resolve())
        os.environ.update(EVIDENCELAB_DATA_DIR=str(temporary / "data"), LLM_API_KEY="", OPENAI_API_KEY="",
                          LLM_MODEL="", LLM_BASE_URL="http://127.0.0.1:9/v1", LLM_TIMEOUT="5", EVIDENCE_PARSER="pymupdf")
        from backend.app.main import create_app

        app = create_app(temporary / "data")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
            worker = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
            worker.start()
            try:
                deadline = time.monotonic() + 15
                while not server.started and worker.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert server.started, "Temporary Uvicorn app did not start"
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False) as client:
                    assert client.get("/api/health").status_code == 200
                    assert client.get("/").status_code == 200
                    assert client.get("/api/settings").json()["llm_configured"] is False
                    upload = client.post("/api/papers/upload", files={"files": ("smoke.md", b"# Synthetic smoke paper\n\nA separate synthetic source contains no measured result.", "text/markdown")})
                    assert not upload.json()["errors"]
                    uploaded_id = upload.json()["papers"][0]["id"]
                    assert client.get(f"/api/papers/{uploaded_id}/file").status_code == 200
                    manual_upload = client.post("/api/papers/upload", files={"files": ("manual.md",
                        b"# Manual evidence\n\nDelta achieves accuracy 71.2% on Bench test set.", "text/markdown")})
                    manual_id = manual_upload.json()["papers"][0]["id"]
                    before_manual = client.get(f"/api/papers/{manual_id}").json()
                    source = next(e for e in before_manual["evidence"] if "71.2" in e["text"])
                    body = {"name": "User established experiment", "fields": {"method": "Delta", "dataset": "Bench",
                            "split": "test", "metric": "accuracy", "value": 71.2, "unit": "%"},
                            "evidence_quotes": {k: [{"id": source["id"], "quote": source["text"]}]
                                                for k in ("method", "dataset", "split", "metric", "value", "unit")}}
                    bad = {**body, "fields": {**body["fields"], "value": 99.9}}
                    assert client.post(f"/api/papers/{manual_id}/manual-candidates", json=bad).status_code == 422
                    assert client.get(f"/api/papers/{manual_id}").json() == before_manual
                    created = client.post(f"/api/papers/{manual_id}/manual-candidates", json=body)
                    assert created.status_code == 200 and created.json()["experiments"] == before_manual["experiments"]
                    pending = created.json()["extraction_candidates"][0]
                    assert pending["mode"] == "manual" and pending["status"] == "pending"
                    reviewed_manual = client.post(f"/api/papers/{manual_id}/extraction-candidates/{pending['id']}/accept", json={})
                    assert reviewed_manual.status_code == 200
                    assert any(e["extraction_method"] == "manual" for e in reviewed_manual.json()["experiments"])
                    assert client.post("/api/qa", json={"question": "accuracy?", "engine": "local", "paper_ids": []}).status_code == 400
                    papers = client.post("/api/demo/seed").json()["papers"]
                    ids = [p["id"] for p in papers]
                    answer = client.post("/api/qa", json={"question": "accuracy?", "engine": "local", "paper_ids": ids[:1]}).json()
                    assert answer["citations"] and all(e["paper_id"] == ids[0] for e in answer["citations"])
                    response = client.post("/api/compare", json={"paper_ids": ids})
                    assert response.status_code == 200
                    report = response.json()
                    assert len(report["comparisons"]) == 3 and len(report["traces"]) == 4
                    assert any(c["comparability"] == "direct" for c in report["comparisons"])
                    detail = client.get(f"/api/papers/{ids[0]}").json()
                    generated = client.post(f"/api/papers/{ids[0]}/extract", json={"mode": "rule"})
                    assert generated.status_code == 200
                    reviewed = generated.json()
                    assert reviewed["experiments"] == detail["experiments"]
                    candidate = next(item for item in reviewed["extraction_candidates"]
                                     if item["id"] == reviewed["latest_extraction_candidate_id"])
                    assert candidate["status"] == "pending"
                    acceptance = f"/api/papers/{ids[0]}/extraction-candidates/{candidate['id']}/accept"
                    accepted = client.post(acceptance, json={})
                    assert accepted.status_code == 200
                    assert accepted.json()["experiments"] == detail["experiments"]
                    assert client.post(acceptance, json={}).json() == accepted.json()
                    for format in ("json", "markdown"):
                        assert client.get(f"/api/reports/{report['id']}/export?format={format}").status_code == 200
                    verified = client.post("/api/verify", json={"claim": "GraphLite accuracy is 91.2%", "paper_ids": ids[:1], "engine": "local"}).json()
                    assert verified["verdict"] == "insufficient" and verified["mode"] == "local"
                    experiment = client.get(f"/api/papers/{ids[0]}").json()["experiments"][0]
                    assert client.patch(f"/api/experiments/{experiment['id']}", json={"value": 0}).status_code == 200
                    assert client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 404
                    history = client.get(f"/api/experiments/{experiment['id']}/history").json()
                    assert history and history[0]["experiment"] == experiment
                    reports = client.get("/api/reports").json()
                    assert any(item["id"] == report["id"] and item["stale"] for item in reports)
                    archived = client.get(f"/api/reports/{report['id']}").json()
                    assert archived["stale"] and archived["experiments"] == report["experiments"]
                    stale_export = client.get(f"/api/reports/{report['id']}/export?include_stale=true")
                    assert stale_export.status_code == 200 and "历史快照" in stale_export.text
                    assert client.delete(f"/api/papers/{uploaded_id}").status_code == 200
                    assert client.get(f"/api/papers/{uploaded_id}/file").status_code == 404
            finally:
                server.should_exit = True
                worker.join(timeout=15)
                assert not worker.is_alive(), "Temporary test server did not stop"
    assert not temporary.exists()
    print(json.dumps({"status": "passed", "checks": ["uvicorn_startup", "frontend", "upload_and_file", "explicit_scope",
          "local_qa", "comparison_workflow", "json_and_markdown_export", "offline_verification", "manual_edit_invalidation", "delete_cleanup",
          "extraction_candidate_preserves_current", "candidate_acceptance_deduplication", "candidate_acceptance_idempotency",
          "manual_edit_history", "report_history_recovery", "explicit_stale_snapshot_export",
          "manual_candidate_rejects_ungrounded_value", "manual_candidate_preserves_current", "manual_candidate_browser_review_boundary"],
          "model_requests": 0, "data": "temporary_synthetic", "temporary_data_removed": True}))


if __name__ == "__main__":
    main()
