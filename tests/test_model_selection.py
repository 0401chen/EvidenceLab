"""No real model network: validate server allowlist and captured transport JSON."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import providers
from backend.app.config import Config
from backend.app.errors import ProviderError


@pytest.fixture
def model_client(client, monkeypatch):
    # Only mutate this test's isolated Config; never read or edit .env.
    cfg = client.app.state.config
    object.__setattr__(cfg, "api_key", "synthetic-test-key")
    object.__setattr__(cfg, "model", "gpt-4o-mini")
    object.__setattr__(cfg, "allowed_models", ("gpt-4.1-mini", "gpt-5-mini"))
    async def forbidden(*args, **kwargs):
        pytest.fail("Unmocked model networking is forbidden")
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    pid = client.post("/api/papers/upload", files={"files": ("source.txt", b"# Synthetic\n\nAccuracy was 81.5%.", "text/plain")}).json()["papers"][0]["id"]
    return client, pid


def test_allowlist_and_public_catalog_never_claim_all_providers_support_presets(model_client):
    client, _ = model_client
    settings = client.get("/api/settings").json()
    options = {m["id"]: m for m in settings["models"]}
    assert options["gpt-4.1-mini"]["enabled"]
    assert not options["deepseek-flash"]["enabled"]
    assert all(m["availability"] == "untested" and "服务商报价" in m["price_notice"] for m in options.values())
    assert "synthetic-test-key" not in json.dumps(settings)


@pytest.mark.parametrize("path,fields", [("qa", {"question": "accuracy?", "engine": "llm"}),
    ("verify", {"claim": "accuracy?", "engine": "llm"}), ("extract", {"mode": "llm"})])
@pytest.mark.parametrize("model", ["unapproved-model", "https://evil.invalid?key=private"])
def test_unapproved_or_invalid_model_rejected_before_provider(model_client, path, fields, model):
    client, pid = model_client
    url = f"/api/papers/{pid}/extract" if path == "extract" else f"/api/{path}"
    assert client.post(url, json={"paper_ids": [pid], **fields, "model_id": model}).status_code == 422


def test_qa_and_independent_verification_use_identical_selected_model(model_client, monkeypatch):
    client, pid = model_client
    sources = client.get(f"/api/papers/{pid}").json()["evidence"]
    seen = []
    async def answer(question, evidence, config):
        seen.append(("answer", config["model"]))
        return {"mode": "llm", "answer": "Synthetic answer", "verdict": "evidence_found", "citations": sources}
    async def verify(claim, evidence, config):
        seen.append(("verify", config["model"]))
        return {"mode": "llm", "verdict": "supported", "explanation": "Synthetic", "citations": sources}
    monkeypatch.setattr(providers, "answer_llm", answer)
    monkeypatch.setattr(providers, "verify_claim", verify)
    first = client.post("/api/qa", json={"paper_ids": [pid], "question": "accuracy?", "engine": "llm", "model_id": "gpt-4.1-mini"})
    assert first.status_code == 200
    snapshot = first.json()
    assert seen == [("answer", "gpt-4.1-mini"), ("verify", "gpt-4.1-mini")]
    assert snapshot["model_id"] == snapshot["verification"]["model_id"] == "gpt-4.1-mini"
    second = client.post("/api/qa", json={"paper_ids": [pid], "question": "accuracy?", "engine": "llm"})
    assert second.status_code == 200 and seen[-1][1] == "gpt-4o-mini"
    assert snapshot["model_id"] == "gpt-4.1-mini"  # No mutable global model selection.


def test_candidate_selection_records_model_and_never_changes_existing_experiments(model_client, monkeypatch):
    from backend.app.analysis import extract_experiments
    client, pid = model_client
    before = client.get(f"/api/papers/{pid}").json()
    async def extract(evidence, paper_id, config):
        assert config["model"] == "gpt-4.1-mini"
        return extract_experiments(evidence, paper_id)
    monkeypatch.setattr(providers, "extract_llm", extract)
    result = client.post(f"/api/papers/{pid}/extract", json={"mode": "llm", "model_id": "gpt-4.1-mini"})
    assert result.status_code == 200
    data = result.json()
    assert data["experiments"] == before["experiments"]
    assert data["extraction_candidates"][0]["experiments"][0]["model_id"] == "gpt-4.1-mini"


def test_local_mode_cannot_activate_model_even_with_model_id(model_client):
    client, pid = model_client
    for endpoint, field in [("qa", "question"), ("verify", "claim")]:
        r = client.post(f"/api/{endpoint}", json={"paper_ids": [pid], field: "accuracy?", "engine": "local", "model_id": "gpt-5-mini"})
        assert r.status_code == 200
        assert not r.json().get("model_runs")


def test_switching_models_does_not_mutate_saved_report_snapshots(model_client, monkeypatch):
    client, pid = model_client
    ids = [p["id"] for p in client.post("/api/demo/seed").json()["papers"]]
    report = client.post("/api/compare", json={"paper_ids": ids}).json()
    url = f"/api/reports/{report['id']}"
    before = client.get(url).content
    sources = client.get(f"/api/papers/{pid}").json()["evidence"]
    async def verify(claim, evidence, config):
        return {"mode": "llm", "verdict": "insufficient", "citations": sources, "explanation": "Synthetic"}
    monkeypatch.setattr(providers, "verify_claim", verify)
    for model in ["gpt-4.1-mini", "gpt-5-mini"]:
        result = client.post("/api/verify", json={"paper_ids": [pid], "claim": "accuracy?", "engine": "llm", "model_id": model})
        assert result.status_code == 200 and result.json()["model_id"] == model
    assert client.get(url).content == before


def test_wrong_response_model_is_rejected_without_fallback(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"model": "unexpected-expensive-model",
            "choices": [{"message": {"content": '{"ok":true}'}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ProviderError) as error:
        asyncio.run(providers._chat_json([], {"model": "gpt-4.1-mini", "base_url": "https://mock.invalid/v1"}))
    assert error.value.code == "model_mismatch"


@pytest.mark.parametrize("model,token_key,temperature,json_mode", [
    ("gpt-4o-mini", "max_tokens", True, True), ("gpt-4.1-mini", "max_tokens", True, True),
    ("gpt-4.1-nano", "max_tokens", True, True), ("gpt-5-mini", "max_completion_tokens", False, True),
    ("deepseek-flash", "max_tokens", False, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "max_tokens", False, False)])
def test_actual_transport_payload_and_safe_model_usage_record(monkeypatch, model, token_key, temperature, json_mode):
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"model": model, "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    runs = []
    assert asyncio.run(providers._chat_json([{"role": "user", "content": "synthetic"}],
        {"model": model, "api_key": "synthetic-test-key", "base_url": "https://mock.invalid/v1", "_model_runs": runs})) == {"ok": True}
    assert len(calls) == 1 and calls[0]["model"] == model
    assert token_key in calls[0] and ("temperature" in calls[0]) == temperature
    assert ("response_format" in calls[0]) == json_mode
    assert runs[0]["response_model"] == model and runs[0]["usage"]["prompt_tokens"] == 9
    assert "synthetic-test-key" not in json.dumps(runs)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permission_and_parameter_errors_do_not_retry_or_switch_model(monkeypatch, status):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(status, json={"error": "unsupported model or temperature"})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ProviderError):
        asyncio.run(providers._chat_json([], {"model": "gpt-5-mini", "base_url": "https://mock.invalid/v1"}))
    assert calls == ["gpt-5-mini"]


def test_json_compatibility_retry_is_bounded_and_keeps_model(monkeypatch):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": "response_format not supported"})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ProviderError):
        asyncio.run(providers._chat_json([], {"model": "gpt-4.1-mini", "base_url": "https://mock.invalid/v1"}))
    assert len(calls) == 2 and all(p["model"] == "gpt-4.1-mini" for p in calls)
    assert "response_format" not in calls[1]


def test_offline_launcher_blocks_transport_even_after_explicit_model_selection(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = r'''
import asyncio, runpy, sys, tempfile
from pathlib import Path
import httpx, uvicorn
root=Path.cwd()
with tempfile.TemporaryDirectory(dir=root/'artifacts', prefix='offline-model-test-') as directory:
    uvicorn.run=lambda *a, **k: None
    sys.argv=['serve_offline.py', '--data-dir', directory, '--port', '8790']
    runpy.run_path(str(root/'scripts/serve_offline.py'), run_name='__main__')
    from backend.app import providers
    from backend.app.errors import ProviderError
    async def check():
        try:
            await providers._chat_json([], {'model':'gpt-5-mini','api_key':'synthetic'})
        except ProviderError as exc:
            assert exc.code=='offline_review'
        else: raise AssertionError('Offline model seam bypassed')
        async with httpx.AsyncClient() as client:
            try: await client.get('https://must-not-connect.invalid')
            except RuntimeError as exc: assert 'forbids model networking' in str(exc)
            else: raise AssertionError('Offline HTTP transport bypassed')
    asyncio.run(check())
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, "Isolated offline-launcher contract failed"
