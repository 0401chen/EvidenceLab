"""Supported public modes and non-destructive demo startup."""
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.mark.parametrize("endpoint,field", [("qa", "question"), ("verify", "claim")])
def test_removed_engine_is_rejected_without_provider_call(client, monkeypatch, endpoint, field):
    from backend.app import providers

    async def forbidden(*args, **kwargs):
        pytest.fail("A retired engine must never invoke a provider")

    monkeypatch.setattr(providers, "answer_llm", forbidden)
    monkeypatch.setattr(providers, "verify_claim", forbidden)
    response = client.post(f"/api/{endpoint}", json={field: "A claim", "engine": "llm_tasks", "paper_ids": []})
    assert response.status_code == 422


def test_normal_qa_still_performs_second_independent_verification(client, monkeypatch):
    from backend.app import providers

    paper = client.post("/api/papers/upload", files={"files": ("source.md", b"Atlas-A uses Adam.", "text/markdown")}).json()["papers"][0]
    source = client.get(f"/api/papers/{paper['id']}").json()["evidence"][0]
    client.app.state.config.__dict__.update(api_key="offline", model="offline")
    calls = []

    async def answer(question, evidence, config):
        calls.append(("answer", evidence))
        return {"answer": "Atlas-A uses Adam.", "mode": "llm", "verdict": "unverified",
                "citations": [source], "traces": [], "warnings": []}

    async def verify(claim, evidence, config):
        calls.append(("verify", evidence))
        assert claim == "Atlas-A uses Adam."
        return {"verdict": "supported", "explanation": "Direct source", "mode": "llm",
                "citations": [source], "traces": [], "warnings": []}

    monkeypatch.setattr(providers, "answer_llm", answer)
    monkeypatch.setattr(providers, "verify_claim", verify)
    response = client.post("/api/qa", json={"question": "Which optimizer?", "paper_ids": [paper["id"]], "engine": "llm"})
    assert response.status_code == 200
    assert response.json()["verification"]["verdict"] == "supported"
    assert response.json()["verdict"] == "evidence_found"
    assert [name for name, _ in calls] == ["answer", "verify"]
    assert calls[0][1] == calls[1][1]


def test_demo_start_requires_existing_nonempty_library_and_does_not_copy_secrets(client, tmp_path, monkeypatch):
    from scripts.serve_demo_ai import prepare

    with pytest.raises(ValueError):
        prepare(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    client.post("/api/papers/upload", files={"files": ("source.md", b"An original synthetic source.", "text/markdown")})
    config = tmp_path / "private.env"
    config.write_text("LLM_API_KEY=private-test-key\nLLM_BASE_URL=https://example.invalid/v1\nLLM_MODEL=demo-model\nEVIDENCELAB_DATA_DIR=forbidden\n")
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT", "EVIDENCELAB_DATA_DIR", "EVIDENCE_PARSER"):
        monkeypatch.setenv(name, "")
    import os
    data = client.app.state.config.data_dir
    assert prepare(data, config) == []
    assert os.environ["EVIDENCELAB_DATA_DIR"] == str(data.resolve())
    assert os.environ["LLM_API_KEY"] == "private-test-key"
    assert not (data / ".env").exists()


def test_optional_usage_capture_never_imposes_a_request_quota(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import demo_model_audit
    from backend.app import providers

    monkeypatch.setattr(demo_model_audit, "__file__", str(tmp_path / "scripts/demo_model_audit.py"))
    output = tmp_path / "artifacts/capture"

    async def fake(client, request, **kwargs):
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 2},
                                        "choices": [{"message": {"content": "private-test-key"}}]}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake)
    monkeypatch.setattr(providers, "answer_llm", providers.answer_llm)
    monkeypatch.setattr(providers, "verify_claim", providers.verify_claim)
    from fastapi import FastAPI
    app = FastAPI()
    app.state.config = SimpleNamespace(api_key="private-test-key")
    demo_model_audit.install(app, output)

    async def run():
        async with httpx.AsyncClient() as client:
            for _ in range(8):
                response = await client.post("https://example.invalid/chat/completions",
                                             headers={"Authorization": "Bearer private-test-key"},
                                             json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]})
                assert response.status_code == 200
    asyncio.run(run())
    assert json.loads((output / "model-usage.json").read_text())["http_requests"] == 8
    text = "".join(p.read_text() for p in output.glob("*.json"))
    assert "Authorization" not in text and "private-test-key" not in text
