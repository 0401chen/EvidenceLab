"""Research workflow regressions. All files are temporary; providers are mocked."""
from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.errors import ProviderError
from test_extraction_api import configured_app


def seeded(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    return papers, client.get(f"/api/papers/{papers[0]['id']}").json()


def candidate_url(paper, candidate):
    return f"/api/papers/{paper['id']}/extraction-candidates/{candidate['id']}"


def test_reextraction_is_a_reviewable_candidate_and_accept_is_idempotent(client):
    papers, before = seeded(client)
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    generated = client.post(f"/api/papers/{before['id']}/extract", json={"mode": "rule"})
    assert generated.status_code == 200, generated.text
    detail = generated.json()
    assert detail["experiments"] == before["experiments"]
    assert detail["experiment_count"] == before["experiment_count"]
    candidate = detail["extraction_candidates"][0]
    assert candidate["status"] == "pending" and not candidate["stale"]
    assert candidate["id"] == detail["latest_extraction_candidate_id"]
    assert not ({e["id"] for e in candidate["experiments"]} & {e["id"] for e in before["experiments"]})
    assert client.get(f"/api/reports/{report['id']}/export").status_code == 200
    accepted = client.post(candidate_url(before, candidate) + "/accept", json={})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["experiments"] == before["experiments"]  # Same measurements deduplicate.
    again = client.post(candidate_url(before, candidate) + "/accept", json={})
    assert again.json() == accepted.json()
    assert client.get(f"/api/reports/{report['id']}/export").status_code == 200


def test_manual_edits_survive_new_candidate_acceptance_and_have_history(client):
    _, before = seeded(client)
    original = before["experiments"][0]
    edited = client.patch(f"/api/experiments/{original['id']}", json={"name": "人工复核后的名称", "value": 0}).json()
    generated = client.post(f"/api/papers/{before['id']}/extract", json={"mode": "rule"}).json()
    assert generated["experiments"][0] == edited
    candidate = generated["extraction_candidates"][0]
    accepted = client.post(candidate_url(before, candidate) + "/accept", json={}).json()
    assert next(e for e in accepted["experiments"] if e["id"] == original["id"]) == edited
    assert accepted["experiment_count"] == before["experiment_count"] + 1
    history = client.get(f"/api/experiments/{original['id']}/history").json()
    assert history[0]["experiment"] == original
    assert history[0]["reason"] == "manual_edit"


def test_pending_candidate_cannot_override_a_later_manual_edit(client):
    _, before = seeded(client)
    candidate = client.post(f"/api/papers/{before['id']}/extract", json={"mode": "rule"}).json()["extraction_candidates"][0]
    original = before["experiments"][0]
    assert client.patch(f"/api/experiments/{original['id']}", json={"value": 0}).status_code == 200
    assert client.get(f"/api/papers/{before['id']}").json()["extraction_candidates"][0]["stale"]
    response = client.post(candidate_url(before, candidate) + "/accept", json={})
    assert response.status_code == 409
    assert "重新提取" in response.json()["detail"]
    assert client.get(f"/api/papers/{before['id']}").json()["experiments"][0]["value"] == 0


def test_edit_during_extraction_makes_candidate_stale(client, monkeypatch):
    from backend.app import analysis
    _, before = seeded(client)
    original = analysis.extract_experiments
    store = client.app.state.store

    def concurrent_edit(*args):
        edited = copy.deepcopy(before["experiments"][0])
        edited["value"] = 0
        store.update_experiment(edited)
        return original(*args)

    monkeypatch.setattr(analysis, "extract_experiments", concurrent_edit)
    generated = client.post(f"/api/papers/{before['id']}/extract", json={"mode": "rule"}).json()
    candidate = generated["extraction_candidates"][0]
    assert candidate["stale"] is True
    assert client.post(candidate_url(before, candidate) + "/accept", json={}).status_code == 409


def test_candidate_selection_and_paper_scope_are_validated(client):
    papers, before = seeded(client)
    candidate = client.post(f"/api/papers/{before['id']}/extract", json={"mode": "rule"}).json()["extraction_candidates"][0]
    for ids in ([], [before["experiments"][0]["id"]], ["foreign"]):
        assert client.post(candidate_url(before, candidate) + "/accept", json={"experiment_ids": ids}).status_code == 409
    wrong = {"id": papers[1]["id"]}
    assert client.post(candidate_url(wrong, candidate) + "/accept", json={}).status_code == 404
    rejected = client.post(candidate_url(before, candidate) + "/reject")
    assert rejected.status_code == 200
    assert rejected.json()["extraction_candidates"][0]["status"] == "rejected"
    assert client.post(candidate_url(before, candidate) + "/accept", json={}).status_code == 409
    assert rejected.json()["experiments"] == before["experiments"]


def test_saved_reports_remain_recoverable_as_explicit_stale_snapshots(client):
    papers, before = seeded(client)
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    store = client.app.state.store
    with store.connect() as db:
        original_payload = db.execute("SELECT payload FROM reports WHERE id=?", (report["id"],)).fetchone()[0]
    client.patch(f"/api/experiments/{before['experiments'][0]['id']}", json={"value": 0})
    assert client.get(f"/api/reports/{report['id']}/export").status_code == 404
    metadata = client.get("/api/reports").json()[0]
    assert metadata["id"] == report["id"] and metadata["stale"] is True
    recovered = client.get(f"/api/reports/{report['id']}").json()
    assert recovered["stale"] is True and recovered["experiments"] == report["experiments"]
    exported = client.get(f"/api/reports/{report['id']}/export?format=json&include_stale=true").json()
    assert exported == recovered
    markdown = client.get(f"/api/reports/{report['id']}/export?include_stale=true")
    assert "历史快照" in markdown.text
    with store.connect() as db:
        assert db.execute("SELECT payload FROM reports WHERE id=?", (report["id"],)).fetchone()[0] == original_payload


def test_local_verification_never_calls_configured_provider(tmp_path, monkeypatch):
    from backend.app import providers
    app = configured_app(tmp_path, monkeypatch)

    async def forbidden(*args):
        raise AssertionError("Offline verification must not call a configured provider")

    monkeypatch.setattr(providers, "_chat_json", forbidden)
    with TestClient(app) as scoped:
        papers, _ = seeded(scoped)
        for extra in ({}, {"engine": "local"}):
            response = scoped.post("/api/verify", json={"paper_ids": [papers[0]["id"]], "claim": "accuracy is 85 percent", **extra})
            assert response.status_code == 200, response.text
            assert response.json()["verdict"] == "insufficient"
        assert scoped.get("/api/health").json()["providers"]["llm_status"] == "untested"


def test_llm_verification_requires_explicit_configuration(client):
    papers, _ = seeded(client)
    response = client.post("/api/verify", json={"paper_ids": [papers[0]["id"]], "claim": "accuracy is 85 percent", "engine": "llm"})
    assert response.status_code == 503
    assert "模型尚未配置" in response.json()["detail"]


def test_service_status_tracks_failures_without_probing_or_leaking_credentials(tmp_path, monkeypatch):
    from backend.app import providers
    app = configured_app(tmp_path, monkeypatch)
    calls = []

    async def failure(*args):
        calls.append(1)
        raise ProviderError("connection", "无法连接模型服务，请检查 API 地址与网络。")

    monkeypatch.setattr(providers, "answer_llm", failure)
    with TestClient(app) as scoped:
        assert scoped.get("/api/health").json()["providers"]["llm_status"] == "untested"
        assert set(scoped.get("/api/settings").json()["provider_status"]) == {"chat"}
        assert calls == []
        papers, _ = seeded(scoped)
        response = scoped.post("/api/qa", json={"question": "accuracy", "engine": "llm", "paper_ids": [papers[0]["id"]]})
        assert response.status_code == 502
        assert response.headers["X-EvidenceLab-Error-Code"] == "connection"
        assert scoped.get("/api/health").json()["providers"]["llm_status"] == "failed"
        settings = scoped.get("/api/settings")
        assert "test-placeholder" not in settings.text
        assert settings.json()["provider_status"]["chat"]["status"] == "failed"
        assert len(calls) == 1


def test_answer_survives_failure_of_its_separate_semantic_check(tmp_path, monkeypatch):
    from backend.app import providers
    app = configured_app(tmp_path, monkeypatch)
    with TestClient(app) as scoped:
        _, paper = seeded(scoped)

        async def answer(*args):
            return {"answer": "A source-grounded answer.", "mode": "llm", "citations": paper["evidence"][:1],
                    "warnings": [], "traces": [], "verdict": "evidence_found"}

        async def failure(*args):
            raise ProviderError("connection", "无法连接模型服务，请检查 API 地址与网络。")

        monkeypatch.setattr(providers, "answer_llm", answer)
        monkeypatch.setattr(providers, "verify_claim", failure)
        result = scoped.post("/api/qa", json={"question": "accuracy", "engine": "llm", "paper_ids": [paper["id"]]})
        assert result.status_code == 200, result.text
        assert result.json()["verdict"] == "unverified"
        assert result.json()["citations"] == paper["evidence"][:1]
        assert result.json()["verification"]["status"] == "failed"
        assert any("独立核验未完成" in warning for warning in result.json()["warnings"])


def test_duplicate_inflight_model_requests_are_rejected_and_can_retry(tmp_path, monkeypatch):
    from backend.app import providers
    app = configured_app(tmp_path, monkeypatch)
    with TestClient(app) as scoped:
        _, paper = seeded(scoped)

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def answer(*args):
            calls.append(1)
            entered.set()
            await release.wait()
            return {"answer": "证据不足", "mode": "llm", "citations": [], "warnings": [], "traces": [], "verdict": "insufficient"}

        monkeypatch.setattr(providers, "answer_llm", answer)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://offline") as async_client:
            payload = {"question": "accuracy", "engine": "llm", "paper_ids": [paper["id"]]}
            first = asyncio.create_task(async_client.post("/api/qa", json=payload))
            await entered.wait()
            duplicate = await async_client.post("/api/qa", json=payload)
            assert duplicate.status_code == 409
            assert duplicate.headers["X-EvidenceLab-Error-Code"] == "request_in_progress"
            release.set()
            assert (await first).status_code == 200
            assert (await async_client.post("/api/qa", json=payload)).status_code == 200
            assert len(calls) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("timeout", ["invalid", "nan", "inf", "-inf"])
def test_invalid_timeout_configuration_does_not_break_startup(tmp_path, monkeypatch, timeout):
    from backend.app.config import Config
    monkeypatch.setenv("LLM_TIMEOUT", timeout)
    assert Config.from_env(tmp_path).timeout == 90


def test_experiment_update_uses_optimistic_lock_to_protect_a_concurrent_edit(client):
    _, paper = seeded(client)
    store = client.app.state.store
    old = paper["experiments"][0]
    first = {**old, "name": "First editor"}
    assert store.update_experiment(first, expected=old)
    assert not store.update_experiment({**old, "name": "Second editor"}, expected=old)
    assert store.get_experiment(old["id"])["name"] == "First editor"
    assert len(store.experiment_history(old["id"])) == 1


def test_reparse_adds_new_evidence_only_after_explicit_acceptance(client, monkeypatch):
    from backend.app import parsing
    _, paper = seeded(client)
    original_parse = parsing.parse_document

    def upgraded_parser(*args):
        parsed = original_parse(*args)
        # Emulate a parser upgrade that attaches improved table/section context
        # to a stable text ID. The original record must stay byte-for-byte intact.
        parsed["evidence"] = [{**e, "section": "Improved experimental context"} for e in parsed["evidence"]]
        return parsed

    monkeypatch.setattr(parsing, "parse_document", upgraded_parser)
    response = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule", "reparse": True})
    assert response.status_code == 200, response.text
    generated = response.json()
    assert generated["evidence"] == paper["evidence"]
    assert generated["experiments"] == paper["experiments"]
    candidate = generated["extraction_candidates"][0]
    assert candidate["reparsed"] is True
    assert all(e["section"] == "Improved experimental context" for e in candidate["source_evidence"])
    original_ids = {e["id"] for e in paper["evidence"]}
    candidate_ids = {e["id"] for e in candidate["source_evidence"]}
    assert not original_ids & candidate_ids
    assert all(set(e["evidence_ids"]) <= candidate_ids for e in candidate["experiments"])
    accepted = client.post(candidate_url(paper, candidate) + "/accept", json={})
    assert accepted.status_code == 200, accepted.text
    result = accepted.json()
    records = {e["id"]: e for e in result["evidence"]}
    assert all(records[e["id"]] == e for e in paper["evidence"])
    assert all(records[e["id"]] == e for e in candidate["source_evidence"])
    assert result["evidence_count"] == len(paper["evidence"]) + len(candidate["source_evidence"])
    assert client.get(f"/api/papers/{paper['id']}/file").status_code == 200


def test_reparse_refuses_modified_original_and_preserves_all_saved_records(client):
    _, paper = seeded(client)
    path = client.app.state.store.get_file(paper["id"])
    path.write_bytes(path.read_bytes() + b"\nchanged externally")
    response = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule", "reparse": True})
    assert response.status_code == 409
    assert "SHA256" in response.json()["detail"]
    assert client.get(f"/api/papers/{paper['id']}").json() == paper


def test_repeated_reparse_acceptance_reuses_evidence_and_does_not_duplicate_experiments(client, monkeypatch):
    from backend.app import parsing
    _, paper = seeded(client)
    original_parse = parsing.parse_document

    def upgraded_parser(*args):
        parsed = original_parse(*args)
        parsed["evidence"] = [{**e, "section": "Upgraded parser section"} for e in parsed["evidence"]]
        return parsed

    monkeypatch.setattr(parsing, "parse_document", upgraded_parser)
    first = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule", "reparse": True}).json()
    accepted_first = client.post(candidate_url(paper, first["extraction_candidates"][0]) + "/accept", json={}).json()
    second = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule", "reparse": True}).json()
    second_candidate = second["extraction_candidates"][0]
    assert second_candidate["source_evidence"] == first["extraction_candidates"][0]["source_evidence"]
    accepted_second = client.post(candidate_url(paper, second_candidate) + "/accept", json={})
    assert accepted_second.status_code == 200, accepted_second.text
    after = accepted_second.json()
    assert after["evidence"] == accepted_first["evidence"]
    assert after["experiments"] == accepted_first["experiments"]
    assert after["evidence_count"] == accepted_first["evidence_count"]
    assert after["experiment_count"] == accepted_first["experiment_count"]


def test_manual_field_association_creates_only_literal_source_quotes(client):
    _, paper = seeded(client)
    experiment = next(e for e in paper["experiments"] if e.get("method") and e.get("field_evidence", {}).get("method"))
    source_id = experiment["field_evidence"]["method"][0]
    patch = {"method": experiment["method"], "field_evidence": {"method": [source_id]}, "evidence_ids": experiment["evidence_ids"]}
    valid = client.patch(f"/api/experiments/{experiment['id']}", json=patch)
    assert valid.status_code == 200, valid.text
    quote = valid.json()["evidence_quotes"]["method"][0]
    assert experiment["method"] in quote["quote"]
    assert quote["quote"] in next(e["text"] for e in paper["evidence"] if e["id"] == source_id)
    invalid = client.patch(f"/api/experiments/{experiment['id']}", json={**patch, "method": "InventedModel"})
    assert invalid.status_code == 200
    assert "method" not in invalid.json()["evidence_quotes"]
    assert invalid.json()["review_required"] is True


def test_chat_configuration_public_fields_are_sanitized(tmp_path, monkeypatch):
    from backend.app.config import Config
    monkeypatch.setenv("LLM_API_KEY", "chat-private-key")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://user:secret@chat.example/private/path?token=private")
    cfg = Config.from_env(tmp_path)
    assert cfg.provider_config()["api_key"] == "chat-private-key"
    public = cfg.public()
    assert public["llm_base_url"] == "https://chat.example"
    assert set(public["provider_status"]) == {"chat"}
    encoded = json.dumps(public)
    for secret in ("chat-private-key", "secret", "private/path", "token=private"):
        assert secret not in encoded


@pytest.mark.parametrize("failure,status", [(ImportError("missing component"), 503), (RuntimeError("private-runtime-marker"), 500)])
def test_comparison_failures_are_actionable_and_keep_previous_reports(client, monkeypatch, failure, status):
    from backend.app import orchestration
    papers, _ = seeded(client)
    scope = {"paper_ids": [p["id"] for p in papers]}
    original = client.post("/api/compare", json=scope).json()

    def fail(*args):
        raise failure

    monkeypatch.setattr(orchestration, "run_comparison", fail)
    response = client.post("/api/compare", json=scope)
    assert response.status_code == status
    assert "private-runtime-marker" not in response.text
    assert client.get(f"/api/reports/{original['id']}").json() == original


def test_markdown_report_exposes_conditions_field_sources_and_review_limits(client):
    papers, _ = seeded(client)
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    exported = client.get(f"/api/reports/{report['id']}/export").text
    assert "## 字段原文与复核" in exported
    assert "字面出现不能代替实验归属和语义复核" in exported
    assert "|样本量|条件|" in exported
    for experiment in report["experiments"]:
        for key, value in experiment["conditions"].items():
            assert f"{key}={value}" in exported
        for references in experiment.get("evidence_quotes", {}).values():
            for reference in references:
                assert reference["quote"].replace("\n", "\n> ") in exported
