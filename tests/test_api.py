"""Offline end-to-end behavior through the public HTTP API."""

import pytest

from backend.app.errors import ProviderError


def upload(client, name, text):
    response = client.post(
        "/api/papers/upload",
        files=[("files", (name, text.encode("utf-8"), "text/markdown"))],
    )
    assert response.status_code == 200, response.text
    assert not response.json()["errors"], response.text
    return response.json()["papers"][0]


def test_empty_workspace_and_health(client):
    assert client.get("/api/papers").json() == []
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "llm_configured" in health.json()["providers"]



def test_public_settings_never_expose_key_or_url_credentials(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    monkeypatch.setenv("LLM_API_KEY", "api-key-private-marker")
    monkeypatch.setenv("LLM_MODEL", "offline-test-model")
    monkeypatch.setenv(
        "LLM_BASE_URL",
        "https://private-user:password-marker@example.org:8443/private-path-marker"
        "?token=query-marker#fragment-marker",
    )
    with TestClient(create_app(tmp_path / "safe-settings")) as scoped_client:
        response = scoped_client.get("/api/settings")
        assert response.status_code == 200
        for secret in (
            "api-key-private-marker", "private-user", "password-marker",
            "private-path-marker", "query-marker", "fragment-marker",
        ):
            assert secret not in response.text
        assert response.json()["llm_base_url"] == "https://example.org:8443"


def test_markdown_upload_and_duplicate_content_are_deduplicated(client):
    text = "# Synthetic upload\n\nA cobalt catalyst has a measured efficiency of 84 percent."
    first = upload(client, "first.md", text)
    second = upload(client, "renamed.md", text)
    assert first["status"] == "ready"
    assert first["id"] == second["id"]
    assert len(client.get("/api/papers").json()) == 1
    detail = client.get(f"/api/papers/{first['id']}").json()
    assert detail["evidence"]
    assert all(e["paper_id"] == first["id"] for e in detail["evidence"])
    file_response = client.get(f"/api/papers/{first['id']}/file")
    assert file_response.status_code == 200
    assert "cobalt catalyst" in file_response.text


def test_invalid_and_empty_uploads_do_not_leave_paper_records(client):
    response = client.post(
        "/api/papers/upload",
        files=[
            ("files", ("invalid.exe", b"unsupported", "application/octet-stream")),
            ("files", ("empty.md", b"", "text/markdown")),
        ],
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["papers"] == []
    assert len(result["errors"]) == 2
    assert client.get("/api/papers").json() == []


def test_scoped_qa_never_returns_an_unselected_paper(client):
    cobalt = upload(
        client, "cobalt.md",
        "# Cobalt experiment\n\nThe cobalt catalyst efficiency is 84 percent at room temperature.",
    )
    battery = upload(
        client, "battery.md",
        "# Battery experiment\n\nGraphite battery capacity is 320 milliamp hours per gram.",
    )
    response = client.post("/api/qa", json={
        "question": "What is the cobalt catalyst efficiency?",
        "paper_ids": [cobalt["id"]],
        "engine": "local",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["verdict"] == "evidence_found"
    assert result["citations"]
    assert {e["paper_id"] for e in result["citations"]} == {cobalt["id"]}
    scoped = client.post("/api/qa", json={
        "question": "What is the graphite battery capacity?",
        "paper_ids": [cobalt["id"]],
        "engine": "local",
    })
    assert scoped.status_code == 200
    assert all(e["paper_id"] != battery["id"] for e in scoped.json()["citations"])


def test_nonexistent_ids_are_rejected_in_all_corpus_routes(client):
    paper = upload(client, "valid.md", "# Valid\n\nA synthetic experiment measures 84 percent efficiency.")
    missing = "does-not-exist"
    assert client.get(f"/api/papers/{missing}").status_code == 404
    assert client.get(f"/api/papers/{missing}/file").status_code == 404
    assert client.delete(f"/api/papers/{missing}").status_code == 404
    assert client.post(f"/api/papers/{missing}/extract", json={"mode": "rule"}).status_code == 404
    for path, body in [
        ("/api/qa", {"question": "What is the efficiency?", "engine": "local"}),
        ("/api/verify", {"claim": "The efficiency is 84 percent."}),
        ("/api/compare", {}),
    ]:
        response = client.post(path, json={**body, "paper_ids": [paper["id"], missing]})
        assert response.status_code == 404, (path, response.text)


def test_demo_seed_is_explicit_idempotent_and_labeled(client):
    assert client.get("/api/papers").json() == []
    first = client.post("/api/demo/seed")
    second = client.post("/api/demo/seed")
    assert first.status_code == second.status_code == 200
    first_papers = first.json()["papers"]
    second_papers = second.json()["papers"]
    assert len(first_papers) == 3
    assert {p["id"] for p in first_papers} == {p["id"] for p in second_papers}
    assert all(p["is_demo"] for p in first_papers)
    assert all(p["experiment_count"] >= 1 for p in first_papers)
    assert len(client.get("/api/papers").json()) == 3


def test_export_references_resolve_to_actual_paper_evidence(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    response = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["comparisons"]
    actual_evidence = {}
    for paper in papers:
        for item in client.get(f"/api/papers/{paper['id']}").json()["evidence"]:
            actual_evidence[item["id"]] = item
    referenced = {e for comparison in report["comparisons"] for e in comparison["evidence_ids"]}
    assert referenced
    assert referenced <= actual_evidence.keys()

    exported_json = client.get(f"/api/reports/{report['id']}/export?format=json")
    assert exported_json.status_code == 200, exported_json.text
    exported = exported_json.json()
    exported_evidence = {e["id"]: e for e in exported["evidence"]}
    assert referenced <= exported_evidence.keys()
    for evidence_id, item in exported_evidence.items():
        assert item == actual_evidence[evidence_id]

    markdown = client.get(f"/api/reports/{report['id']}/export?format=markdown")
    assert markdown.status_code == 200
    for evidence_id in referenced:
        assert evidence_id in markdown.text
        quoted_source = "> " + actual_evidence[evidence_id]["text"].replace("\n", "\n> ")
        assert quoted_source in markdown.text


def test_offline_verification_keeps_semantic_judgment_insufficient(client):
    paper = upload(client, "claim.md", "# Result\n\nThe cobalt catalyst efficiency is 84 percent.")
    response = client.post("/api/verify", json={
        "claim": "The cobalt catalyst efficiency is 84 percent.",
        "paper_ids": [paper["id"]],
    })
    assert response.status_code == 200, response.text
    assert response.json()["verdict"] == "insufficient"
    assert response.json()["warnings"]


def test_delete_removes_detail_file_and_retrievable_evidence(client):
    paper = upload(client, "delete.md", "# Disposable\n\nZirconium conversion reaches 71 percent.")
    paper_id = paper["id"]
    assert client.get(f"/api/papers/{paper_id}").json()["evidence"]
    deleted = client.delete(f"/api/papers/{paper_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get("/api/papers").json() == []
    assert client.get(f"/api/papers/{paper_id}").status_code == 404
    assert client.get(f"/api/papers/{paper_id}/file").status_code == 404
    qa = client.post("/api/qa", json={
        "question": "What is the zirconium conversion?",
        "paper_ids": [],
        "engine": "local",
    })
    assert qa.status_code == 400, qa.text  # No corpus remains to search.


def test_deleting_a_referenced_paper_invalidates_its_saved_report(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    assert client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 200
    assert client.delete(f"/api/papers/{papers[0]['id']}").status_code == 200
    assert client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 404


def test_manual_edit_rejects_foreign_evidence_and_invalidates_old_report(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    first = client.get(f"/api/papers/{papers[0]['id']}").json()
    second = client.get(f"/api/papers/{papers[1]['id']}").json()
    experiment = first["experiments"][0]
    rejected = client.patch(f"/api/experiments/{experiment['id']}", json={
        "evidence_ids": [second["evidence"][0]["id"]],
    })
    assert rejected.status_code == 400
    report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()
    patched = client.patch(f"/api/experiments/{experiment['id']}", json={"value": 0})
    assert patched.status_code == 200, patched.text
    assert patched.json()["value"] == 0
    assert any("人工修正" in note for note in patched.json()["notes"])
    assert client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 404


@pytest.mark.parametrize(
    ("failure", "expected_message"),
    [
        (
            ProviderError("authentication", "模型服务拒绝认证，请检查服务器 API 密钥与访问权限。"),
            "模型服务拒绝认证，请检查服务器 API 密钥与访问权限。",
        ),
        (
            ProviderError("unknown_evidence", "模型引用了不存在或超出所选文献范围的证据，结果已拒绝。"),
            "模型引用了不存在或超出所选文献范围的证据，结果已拒绝。",
        ),
        (ValueError("sensitive-provider-token-marker: private prompt contents"), None),
        (RuntimeError("sensitive-provider-token-marker: private provider response"), None),
    ],
)
def test_model_failure_is_specific_when_safe_and_never_replaces_saved_results(
    tmp_path, monkeypatch, caplog, failure, expected_message,
):
    from fastapi.testclient import TestClient
    from backend.app import providers
    from backend.app.main import create_app

    monkeypatch.setenv("LLM_API_KEY", "test-configured-key")
    monkeypatch.setenv("LLM_MODEL", "offline-test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://offline.invalid/v1")

    async def failed_extraction(*args):
        raise failure

    monkeypatch.setattr(providers, "extract_llm", failed_extraction)
    caplog.set_level("WARNING", logger="evidencelab")
    with TestClient(create_app(tmp_path / "model-failure")) as scoped_client:
        papers = scoped_client.post("/api/demo/seed").json()["papers"]
        paper_id = papers[0]["id"]
        previous = scoped_client.get(f"/api/papers/{paper_id}").json()
        assert previous["experiments"]
        report = scoped_client.post("/api/compare", json={
            "paper_ids": [paper["id"] for paper in papers],
        }).json()

        response = scoped_client.post(f"/api/papers/{paper_id}/extract", json={"mode": "llm"})

        assert response.status_code == 502
        if expected_message:
            assert response.json()["detail"] == expected_message
        else:
            assert response.json()["detail"].startswith("模型服务调用失败。")
        assert "sensitive-provider-token-marker" not in response.text
        assert "private prompt" not in response.text
        assert "private provider" not in response.text
        assert "sensitive-provider-token-marker" not in caplog.text
        assert "test-configured-key" not in response.text
        assert scoped_client.get(f"/api/papers/{paper_id}").json() == previous
        assert scoped_client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 200


def test_model_extraction_persists_only_grounded_rows_and_partial_notice(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient
    from backend.app import providers

    monkeypatch.setenv("EVIDENCELAB_DATA_DIR", str(tmp_path / "module-data"))
    monkeypatch.setenv("LLM_API_KEY", "offline-dummy-key")
    monkeypatch.setenv("LLM_MODEL", "offline-test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://offline.invalid/v1")
    monkeypatch.setenv("EVIDENCE_PARSER", "pymupdf")
    from backend.app.main import create_app

    calls = []

    async def mixed_experiments(messages, config):
        source = json.loads(messages[1]["content"])
        calls.append(source)
        evidence = next(item for item in source["source_data"] if "AlphaNet" in item["text"])

        def field(value, quote):
            return {"value": value, "evidence": [{"id": evidence["id"], "quote": quote}]}

        def candidate(value):
            return {
                "name": "合格候选" if value == 85 else "错误数值候选",
                "method": field("AlphaNet", "AlphaNet on DemoSet"),
                "dataset": field("DemoSet", "AlphaNet on DemoSet"),
                "split": field("test-A", "on test-A"),
                "metric": field("accuracy", "accuracy 85%"),
                # The quote itself is genuine, but it cannot support 99.
                "value": field(value, "accuracy 85%"),
                "unit": field("%", "accuracy 85%"),
                "sample_size": None,
                "conditions": field({"batch size": "32"}, "batch size 32"),
            }

        return {"experiments": [candidate(85), candidate(99)]}

    monkeypatch.setattr(providers, "_chat_json", mixed_experiments)
    with TestClient(create_app(tmp_path / "partial-model-extraction")) as scoped_client:
        paper = upload(
            scoped_client, "grounded.md",
            "AlphaNet on DemoSet reached accuracy 85% on test-A with batch size 32.",
        )
        paper_id = paper["id"]
        before = scoped_client.get(f"/api/papers/{paper_id}").json()
        original_experiment = before["experiments"][0]
        manually_corrected = scoped_client.patch(f"/api/experiments/{original_experiment['id']}", json={"value": 0, "name": "人工核查记录"}).json()
        before = scoped_client.get(f"/api/papers/{paper_id}").json()
        actual_evidence_ids = {item["id"] for item in before["evidence"]}

        response = scoped_client.post(f"/api/papers/{paper_id}/extract", json={"mode": "llm"})

        assert response.status_code == 200, response.text
        result = response.json()
        assert len(calls) == 2  # Exactly one correction, still with a bad row.
        assert "validation_feedback" in calls[1]
        assert all(
            item["paper_id"] == paper_id
            for call in calls for item in call["source_data"]
        )
        assert result["experiment_count"] == len(result["experiments"]) == len(before["experiments"])
        assert result["experiments"] == before["experiments"]
        candidate = result["extraction_candidates"][0]
        assert candidate["status"] == "pending"
        assert len(candidate["experiments"]) == 1
        experiment = candidate["experiments"][0]
        assert experiment["value"] == 85
        assert experiment["extraction_method"] == "llm"
        assert experiment["method"] == "AlphaNet"
        assert experiment["dataset"] == "DemoSet"
        assert any(
            "部分" in note and "1 条" in note and "已排除" in note and "不代表全文" in note
            for note in experiment["notes"]
        )
        assert set(experiment["evidence_ids"]) == actual_evidence_ids
        for references in experiment["field_evidence"].values():
            assert references
            assert set(references) <= actual_evidence_ids

        persisted = scoped_client.get(f"/api/papers/{paper_id}").json()
        assert persisted == {k: v for k, v in result.items() if k != "latest_extraction_candidate_id"}
        assert persisted["evidence"] == before["evidence"]
        accepted = scoped_client.post(f"/api/papers/{paper_id}/extraction-candidates/{candidate['id']}/accept", json={})
        assert accepted.status_code == 200, accepted.text
        accepted_experiments = accepted.json()["experiments"]
        assert next(item for item in accepted_experiments if item["id"] == manually_corrected["id"]) == manually_corrected
        assert next(item for item in accepted_experiments if item["id"] == experiment["id"]) == experiment
        listed = next(item for item in scoped_client.get("/api/papers").json() if item["id"] == paper_id)
        assert listed["experiment_count"] == len(before["experiments"]) + 1
