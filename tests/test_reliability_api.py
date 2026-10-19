"""Explicit selection, upload cleanup, and honest manual provenance."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_extraction_api import configured_app


@pytest.mark.parametrize("path,body", [("/api/qa", {"question": "accuracy?", "engine": "local"}),
                                      ("/api/verify", {"claim": "accuracy is 85"}), ("/api/compare", {})])
@pytest.mark.parametrize("include_empty", [True, False])
def test_empty_selection_never_expands_to_all_ready_papers(client, path, body, include_empty):
    client.post("/api/demo/seed")
    payload = {**body, **({"paper_ids": []} if include_empty else {})}
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert "选择" in response.json()["detail"]


def test_parser_exception_details_stay_private_and_failed_file_is_cleaned(client, monkeypatch):
    from backend.app import parsing

    def fail(*args):
        raise ValueError("private-parser-token-marker and full private source")

    monkeypatch.setattr(parsing, "parse_document", fail)
    response = client.post("/api/papers/upload", files=[("files", ("broken.md", b"not empty", "text/markdown"))])
    assert response.json()["papers"] == []
    assert "private-parser-token-marker" not in response.text
    assert not list(client.app.state.store.files.iterdir())


def test_partial_file_write_failure_is_cleaned(client, monkeypatch):
    original = Path.write_bytes
    files = client.app.state.store.files

    def fail(path, content):
        if path.parent == files:
            original(path, b"partial")
            raise OSError("simulated disk failure")
        return original(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail)
    response = client.post("/api/papers/upload", files=[("files", ("broken.md", b"source text", "text/markdown"))])
    assert response.json()["papers"] == []
    assert not list(files.iterdir())


@pytest.mark.parametrize("patch", [{"value": True}, {"value": []}, {"value": {}}, {"name": "  "}])
def test_invalid_manual_values_are_validation_errors_and_do_not_write(client, patch):
    pid = client.post("/api/demo/seed").json()["papers"][0]["id"]
    previous = client.get(f"/api/papers/{pid}").json()
    eid = previous["experiments"][0]["id"]
    assert client.patch(f"/api/experiments/{eid}", json=patch).status_code == 422
    assert client.get(f"/api/papers/{pid}").json() == previous


def test_manual_change_cannot_keep_a_models_old_field_quote(client):
    pid = client.post("/api/demo/seed").json()["papers"][0]["id"]
    experiment = client.get(f"/api/papers/{pid}").json()["experiments"][0]
    assert experiment["evidence_quotes"]["value"]
    result = client.patch(f"/api/experiments/{experiment['id']}", json={"value": 0}).json()
    assert result["value"] == 0
    assert "value" not in result["evidence_quotes"]
    assert "value" not in result["field_evidence"]
    assert result["review_required"] is True


def test_edit_form_unchanged_fields_keep_their_original_quotes(client):
    pid = client.post("/api/demo/seed").json()["papers"][0]["id"]
    experiment = client.get(f"/api/papers/{pid}").json()["experiments"][0]
    result = client.patch(f"/api/experiments/{experiment['id']}", json={
        "value": 0, "method": experiment["method"], "evidence_ids": experiment["evidence_ids"],
    }).json()
    assert "value" not in result["evidence_quotes"]
    assert result["evidence_quotes"]["method"] == experiment["evidence_quotes"]["method"]


def test_delete_removes_only_selected_papers_source(client):
    papers = client.post("/api/demo/seed").json()["papers"]
    selected, retained = papers[:2]
    path = client.app.state.store.get_file(selected["id"])
    retained_path = client.app.state.store.get_file(retained["id"])
    assert client.delete(f"/api/papers/{selected['id']}").status_code == 200
    assert not path.exists()
    assert retained_path.is_file()
    assert client.get(f"/api/papers/{selected['id']}").status_code == 404


def test_edit_during_comparison_cannot_resurrect_a_stale_report(client, monkeypatch):
    from backend.app import orchestration
    original = orchestration.run_comparison
    papers = client.post("/api/demo/seed").json()["papers"]
    store = client.app.state.store

    def edit_during_analysis(experiments, evidence):
        result = original(experiments, evidence)
        edited = dict(experiments[0])
        edited["value"] = 0
        store.update_experiment(edited)
        return result

    monkeypatch.setattr(orchestration, "run_comparison", edit_during_analysis)
    response = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]})
    assert response.status_code == 409
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0
