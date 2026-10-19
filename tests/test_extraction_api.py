"""Extraction persistence boundaries, using temporary files and no model traffic."""
import pytest
from fastapi.testclient import TestClient


def configured_app(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCELAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_API_KEY", "test-placeholder")
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://offline.invalid/v1")
    from backend.app.main import create_app
    return create_app(tmp_path / "data")


@pytest.mark.parametrize("failure", ["empty", "foreign_field", "misquote", "boolean_value"])
def test_invalid_extraction_never_replaces_saved_records_or_reports(tmp_path, monkeypatch, failure):
    from backend.app import providers
    app = configured_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        papers = client.post("/api/demo/seed").json()["papers"]
        pid = papers[0]["id"]
        previous = client.get(f"/api/papers/{pid}").json()
        foreign = client.get(f"/api/papers/{papers[1]['id']}").json()["evidence"][0]["id"]
        report = client.post("/api/compare", json={"paper_ids": [p["id"] for p in papers]}).json()

        async def extract(*args):
            if failure == "empty":
                return []
            record = dict(previous["experiments"][0])
            record["field_evidence"] = dict(record["field_evidence"])
            if failure == "foreign_field":
                record["field_evidence"]["value"] = [foreign]
            elif failure == "misquote":
                record["evidence_quotes"] = {"value": [{"id": record["field_evidence"]["value"][0], "quote": "an invented measurement"}]}
            elif failure == "boolean_value":
                record["value"] = True
            return [record]

        monkeypatch.setattr(providers, "extract_llm", extract)
        response = client.post(f"/api/papers/{pid}/extract", json={"mode": "llm"})
        assert response.status_code in {422, 502}
        assert client.get(f"/api/papers/{pid}").json() == previous
        assert client.get(f"/api/reports/{report['id']}/export?format=json").status_code == 200


def test_empty_rule_extraction_preserves_existing_experiments(client, monkeypatch):
    from backend.app import analysis
    paper = client.post("/api/demo/seed").json()["papers"][0]
    previous = client.get(f"/api/papers/{paper['id']}").json()
    monkeypatch.setattr(analysis, "extract_experiments", lambda *args: [])
    response = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule"})
    assert response.status_code == 422
    assert client.get(f"/api/papers/{paper['id']}").json() == previous


def test_new_records_have_source_derived_title_and_pages(client):
    paper = client.post("/api/demo/seed").json()["papers"][0]
    detail = client.get(f"/api/papers/{paper['id']}").json()
    known = {e["id"]: e for e in detail["evidence"]}
    for experiment in detail["experiments"]:
        assert experiment["paper_title"] == detail["title"]
        assert experiment["source_pages"] == sorted({known[eid]["page"] for eid in experiment["evidence_ids"]})
        assert isinstance(experiment["review_required"], bool)
