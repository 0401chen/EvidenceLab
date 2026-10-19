"""Ordinary users can establish grounded records without a model or a seed script."""
import pytest


def original(client, filename="study.md"):
    text = "# Research\n\nAlpha achieves accuracy 71.2% on Bench test set.\n\nBeta achieves accuracy 76.4% on Other test set."
    response = client.post("/api/papers/upload", files={"files": (filename, text.encode(), "text/markdown")})
    pid = response.json()["papers"][0]["id"]
    paper = client.get(f"/api/papers/{pid}").json()
    source = next(e for e in paper["evidence"] if "Alpha achieves" in e["text"])
    quote = {"id": source["id"], "quote": "Alpha achieves accuracy 71.2% on Bench test set."}
    body = {"name": "Manually reviewed result", "fields": {"method": "Alpha", "dataset": "Bench",
            "split": "test", "metric": "accuracy", "value": 71.2, "unit": "%"},
            "evidence_quotes": {key: [quote] for key in ("method", "dataset", "split", "metric", "value", "unit")}}
    return pid, paper, body


def test_manual_candidate_review_is_append_only_idempotent_and_has_original_quotes(client, monkeypatch):
    from backend.app import providers
    async def forbidden(*args, **kwargs):
        raise AssertionError("Manual entry must never call a model")
    monkeypatch.setattr(providers, "_chat_json", forbidden)
    pid, before, body = original(client)
    response = client.post(f"/api/papers/{pid}/manual-candidates", json=body)
    assert response.status_code == 200, response.text
    pending = response.json()
    assert pending["experiments"] == before["experiments"]
    candidate = pending["extraction_candidates"][0]
    assert candidate["mode"] == "manual" and candidate["status"] == "pending"
    record = candidate["experiments"][0]
    assert record["extraction_method"] == "manual" and record["review_required"]
    assert record["evidence_quotes"]["value"] == body["evidence_quotes"]["value"]
    url = f"/api/papers/{pid}/extraction-candidates/{candidate['id']}/accept"
    accepted = client.post(url, json={}).json()
    assert client.post(url, json={}).json()["experiments"] == accepted["experiments"]
    assert all(item in accepted["experiments"] for item in before["experiments"])
    # A second equivalent candidate cannot duplicate the stored measurement.
    second = client.post(f"/api/papers/{pid}/manual-candidates", json=body).json()
    client.post(f"/api/papers/{pid}/extraction-candidates/{second['latest_extraction_candidate_id']}/accept", json={})
    assert client.get(f"/api/papers/{pid}").json()["experiments"] == accepted["experiments"]


@pytest.mark.parametrize("failure", ["invented_quote", "foreign_source", "wrong_method", "wrong_unit", "bool_value", "missing_value"])
def test_invalid_manual_sources_do_not_write(client, failure):
    pid, before, body = original(client)
    if failure == "invented_quote":
        body["evidence_quotes"]["value"] = [{"id": body["evidence_quotes"]["value"][0]["id"], "quote": "Alpha achieves accuracy 91.2%"}]
    elif failure == "foreign_source":
        body["evidence_quotes"]["value"] = [{"id": "not-this-paper", "quote": "71.2%"}]
    elif failure == "wrong_method":
        body["fields"]["method"] = "Beta"
        other = next(e for e in before["evidence"] if "Beta achieves" in e["text"])
        body["evidence_quotes"]["method"] = [{"id": other["id"], "quote": other["text"]}]
    elif failure == "wrong_unit":
        body["fields"]["unit"] = "seconds"
    elif failure == "bool_value":
        body["fields"]["value"] = True
    else:
        del body["fields"]["value"]
    response = client.post(f"/api/papers/{pid}/manual-candidates", json=body)
    assert response.status_code == 422, response.text
    assert client.get(f"/api/papers/{pid}").json() == before


def test_manual_candidate_still_requires_review_against_current_experiment_revision(client):
    pid, _, body = original(client)
    candidate = client.post(f"/api/papers/{pid}/manual-candidates", json=body).json()
    cid = candidate["latest_extraction_candidate_id"]
    existing = candidate["experiments"][0]
    assert client.patch(f"/api/experiments/{existing['id']}", json={"name": "new human review"}).status_code == 200
    assert client.post(f"/api/papers/{pid}/extraction-candidates/{cid}/accept", json={}).status_code == 409
    current = client.get(f"/api/papers/{pid}").json()
    assert current["experiments"][0]["name"] == "new human review"
    assert client.get(f"/api/experiments/{existing['id']}/history").json()
