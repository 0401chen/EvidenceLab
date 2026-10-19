"""Diagnostics distinguish failures without changing acceptance, calls or privacy."""
import asyncio
import json

import httpx
import pytest

from backend.app import providers
from backend.app.errors import ProviderError
from backend.app.extraction_diagnostics import LocalCapture, diagnostic_session, REQUEST_OPTIONS


def supplied(value, quote, identifier="S1"):
    return {"value": value, "evidence": [{"id": identifier, "quote": quote}]}


@pytest.mark.parametrize("failure,reason", [
    ("id", "evidence_id_invalid"), ("quote", "quote_not_in_source"),
    ("literal", "field_literal_mismatch"), ("missing", "evidence_missing"),
    ("short", "quote_missing_or_short"), ("unit", "unit_not_bound_to_result"),
    ("metric", "metric_result_conflict"), ("owner", "cross_sentence_owner_unconfirmed"),
])
def test_failure_categories_preserve_public_errors_and_rejections(evidence_factory, failure, reason):
    source = evidence_factory("Alpha has accuracy 82%. Beta has accuracy 90%.", "source")
    row = {"metric": supplied("accuracy", "accuracy 82%"), "value": supplied(82, "accuracy 82%"),
           "unit": supplied("%", "accuracy 82%")}
    if failure == "id": row["value"]["evidence"][0]["id"] = "S999"
    if failure == "quote": row["value"]["evidence"][0]["quote"] = "accuracy 99%"
    if failure == "literal": row["metric"]["value"] = "error rate"
    if failure == "missing": row["value"]["evidence"] = []
    if failure == "short": row["unit"]["evidence"][0]["quote"] = "%"
    if failure == "unit": row["unit"]["evidence"][0]["quote"] = "accuracy 90%"
    if failure == "metric": row["metric"]["evidence"][0]["quote"] = "accuracy 90%"
    if failure == "owner": row["method"] = supplied("Beta", "Beta")
    args = ({"experiments": [row]}, [source], "paper-a", {"S1":"source"}, 0)
    with pytest.raises(ProviderError) as plain: providers._checked_extraction(*args)
    details = []
    with pytest.raises(ProviderError) as traced: providers._checked_extraction(*args, diagnostics=details)
    assert str(plain.value) == str(traced.value) and traced.value.code == "grounding"
    assert reason in {x["reason"] for x in details}


@pytest.mark.parametrize("include_content", [False, True])
def test_opt_in_capture_keeps_call_count_and_redacts_secrets(tmp_path, monkeypatch, evidence_factory, include_content):
    secret = "test-private-credential-987654321"
    source = evidence_factory("Alpha has accuracy 82%. LOCAL_PRIVATE_PAPER_TEXT", "source")
    raw = {"experiments": [{"name": secret, "metric": supplied("error rate", "accuracy 82%"),
                            "value": supplied(82, "accuracy 82%"), "unit": supplied("%", "accuracy 82%")}]}
    calls = []
    async def offline_send(client, request, *args, **kwargs):
        calls.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={"choices":[{"message":{"content":json.dumps(raw)}, "finish_reason":"stop"}]})
    monkeypatch.setattr(httpx.AsyncClient, "send", offline_send)
    folder = tmp_path/"artifacts/diagnostics"
    options = {"mode":"llm", "pages":[7], "max_results":12, "reparse":False}
    token = REQUEST_OPTIONS.set(options)
    try:
        with diagnostic_session(folder, include_content=include_content, workspace=tmp_path):
            with pytest.raises(ProviderError) as exc:
                asyncio.run(providers.extract_llm([source],"paper-a",{"model":"offline", "api_key":secret, "enrich_metadata":False}))
            assert exc.value.code == "grounding"
    finally:
        REQUEST_OPTIONS.reset(token)
    assert len(calls) == 2
    content = next(folder.glob("*.jsonl")).read_text(encoding="utf8")
    assert secret not in content and "Bearer" not in content
    assert ("LOCAL_PRIVATE_PAPER_TEXT" in content) is include_content
    events = [json.loads(line) for line in content.splitlines()]
    assert events[-1]["event"] == "capture_finished" and events[-1]["capture_complete"]
    assert any(x.get("aliases") == {"S1":"source"} for x in events)
    assert next(x for x in events if x["event"]=="extraction_input")["request_options"] == options
    if include_content:
        response = next(x["data"] for x in events if x["event"]=="model_response")
        assert "[REDACTED]" in response["choices"][0]["message"]["content"]


def test_capture_unavailable_does_not_change_a_valid_result(tmp_path, monkeypatch, evidence_factory):
    source = evidence_factory("Alpha accuracy 82%", "source")
    calls = []
    async def reply(*args):
        calls.append(1)
        return {"experiments":[{"value":supplied(82,"accuracy 82%")}]}
    monkeypatch.setattr(providers,"_chat_json",reply)
    # Outside artifacts is refused, without creating a directory or failing extraction.
    with diagnostic_session(tmp_path/"refused", include_content=True, workspace=tmp_path):
        rows = asyncio.run(providers.extract_llm([source],"paper-a",{"model":"offline","enrich_metadata":False}))
    assert len(rows)==1 and rows[0]["value"]==82 and len(calls)==1
    assert not (tmp_path/"refused").exists()


def test_capture_quota_never_deletes_old_records_and_marks_truncation(tmp_path, monkeypatch):
    folder=tmp_path/"artifacts/diagnostics"
    monkeypatch.setattr(LocalCapture,"MAX_BYTES",6000)
    cap=LocalCapture(folder,include_content=True,workspace=tmp_path)
    cap.record("large",{"text":"x"*8000});cap.finish()
    events=[json.loads(line) for line in cap.path.read_text(encoding="utf8").splitlines()]
    assert not events[-1]["capture_complete"] and cap.path.stat().st_size<=6000
    previous=cap.path.read_bytes()
    monkeypatch.setattr(LocalCapture,"MAX_FILES",1)
    refused=LocalCapture(folder,include_content=True,workspace=tmp_path)
    assert not refused.enabled and cap.path.read_bytes()==previous


@pytest.mark.parametrize("quote", [
    "Alpha has accuracy 84%. Our Omega model achieves accuracy 82%.",
    "accuracy 84%. Our Omega model achieves accuracy 82%",
])
def test_long_and_cross_sentence_quotes_cannot_borrow_a_foreign_owner(evidence_factory, quote):
    text="Alpha has accuracy 84%. Our Omega model achieves accuracy 82%."
    source=evidence_factory(text,"source")
    row={"method":supplied("Alpha","Alpha"),
         "metric":supplied("accuracy",quote),"value":supplied(82,quote),"unit":supplied("%",quote)}
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments":[row]},[source],"paper-a",{"S1":"source"},0)
    # The same long quote may still support its own explicit result sentence.
    row["method"]=supplied("Omega","Our Omega model")
    assert providers._checked_extraction({"experiments":[row]},[source],"paper-a",{"S1":"source"},0)[0]["method"]=="Omega"
