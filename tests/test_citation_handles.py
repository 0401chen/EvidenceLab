"""Offline protocol checks: source fidelity is distinct from semantic support."""

import asyncio
import copy
import json

import pytest

from backend.app import providers
from backend.app.errors import ProviderError


# Archived V1 protocol fixtures remain explicit; production defaults to V2.
CONFIG = {"model": "offline-test-model", "base_url": "http://offline.invalid/v1"}


def test_chinese_answer_selects_english_source_without_copying_long_ids(monkeypatch, evidence_factory):
    source = evidence_factory("AlphaNet reached accuracy 84% on the held-out set.", "persistent-very-long-source-id")
    original = copy.deepcopy(source)
    calls = []

    async def reply(messages, config):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        assert payload["citation_protocol"] == "server_bound_fragments_v1"
        assert payload["available_evidence_ids"] == ["F1"]
        assert "persistent-very-long-source-id" not in messages[-1]["content"]
        assert payload["source_data"][0]["text"] == source["text"]
        return {"answer": "Unchecked claim: 99%.", "assertions": [{"text": "AlphaNet 的留出集准确率为84%。", "evidence": [{"handle": "F1"}]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.answer_llm("AlphaNet accuracy?", [source], CONFIG))
    assert len(calls) == 1
    assert "99" not in result["answer"]
    assert result["citations"] == [original]
    assert result["citation_quotes"][0]["quote"] == original["text"]
    assert result["citation_quotes"][0]["id"] == original["id"]
    assert result["assertion_evidence"][0]["citation_quotes"] == result["citation_quotes"]
    assert source == original
    assert result["verdict"] == "unverified"


@pytest.mark.parametrize("reference", [
    {"handle": "F999"},
    {"handle": "F1", "quote": "模型准确率为84%。"},
    {"handle": "F1", "id": "known-source"},
    {"id": "known-source", "quote": "模型准确率为84%。"},
    {"id": "fabricated", "quote": "AlphaNet accuracy is 84%."},
])
def test_bad_handle_or_legacy_translation_is_never_repaired_by_text(monkeypatch, evidence_factory, reference):
    calls = []

    async def reply(messages, config):
        calls.append(json.loads(messages[-1]["content"]))
        return {"assertions": [{"text": "准确率为84%。", "evidence": [reference]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError) as error:
        asyncio.run(providers.answer_llm("AlphaNet accuracy?", [evidence_factory("AlphaNet accuracy is 84%.", "known-source")], CONFIG))
    assert error.value.code == "grounding"
    assert len(calls) == 2
    assert calls[0]["source_data"] == calls[1]["source_data"]


def test_fragment_quotes_retain_raw_pdf_wrapping_and_bind_identical_text_to_its_source(evidence_factory):
    text = "AlphaNet classiﬁcation with batch size 32.\n" * 45 + "Accuracy is 84%."
    evidence = [evidence_factory(text, "first", "paper-a"), evidence_factory(text, "second", "paper-b")]
    payload, bindings = providers._citation_fragments(evidence)
    assert len(bindings) > 2
    for fragment in payload:
        binding = bindings[fragment["id"]]
        assert binding["source"]["text"][binding["start"]:binding["end"]] == fragment["text"]
    chosen = next(handle for handle, binding in bindings.items() if binding["source"]["id"] == "second")
    refs, quotes = providers._resolve_citation_references([{"handle": chosen}], evidence, bindings)
    assert [ref["id"] for ref in refs] == ["second"]
    assert quotes[0]["paper_id"] == "paper-b"
    assert "ﬁ" in quotes[0]["quote"]


def test_qa_combined_assertion_retains_all_selected_source_quotes(monkeypatch, evidence_factory):
    evidence = [evidence_factory("AlphaNet accuracy was 84%.", "measurement"),
                evidence_factory("For this AlphaNet result, we used a single model.", "configuration")]

    async def reply(messages, config):
        return {"assertions": [{"text": "该84%结果使用单模型。", "evidence": [{"handle": "F1"}, {"handle": "F2"}]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.answer_llm("AlphaNet accuracy model?", evidence, CONFIG))
    assert {quote["id"] for quote in result["citation_quotes"]} == {"measurement", "configuration"}
    assert len(result["assertion_evidence"][0]["citation_quotes"]) == 2


@pytest.mark.parametrize("verdicts,expected", [
    (["supported", "supported"], "supported"),
    (["supported", "insufficient"], "partial"),
    (["insufficient", "insufficient"], "insufficient"),
    (["supported", "contradicted"], "contradicted"),
])
def test_verifier_aggregates_checked_subclaims_instead_of_untrusted_overall_verdict(monkeypatch, evidence_factory, verdicts, expected):
    first, second = "AlphaNet accuracy is 84%", "the protocol used five runs"
    claim = first + "; " + second

    async def reply(messages, config):
        return {"verdict": "supported", "checks": [
            {"claim": text, "verdict": verdict, "explanation": "仅针对所选原文的判断。",
             "evidence": [] if verdict == "insufficient" else [{"handle": "F1"}]}
            for text, verdict in zip([first, second], verdicts)]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory("AlphaNet accuracy is 84% using five runs.")], CONFIG))
    assert result["verdict"] == expected
    assert len(result["claim_checks"]) == 2
    assert result["verification_protocol"] == "atomic_checks_v1"
    assert "只保证引文精确" in "".join(result["warnings"])


@pytest.mark.parametrize("check", [
    {"claim": "AlphaNet accuracy is 84%", "verdict": "contradicted", "explanation": "未提到。", "evidence": []},
    {"claim": "AlphaNet accuracy is 84%", "verdict": "supported", "explanation": "支持。", "evidence": []},
    {"claim": "AlphaNet accuracy is 84%", "verdict": "supported", "explanation": "支持。", "evidence": [{"handle": "F1"}]},
    {"claim": "Rewritten assertion", "verdict": "supported", "explanation": "支持。", "evidence": [{"handle": "F1"}]},
])
def test_verifier_rejects_unreferenced_judgments_and_omitted_conditions(monkeypatch, evidence_factory, check):
    async def reply(messages, config):
        return {"checks": [check]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError):
        asyncio.run(providers.verify_claim("AlphaNet accuracy is 84% on test-B", [evidence_factory("AlphaNet accuracy is 84% on test-A.")], CONFIG))


def test_handle_verdict_requires_checks_instead_of_bypassing_atomic_protocol(monkeypatch, evidence_factory):
    async def reply(messages, config):
        return {"verdict": "supported", "explanation": "Unsupported format.", "evidence": [{"handle": "F1"}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError) as error:
        asyncio.run(providers.verify_claim("AlphaNet accuracy is 84%", [evidence_factory("AlphaNet accuracy is 84%.")], CONFIG))
    assert error.value.code == "output_format"


def test_corrected_handle_does_not_leave_quotes_from_rejected_attempt(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        references = [{"handle": "F1"}, {"handle": "unknown"}] if len(calls) == 1 else [{"handle": "F2"}]
        return {"assertions": [{"text": "原文报告该结果。", "evidence": references}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    evidence = [evidence_factory("AlphaNet accuracy is 84%.", "one"), evidence_factory("BetaNet accuracy is 87%.", "two")]
    result = asyncio.run(providers.answer_llm("accuracy?", evidence, CONFIG))
    assert [quote["id"] for quote in result["citation_quotes"]] == ["two"]
    assert [item["id"] for item in result["citations"]] == ["two"]
