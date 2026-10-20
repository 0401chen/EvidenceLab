"""Model output is untrusted; mock JSON responses without any network access."""

import asyncio
import json

import httpx
import pytest

from backend.app import providers
from backend.app.errors import ProviderError


# The original response fixtures explicitly exercise the V1 compatibility path.
CONFIG = {"model": "offline-test-model", "base_url": "http://offline.invalid/v1", "enrich_metadata": False}


def mock_json(monkeypatch, result):
    async def reply(messages, config):
        return result

    monkeypatch.setattr(providers, "_chat_json", reply)


@pytest.mark.parametrize(
    "reference",
    [
        {"id": "fabricated", "quote": "Catalyst efficiency is 84 percent."},
        {"id": "ev-a", "quote": "Catalyst efficiency is 99 percent."},
    ],
)
def test_qa_rejects_fabricated_ids_and_misquoted_sources(monkeypatch, evidence_factory, reference):
    evidence = [evidence_factory("Catalyst efficiency is 84 percent.")]
    mock_json(monkeypatch, {
        "answer": "An unchecked answer.",
        "assertions": [{"text": "Catalyst efficiency is 84 percent.", "evidence": [reference]}],
    })
    with pytest.raises(ValueError):
        asyncio.run(providers.answer_llm("What is the catalyst efficiency?", evidence, CONFIG))


def test_qa_displays_only_checked_assertions(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Catalyst efficiency is 84 percent.")]
    mock_json(monkeypatch, {
        "answer": "Unchecked invented result: efficiency is 99 percent.",
        "assertions": [{
            "text": "报告的催化剂效率为84%。",
            "evidence": [{"id": "ev-a", "quote": "Catalyst efficiency is 84 percent."}],
        }],
    })
    result = asyncio.run(providers.answer_llm("What is the catalyst efficiency?", evidence, CONFIG))
    assert "84" in result["answer"]
    assert "99" not in result["answer"]
    assert result["citations"] == evidence
    assert result["verdict"] == "unverified"


def test_qa_without_candidates_does_not_call_model(monkeypatch, evidence_factory):
    async def unexpected_call(messages, config):
        pytest.fail("No evidence should mean no model call")

    monkeypatch.setattr(providers, "_chat_json", unexpected_call)
    result = asyncio.run(providers.answer_llm(
        "How does a lunar telescope operate?",
        [evidence_factory("A cobalt catalyst converts methane.")], CONFIG,
    ))
    assert result["verdict"] == "insufficient"
    assert result["citations"] == []


def test_verifier_rejects_an_existing_id_outside_candidate_scope(monkeypatch, evidence_factory):
    evidence = [
        evidence_factory("Catalyst efficiency is 84 percent.", "selected"),
        evidence_factory("A lunar telescope uses optical mirrors.", "outside"),
    ]
    mock_json(monkeypatch, {
        "verdict": "supported", "explanation": "This is supported.",
        "evidence": [{"id": "outside", "quote": "A lunar telescope uses optical mirrors."}],
    })
    with pytest.raises(ValueError):
        asyncio.run(providers.verify_claim("Catalyst efficiency is 84 percent.", evidence, CONFIG))


def test_verifier_cannot_report_supported_without_citations(monkeypatch, evidence_factory):
    mock_json(monkeypatch, {"verdict": "supported", "explanation": "Supported.", "evidence": []})
    with pytest.raises(ValueError):
        asyncio.run(providers.verify_claim(
            "Catalyst efficiency is 84 percent.",
            [evidence_factory("Catalyst efficiency is 84 percent.")], CONFIG,
        ))


def test_offline_verifier_does_not_turn_a_keyword_match_into_support(evidence_factory):
    result = asyncio.run(providers.verify_claim(
        "Catalyst efficiency is 84 percent.",
        [evidence_factory("Catalyst efficiency is 84 percent.")], {},
    ))
    assert result["verdict"] == "insufficient"
    assert result["citations"]
    assert any(trace["status"] == "skipped" for trace in result["traces"])


def test_llm_extraction_requires_each_field_to_be_grounded(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Method: AlphaNet\nMetric: accuracy\nValue: 85")]
    mock_json(monkeypatch, {"experiments": [{
        "name": "invented", "metric": "accuracy", "value": 99,
        "field_evidence": {
            "metric": [{"id": "ev-a", "quote": "Metric: accuracy"}],
            "value": [{"id": "ev-a", "quote": "Value: 85"}],
        },
    }]})
    with pytest.raises(ValueError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def test_llm_extraction_does_not_accept_cross_paper_evidence(monkeypatch, evidence_factory):
    evidence = [
        evidence_factory("Metric: accuracy\nValue: 85", "own", "paper-a"),
        evidence_factory("Metric: accuracy\nValue: 90", "foreign", "paper-b"),
    ]
    mock_json(monkeypatch, {"experiments": [{
        "name": "foreign result", "value": 90,
        "field_evidence": {"value": [{"id": "foreign", "quote": "Value: 90"}]},
    }]})
    with pytest.raises(ValueError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def test_llm_extraction_preserves_a_grounded_zero(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Metric: error rate\nValue: 0\nUnit: %")]
    mock_json(monkeypatch, {"experiments": [{
        "name": "zero result", "metric": "error rate", "value": 0, "unit": "%",
        "field_evidence": {
            "metric": [{"id": "ev-a", "quote": "Metric: error rate"}],
            "value": [{"id": "ev-a", "quote": "Value: 0"}],
            "unit": [{"id": "ev-a", "quote": "Unit: %"}],
        },
    }]})
    result = asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert len(result) == 1
    assert result[0]["value"] == 0
    assert result[0]["evidence_ids"] == ["ev-a"]
    assert result[0]["field_evidence"]["value"] == ["ev-a"]


def test_extraction_accepts_values_with_their_own_evidence(monkeypatch, evidence_factory):
    evidence = [evidence_factory("AlphaNet on DemoSet reached accuracy 85% on test-A with batch size 32.")]

    def field(value, quote):
        return {"value": value, "evidence": [{"id": "S1", "quote": quote}]}

    mock_json(monkeypatch, {"experiments": [{
        "name": "measured experiment",
        "method": field("AlphaNet", "AlphaNet on DemoSet"),
        "dataset": field("DemoSet", "AlphaNet on DemoSet"),
        "split": field("test-A", "on test-A"),
        "metric": field("accuracy", "accuracy 85%"),
        "value": field(85, "accuracy 85%"),
        "unit": field("%", "accuracy 85%"),
        "conditions": field({"batch size": "32"}, "batch size 32"),
    }]})
    result = asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert len(result) == 1
    assert result[0]["dataset"] == "DemoSet"
    assert result[0]["value"] == 85
    assert result[0]["conditions"] == {"batch size": "32"}
    assert set(result[0]["field_evidence"]) == {"method", "dataset", "split", "metric", "value", "unit", "conditions"}
    assert result[0]["evidence_ids"] == ["ev-a"]


def test_extraction_pdf_typography_does_not_invalidate_real_evidence(monkeypatch, evidence_factory):
    evidence = [evidence_factory(
        "Amber Nor-\nmalizer achieved classiﬁcation accuracy 85% on DemoSet."
    )]
    mock_json(monkeypatch, {"experiments": [{
        "method": {"value": "Amber Normalizer", "evidence": [{
            "id": "S1", "quote": "Amber Normalizer achieved classification accuracy 85%",
        }]},
        "value": {"value": 85, "evidence": [{"id": "S1", "quote": "accuracy 85%"}]},
    }]})
    result = asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert result[0]["method"] == "Amber Normalizer"
    assert result[0]["value"] == 85


def test_extraction_pdf_normalization_does_not_accept_changed_results(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Amber Nor-\nmalizer achieved classiﬁcation accuracy 85%.")]
    mock_json(monkeypatch, {"experiments": [{
        "value": {"value": 99, "evidence": [{"id": "S1", "quote": "classification accuracy 99%"}]},
    }]})
    with pytest.raises(ProviderError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def test_field_value_cannot_be_assembled_across_separate_quotes(monkeypatch, evidence_factory):
    evidence = [evidence_factory("AlphaNet is one method. BetaNet is a separate method.")]
    mock_json(monkeypatch, {"experiments": [{
        "method": {"value": "AlphaNetBetaNet", "evidence": [
            {"id": "S1", "quote": "AlphaNet"}, {"id": "S1", "quote": "BetaNet"},
        ]},
    }]})
    with pytest.raises(ProviderError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def test_llm_extraction_rejects_boolean_as_a_measurement(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Value: 0")]
    mock_json(monkeypatch, {"experiments": [{
        "name": "invalid boolean", "value": False,
        "field_evidence": {"value": [{"id": "ev-a", "quote": "Value: 0"}]},
    }]})
    with pytest.raises(ValueError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def extracted_value(value, source="S1", name="measured result"):
    return {"name": name, "value": value,
            "field_evidence": {"value": [{"id": source, "quote": f"Value: {value}"}]}}


def test_extraction_short_ids_resolve_within_each_batch(monkeypatch, evidence_factory):
    evidence = [
        evidence_factory("Value: 85\n" + "a" * 12000, "persistent-first"),
        evidence_factory("Value: 86\n" + "b" * 12000, "persistent-second"),
    ]
    calls = []

    async def reply(messages, config):
        source = json.loads(messages[-1]["content"])["source_data"]
        calls.append(source)
        assert source[0]["id"] == "S1"
        return {"experiments": [extracted_value(85 if len(calls) == 1 else 86)]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert len(calls) == 2
    assert [item["evidence_ids"] for item in result] == [["persistent-first"], ["persistent-second"]]
    assert [item["field_evidence"]["value"] for item in result] == [["persistent-first"], ["persistent-second"]]
    assert len({item["id"] for item in result}) == 2
    assert evidence[0]["id"] == "persistent-first"


def test_extraction_correction_replaces_duplicates_and_keeps_invalid_rows_out(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        source = json.loads(messages[-1]["content"])
        calls.append(source)
        if len(calls) == 1:
            return {"experiments": [extracted_value(85, name="discarded"), extracted_value(99)]}
        assert "validation_feedback" in source
        assert source["source_data"] == calls[0]["source_data"]
        return {"experiments": [extracted_value(85, name="corrected")]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.extract_llm([evidence_factory("Value: 85")], "paper-a", CONFIG))
    assert len(calls) == 2
    assert [item["name"] for item in result] == ["corrected"]
    assert result[0]["evidence_ids"] == ["ev-a"]


def test_extraction_quarantines_bad_rows_and_reports_partial_results(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        return {"experiments": [extracted_value(85), extracted_value(99)]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.extract_llm([evidence_factory("Value: 85")], "paper-a", CONFIG))
    assert len(calls) == 2
    assert len(result) == 1
    assert result[0]["value"] == 85
    assert any("1 条候选" in note and "已排除" in note for note in result[0]["notes"])


def test_extraction_transport_failure_after_a_valid_batch_is_still_atomic(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        if len(calls) == 1:
            return {"experiments": [extracted_value(85)]}
        raise ProviderError("connection", "无法连接模型服务。")

    monkeypatch.setattr(providers, "_chat_json", reply)
    evidence = [evidence_factory("Value: 85\n" + "a" * 12000, "first"), evidence_factory("Value: 86", "second")]
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert caught.value.code == "connection"
    assert len(calls) == 2


@pytest.mark.parametrize("value,quote,expected", [
    (1.5, "Measured value is −1.5.", False),
    (-1.5, "Measured value is −1.5.", True),
    (1.5, "Measured value is − 1.5.", False),
    (1.5, "Measured value is -1.5.", False),
    (1, "F1 is 0.85.", False),
    (0.001, "Measured value is 1e−3.", True),
])
def test_numeric_grounding_preserves_signs_and_number_boundaries(value, quote, expected):
    assert providers._field_is_literal("value", value, [], [{"quote": quote}]) is expected


def test_extraction_rejects_a_condition_name_not_present_in_its_evidence(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Value: 85 with batch size 32.")]
    raw = extracted_value(85)
    raw["conditions"] = {"value": {"random seed": "32"}, "evidence": [{"id": "S1", "quote": "batch size 32"}]}
    mock_json(monkeypatch, {"experiments": [raw]})
    with pytest.raises(ProviderError):
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))


def test_optional_enrichment_fills_identity_without_changing_measurement(monkeypatch, evidence_factory):
    sources = [evidence_factory("AlphaNet on DemoSet measured accuracy 85%.")]
    base = providers._checked_extraction({"experiments": [extracted_value(85)]},
                                        [evidence_factory("Value: 85")], "paper-a", {"S1": "ev-a"}, 0)[0]
    base["metric"] = "accuracy"
    mock_json(monkeypatch, {"experiments": [{
        "record_id": "R1", "value": {"value": 99, "evidence": []}, "metric": "invented metric",
        "method": {"value": "AlphaNet", "evidence": [{"id": "S1", "quote": "AlphaNet on DemoSet"}]},
        "dataset": {"value": "DemoSet", "evidence": [{"id": "S1", "quote": "AlphaNet on DemoSet"}]},
    }]})
    asyncio.run(providers._enrich_extraction_fields([base], sources, CONFIG))
    assert base["method"] == "AlphaNet"
    assert base["dataset"] == "DemoSet"
    assert base["value"] == 85
    assert base["metric"] == "accuracy"
    assert base["evidence_ids"] == ["ev-a"]


def test_optional_enrichment_cannot_reuse_another_records_source(monkeypatch, evidence_factory):
    sources = [evidence_factory("AlphaNet: Value: 85", "own"), evidence_factory("BetaNet: Value: 86", "other")]
    rows = [providers._checked_extraction({"experiments": [extracted_value(value)]}, [source], "paper-a", {"S1": source["id"]}, 0)[0]
            for value, source in zip([85, 86], sources)]
    mock_json(monkeypatch, {"experiments": [{
        "record_id": "R1", "method": {"value": "BetaNet", "evidence": [{"id": "S1", "quote": "BetaNet: Value: 86"}]},
    }]})
    asyncio.run(providers._enrich_extraction_fields(rows, sources, CONFIG))
    assert rows[0]["method"] is None
    assert rows[0]["value"] == 85
    assert rows[0]["evidence_ids"] == ["own"]
    assert any("补充字段未通过" in note for note in rows[0]["notes"])


def test_optional_enrichment_failure_preserves_checked_measurements(monkeypatch, evidence_factory):
    sources = [evidence_factory("Value: 85")]
    base = providers._checked_extraction({"experiments": [extracted_value(85)]}, sources, "paper-a", {"S1": "ev-a"}, 0)[0]

    async def failed(messages, config):
        raise ProviderError("connection", "无法连接模型服务。")

    monkeypatch.setattr(providers, "_chat_json", failed)
    asyncio.run(providers._enrich_extraction_fields([base], sources, CONFIG))
    assert base["value"] == 85
    assert base["method"] is None
    assert any("可选字段补全未完成" in note for note in base["notes"])


@pytest.mark.parametrize("record_id", [[], {}, None, 1])
def test_optional_enrichment_ignores_malformed_record_ids(monkeypatch, evidence_factory, record_id):
    sources = [evidence_factory("Value: 85")]
    base = providers._checked_extraction({"experiments": [extracted_value(85)]}, sources, "paper-a", {"S1": "ev-a"}, 0)[0]
    mock_json(monkeypatch, {"experiments": [{"record_id": record_id, "method": {"value": "invented", "evidence": []}}]})
    asyncio.run(providers._enrich_extraction_fields([base], sources, CONFIG))
    assert base["value"] == 85
    assert base["method"] is None


@pytest.mark.parametrize("value,quote,expected", [("10", "sample size 100", False),
                                                ("1000", "sample size 1,000", True),
                                                ("0", "sample size 1e-11", False)])
def test_sample_size_uses_complete_exact_number_tokens(value, quote, expected):
    assert providers._field_is_literal("sample_size", value, [], [{"quote": quote}]) is expected


@pytest.mark.parametrize("reference", [
    {"id": "S999", "quote": "Value: 85"},
    {"id": "S1", "quote": "Value: 99"},
])
def test_extraction_correction_keeps_strict_grounding_and_is_bounded(monkeypatch, evidence_factory, reference):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        return {"experiments": [{"value": 85, "field_evidence": {"value": [reference]}}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers.extract_llm([evidence_factory("Value: 85")], "paper-a", CONFIG))
    assert caught.value.code == "grounding"
    assert len(calls) == 2


def test_extraction_does_not_retry_authentication_failures(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        raise ProviderError("authentication", "模型服务拒绝认证。")

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError):
        asyncio.run(providers.extract_llm([evidence_factory("Value: 85")], "paper-a", CONFIG))
    assert len(calls) == 1


def test_extraction_correction_identifies_all_invalid_fields(monkeypatch, evidence_factory):
    calls = []

    async def reply(messages, config):
        source = json.loads(messages[-1]["content"])
        calls.append(source)
        if len(calls) == 1:
            invalid = extracted_value(85)
            invalid.update(method="InventedNet", dataset="InventedSet")
            return {"experiments": [invalid]}
        assert "method" in source["validation_feedback"]
        assert "dataset" in source["validation_feedback"]
        assert source["previous_output"]["experiments"][0]["method"] == "InventedNet"
        return {"experiments": [extracted_value(85)]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.extract_llm([evidence_factory("Value: 85")], "paper-a", CONFIG))
    assert len(calls) == 2
    assert result[0]["method"] is None
    assert result[0]["dataset"] is None


def test_empty_extraction_is_explicit_and_does_not_return_replacement_records(monkeypatch, evidence_factory):
    mock_json(monkeypatch, {"experiments": []})
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers.extract_llm([evidence_factory("No experimental measurement.")], "paper-a", CONFIG))
    assert caught.value.code == "extraction_empty"


def mock_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize("status,code", [(401, "authentication"), (429, "quota"), (503, "http_error")])
def test_transport_errors_are_specific_and_do_not_echo_provider_body(monkeypatch, status, code):
    mock_transport(monkeypatch, lambda request: httpx.Response(
        status, json={"error": {"message": "private-key-marker full private prompt"}}))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers._chat_json([], CONFIG))
    assert caught.value.code == code
    assert "private-key-marker" not in caught.value.public_message
    assert "private prompt" not in caught.value.public_message


def test_transport_reports_output_truncation_before_parsing_json(monkeypatch):
    mock_transport(monkeypatch, lambda request: httpx.Response(200, json={"choices": [
        {"finish_reason": "length", "message": {"content": '{"experiments":['}},
    ]}))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers._chat_json([], CONFIG))
    assert caught.value.code == "output_truncated"


@pytest.mark.parametrize("choice", [{"message": None}, {"message": []}, None])
def test_transport_rejects_malformed_message_objects(monkeypatch, choice):
    mock_transport(monkeypatch, lambda request: httpx.Response(200, json={"choices": [choice]}))
    with pytest.raises(ProviderError) as caught:
        asyncio.run(providers._chat_json([], CONFIG))
    assert caught.value.code == "output_format"


def test_transport_keeps_compatible_json_mode_fallback(monkeypatch):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": "response_format is unsupported"}})
        return httpx.Response(200, json={"choices": [
            {"finish_reason": "stop", "message": {"content": '{"ok":true}'}},
        ]})

    mock_transport(monkeypatch, handle)
    result = asyncio.run(providers._chat_json([], CONFIG))
    assert result == {"ok": True}
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
