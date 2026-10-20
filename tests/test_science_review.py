"""Offline regressions for source association, retrieval and provider failures.

These deliberately synthetic examples establish program invariants only; they
are not a benchmark of extraction recall or a paid model acceptance test.
"""
import asyncio
import json

import pytest

from backend.app import providers
from backend.app.analysis import (answer_local, extract_experiments, retrieve_evidence,
                                  select_evidence_context)
from backend.app.errors import ProviderError
from backend.app.orchestration import run_comparison
from backend.app.parsing import parse_document


# These saved V1 response fixtures exercise the explicit compatibility seam.
CONFIG = {"model": "offline-fixture", "base_url": "http://offline.invalid/v1", "enrich_metadata": False}


def field(value, quote, source="S1"):
    return {"value": value, "evidence": [{"id": source, "quote": quote}]}


def test_chinese_model_configuration_question_retrieves_english_results(evidence_factory):
    evidence = [
        evidence_factory("Our method includes a normalization step.", "method"),
        evidence_factory("A single model has test error 7.3%. An ensemble has test error 6.8%.", "result"),
        evidence_factory("Related methods in the literature.", "reference"),
    ]
    evidence[2]["section"] = "References"
    result = answer_local("单模型和集成模型分别报告了什么错误率？", evidence)
    assert result["verdict"] == "evidence_found"
    assert [row["id"] for row in result["citations"]] == ["result"]


def test_generic_model_word_and_bibliography_do_not_imply_relevant_results(evidence_factory):
    evidence = [evidence_factory("The method works with a neural model.", "method"),
                evidence_factory("Ensemble methods and classification error. A bibliography entry.", "reference")]
    evidence[1]["section"] = "References"
    result = answer_local("单模型和集成模型的错误率是多少？", evidence)
    assert result["verdict"] == "insufficient"
    assert result["citations"] == []


def test_retrieval_retains_a_later_counter_result_and_each_paper(evidence_factory):
    evidence = [evidence_factory("Catalyst efficiency is 84 percent under fixed conditions.", f"repeat-{index}")
                for index in range(25)]
    evidence.append(evidence_factory("Catalyst efficiency is not 84 percent; measured efficiency was 62 percent.", "counter"))
    evidence.append(evidence_factory("Catalyst efficiency was 71 percent.", "other-paper", "paper-b"))
    found = retrieve_evidence("Is catalyst efficiency 84 percent?", evidence, limit=6)
    assert {"counter", "other-paper"} <= {e["id"] for e in found}


def test_model_context_does_not_admit_unrelated_adjacent_passages(evidence_factory):
    evidence = [evidence_factory("Catalyst efficiency is 84 percent.", "own"),
                evidence_factory("A lunar telescope uses optical mirrors.", "unrelated")]
    context, diagnostic = select_evidence_context("Catalyst efficiency?", evidence)
    assert [e["id"] for e in context] == ["own"]
    assert diagnostic["total"] == 2


def test_context_budget_is_explicit_without_truncating_a_quote(evidence_factory):
    evidence = [evidence_factory(f"Catalyst efficiency is {index} percent.", str(index)) for index in range(25)]
    context, diagnostic = select_evidence_context("Catalyst efficiency?", evidence, limit=3, max_chars=120)
    assert diagnostic["coverage_limited"] is True
    assert len(context) <= 3
    assert all(e in evidence for e in context)


def test_equal_measurements_in_distinct_sources_are_not_merged(evidence_factory):
    evidence = [evidence_factory("Method: Alpha\nMetric: accuracy\nValue: 82%", "explicit"),
                evidence_factory("Beta has accuracy 82% on an independent evaluation.", "other")]
    records = extract_experiments(evidence, "paper-a")
    assert len(records) == 2
    assert {tuple(e["evidence_ids"]) for e in records} == {("explicit",), ("other",)}


def test_demo_restated_results_do_not_create_extra_experiments_or_comparisons():
    from pathlib import Path

    evidence, experiments = [], []
    for path in sorted((Path(__file__).resolve().parents[1] / "examples").glob("study_*.md")):
        parsed = parse_document(path, path.stem)
        rows = extract_experiments(parsed["evidence"], path.stem)
        assert len(rows) == 1
        assert rows[0]["method"] and rows[0]["dataset"] and rows[0]["split"]
        assert any("未额外创建实验" in note for note in rows[0]["notes"])
        assert all(not key.startswith("_candidate") for key in rows[0])
        evidence.extend(parsed["evidence"])
        experiments.extend(rows)
    report = run_comparison(experiments, evidence)
    assert len(report["comparisons"]) == 3
    assert len(report["traces"]) == 4


@pytest.mark.parametrize("prose,page", [
    ("Beta reports accuracy 82%.", 1),
    ("Alpha reports accuracy 82%.", 2),
    ("Alpha on Data-B reports accuracy 82%.", 1),
    ("Alpha on test-B reports accuracy 82%.", 1),
    ("Alpha and Beta report accuracy 82%.", 1),
    ("An independent Alpha evaluation has accuracy 82%.", 1),
])
def test_different_or_ambiguous_weak_results_are_not_suppressed(evidence_factory, prose, page):
    evidence = [evidence_factory("Method: Alpha\nDataset: Data-A\nSplit: test-A\nMetric: accuracy\nValue: 82%", "explicit"),
                evidence_factory(prose, "weak", page=page)]
    rows = extract_experiments(evidence, "paper-a")
    assert len(rows) == 2
    assert {tuple(row["evidence_ids"]) for row in rows} == {("explicit",), ("weak",)}


def test_weak_restatement_of_multiple_possible_splits_remains_a_candidate(evidence_factory):
    evidence = [evidence_factory("Method: Alpha\nDataset: Data-A\nSplit: test-A\nMetric: accuracy\nValue: 82%", "one"),
                evidence_factory("Alpha reports accuracy 82%.", "weak"),
                evidence_factory("Method: Alpha\nDataset: Data-A\nSplit: test-B\nMetric: accuracy\nValue: 82%", "two")]
    assert len(extract_experiments(evidence, "paper-a")) == 3


def test_llm_duplicate_identity_includes_result_provenance(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Value: 82", "one"), evidence_factory("Value: 82", "two")]

    async def reply(messages, config):
        return {"experiments": [{"value": field(82, "Value: 82", "S1")},
                                {"value": field(82, "Value: 82", "S2")}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    records = asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert len(records) == 2
    assert len({e["id"] for e in records}) == 2


def test_rule_table_keeps_model_rows_metric_columns_and_units_separate(evidence_factory):
    source = evidence_factory("| Model | Dataset | Accuracy (%) | Error rate (%) |\n"
                              "| --- | --- | --- | --- |\n"
                              "| Alpha | Data-A | 82 | 18 |\n"
                              "| Beta | Data-B | 82 | 18 |", "table")
    source["source_type"] = "table"
    rows = extract_experiments([source], "paper-a")
    assert len(rows) == 4
    assert {(r["method"], r["dataset"], r["metric"], r["value"], r["unit"]) for r in rows} == {
        (model, dataset, metric, value, "%") for model, dataset in (("Alpha", "Data-A"), ("Beta", "Data-B"))
        for metric, value in (("accuracy", 82), ("error_rate", 18))}
    assert all(row["split"] is None and row["conditions"] == {} for row in rows)


def test_error_metrics_are_recognized_without_collapsing_topk(evidence_factory):
    rows = extract_experiments([evidence_factory("Top-1 error was 7.3%. Top-5 error was 2.1%.")], "paper-a")
    assert [(r["metric"], r["value"]) for r in rows] == [("top1_error", 7.3), ("top5_error", 2.1)]


def test_topk_retrieval_and_extraction_support_arbitrary_k_and_explicit_split_word(evidence_factory):
    evidence = [evidence_factory("A single model has top-2 validation error 4.1%.")]
    assert retrieve_evidence("top-2错误率是多少？", evidence)
    assert extract_experiments(evidence, "paper-a")[0]["metric"] == "top2_error"


def test_same_measurement_in_different_configurations_keeps_own_sentence(evidence_factory):
    rows = extract_experiments([evidence_factory("A single model has accuracy 82%. An ensemble has accuracy 82%.")], "paper-a")
    assert [r["conditions"] for r in rows] == [{"single model": "single model"}, {"ensemble": "ensemble"}]
    assert "single model" in rows[0]["evidence_quotes"]["conditions"][0]["quote"]
    assert "ensemble" in rows[1]["evidence_quotes"]["conditions"][0]["quote"]


def test_ambiguous_single_vs_ensemble_sentence_does_not_fill_configuration(evidence_factory):
    rows = extract_experiments([evidence_factory("For single model and ensemble settings we report accuracy 82%.")], "paper-a")
    assert rows[0]["conditions"] == {}


def test_native_pdf_tables_preserve_explicit_rows(tmp_path):
    import pymupdf

    path = tmp_path / "ruled-table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    x_values, y_values = [60, 180, 300, 440], [80, 110, 140, 170]
    for x in x_values:
        page.draw_line((x, 80), (x, 170))
    for y in y_values:
        page.draw_line((60, y), (440, y))
    for row_index, row in enumerate((("Model", "Dataset", "Accuracy (%)"), ("Alpha", "Data-A", "82"), ("Beta", "Data-B", "79"))):
        for column, text in enumerate(row):
            page.insert_text((x_values[column] + 6, y_values[row_index] + 20), text, fontsize=10)
    doc.save(path)
    doc.close()
    parsed = parse_document(path, "paper-table")
    tables = [e for e in parsed["evidence"] if e["source_type"] == "table"]
    assert tables and all(e["page"] == 1 and len(e["bbox"]) == 4 for e in tables)
    rows = extract_experiments(tables, "paper-table")
    assert {(e["method"], e["dataset"], e["value"]) for e in rows} == {("Alpha", "Data-A", 82), ("Beta", "Data-B", 79)}


def test_llm_table_unit_must_come_from_the_matching_metric_column(evidence_factory):
    source = evidence_factory("| Model | Accuracy (%) | RMSE |\n| --- | --- | --- |\n| Alpha | 82 | 18 |")
    source["source_type"] = "table"
    rows = providers._checked_extraction({"experiments": [{
        "metric": field("accuracy", "| Model | Accuracy (%) | RMSE |"),
        "value": field(82, "| Alpha | 82 | 18 |"),
        "unit": field("%", "| Model | Accuracy (%) | RMSE |"),
    }]}, [source], "paper-a", {"S1": source["id"]}, 0)
    assert rows[0]["unit"] == "%"
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [{
            "metric": field("RMSE", "| Model | Accuracy (%) | RMSE |"),
            "value": field(18, "| Alpha | 82 | 18 |"),
            "unit": field("%", "| Model | Accuracy (%) | RMSE |"),
        }]}, [source], "paper-a", {"S1": source["id"]}, 0)


def test_cross_page_context_reaches_missing_configuration_without_unlocking_result(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Table 2 reports Alpha on Data-A, test split held-out; ensemble of five models.", "setup", page=2),
                evidence_factory("Table 2: Alpha accuracy 82%.", "result", page=3)]
    base = providers._checked_extraction({"experiments": [{"metric": field("accuracy", "Alpha accuracy 82%"),
                                                          "value": field(82, "Alpha accuracy 82%"),
                                                          "unit": field("%", "Alpha accuracy 82%")}]},
                                         [evidence[1]], "paper-a", {"S1": "result"}, 0)[0]

    async def reply(messages, config):
        record = json.loads(messages[-1]["content"])["records"][0]
        assert record["locked_result"]["value"] == 82
        source = next(e for e in record["source_data"] if e["id"] != record["result_source_ids"][0])
        return {"experiments": [{"record_id": record["record_id"],
                                 "split": field("held-out", "test split held-out", source["id"]),
                                 "conditions": field({"ensemble": "five models"}, "ensemble of five models", source["id"]),
                                 "value": field(99, "invented")}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    asyncio.run(providers._enrich_extraction_fields([base], evidence, CONFIG))
    assert base["value"] == 82
    assert base["split"] == "held-out"
    assert base["conditions"] == {"ensemble": "five models"}
    assert base["source_pages"] == [2, 3]


def test_batch_boundary_has_bounded_neighbouring_context(evidence_factory):
    evidence = [evidence_factory("a" * 9800, "first"), evidence_factory("The evaluation protocol is fixed.", "setup"),
                evidence_factory("Accuracy was 82%." + "b" * 500, "result", page=2)]
    batches = providers._extraction_batches(evidence)
    assert len(batches) == 2
    assert {"setup", "result"} <= {e["id"] for e in batches[1]}
    assert all(sum(len(e["text"]) for e in batch) <= 12000 for batch in batches)


def test_dataset_cannot_be_a_reported_metric_and_value(evidence_factory):
    source = evidence_factory("Ratio: 0.45. Accuracy 82%.")
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [{"value": field(82, "Accuracy 82%"),
                                                        "dataset": field("Ratio: 0.45", "Ratio: 0.45")}]},
                                      [source], "paper-a", {"S1": source["id"]}, 0)


def test_illustrative_chat_and_reference_are_not_extraction_sources(monkeypatch, evidence_factory):
    evidence = [evidence_factory("Buyer: Value: 0.45\nSeller: Value: 0.52", "chat"),
                evidence_factory("Method Alpha. Accuracy 82%.", "bibliography")]
    evidence[1]["section"] = "References"

    async def unexpected(*args):
        pytest.fail("Excluded examples and bibliography must not trigger model calls")

    monkeypatch.setattr(providers, "_chat_json", unexpected)
    with pytest.raises(ProviderError) as error:
        asyncio.run(providers.extract_llm(evidence, "paper-a", CONFIG))
    assert error.value.code == "extraction_empty"


def test_condition_key_and_value_need_the_same_quote_and_exact_number():
    assert not providers._field_is_literal("conditions", {"batch size": "32"}, [], [{"quote": "batch size 64"}, {"quote": "random seed 32"}])
    assert not providers._field_is_literal("conditions", {"batch size": "3"}, [], [{"quote": "batch size 32"}])
    assert not providers._field_is_literal("method", "Net", [], [{"quote": "AlphaNet accuracy was measured"}])


def test_comparison_requires_all_populated_protocol_fields(experiment_factory, evidence_factory):
    rows = [experiment_factory(), experiment_factory("exp-b", "paper-b", value=85)]
    evidence = [evidence_factory("Method: AlphaNet\nValue: 90", "ev-exp-a"),
                evidence_factory("Method: AlphaNet\nValue: 85", "ev-exp-b", "paper-b")]
    result = run_comparison(rows, evidence)["comparisons"][0]
    assert result["comparability"] == "insufficient"
    assert result["delta"] is None
    assert any("conditions" in reason and "dataset" in reason for reason in result["reasons"])



def protocol_result(operation, identifier="ev-fixed", quote="Catalyst efficiency is 84 percent."):
    references = [{"id": identifier, "quote": quote}]
    if operation == "qa":
        return {"answer": "Unchecked free text is not rendered.", "assertions": [
            {"text": "报告的催化剂效率为84%。", "evidence": references}]}
    return {"verdict": "supported", "explanation": "该原文报告催化剂效率为84%。", "evidence": references}


def run_protocol_operation(operation, evidence):
    if operation == "qa":
        return asyncio.run(providers.answer_llm("Catalyst efficiency?", evidence, CONFIG))
    return asyncio.run(providers.verify_claim("Catalyst efficiency is 84 percent.", evidence, CONFIG))


@pytest.mark.parametrize("operation", ["qa", "verify"])
def test_unknown_citation_id_has_one_bounded_protocol_correction(monkeypatch, evidence_factory, operation):
    evidence = [evidence_factory("Catalyst efficiency is 84 percent.", "ev-fixed")]
    calls = []

    async def reply(messages, config):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        assert payload["available_evidence_ids"] == ["F1"]
        if len(calls) == 1:
            return protocol_result(operation, "unknown-id")
        assert payload["source_data"] == calls[0]["source_data"]
        assert payload["validation_feedback"]["error_category"] == "grounding"
        assert payload["untrusted_previous_output"] == protocol_result(operation, "unknown-id")
        assert "不得新增事实" in messages[0]["content"]
        assert "不可信" in messages[0]["content"]
        return protocol_result(operation)

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = run_protocol_operation(operation, evidence)
    assert len(calls) == 2
    assert result["citations"] == evidence
    assert any(trace["agent"] == "引用协议纠错" for trace in result["traces"])
    assert any("一次" in warning and "纠错" in warning for warning in result["warnings"])


@pytest.mark.parametrize("operation", ["qa", "verify"])
@pytest.mark.parametrize("second_reference", [("unknown-again", "Catalyst efficiency is 84 percent."),
                                              ("ev-fixed", "Catalyst efficiency is 99 percent.")])
def test_twice_invalid_citations_stay_rejected_without_guessing_ids(monkeypatch, evidence_factory, operation, second_reference):
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        return protocol_result(operation, *(second_reference if len(calls) > 1 else ("unknown-id", "Catalyst efficiency is 84 percent.")))

    monkeypatch.setattr(providers, "_chat_json", reply)
    with pytest.raises(ProviderError) as error:
        run_protocol_operation(operation, [evidence_factory("Catalyst efficiency is 84 percent.", "ev-fixed")])
    assert error.value.code == "grounding"
    assert "一次" in error.value.public_message and "拒绝" in error.value.public_message
    assert len(calls) == 2


@pytest.mark.parametrize("operation", ["qa", "verify"])
def test_invalid_structure_has_one_correction_with_the_same_sources(monkeypatch, evidence_factory, operation):
    calls = []

    async def reply(messages, config):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        if len(calls) == 1:
            return {"assertions": "wrong-type"} if operation == "qa" else {"verdict": []}
        assert payload["validation_feedback"]["error_category"] == "output_format"
        assert payload["source_data"] == calls[0]["source_data"]
        return protocol_result(operation)

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = run_protocol_operation(operation, [evidence_factory("Catalyst efficiency is 84 percent.", "ev-fixed")])
    assert len(calls) == 2 and result["citations"]


@pytest.mark.parametrize("operation", ["qa", "verify"])
@pytest.mark.parametrize("code", ["authentication", "quota", "connection", "timeout", "http_error"])
def test_protocol_correction_never_retries_provider_failures(monkeypatch, evidence_factory, operation, code):
    calls = []

    async def failed(messages, config):
        calls.append(messages)
        raise ProviderError(code, "受控离线错误。")

    monkeypatch.setattr(providers, "_chat_json", failed)
    with pytest.raises(ProviderError) as error:
        run_protocol_operation(operation, [evidence_factory("Catalyst efficiency is 84 percent.", "ev-fixed")])
    assert error.value.code == code
    assert len(calls) == 1
