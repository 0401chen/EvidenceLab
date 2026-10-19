"""Offline, source-backed regression cases; these do not call a model service."""
import asyncio

import pytest

from backend.app import providers
from backend.app.analysis import extract_experiments
from backend.app.errors import ProviderError
from backend.app.grounding import number_in_text, quote_matches_source


@pytest.mark.parametrize("value,source,expected", [
    (0, "Value: 1e-11", False),
    (1e-11, "Value: 2e-11", False),
    (1e-11, "Value: 1e-11", True),
    (0.5, "Value: .5", True),
    (-0.5, "Value: − .5", True),
    (0, "Value: 0", True),
    (0.001, "Value: 1e−3", True),
    (1200, "Value: 1,200", True),
    (12, "Value: 1,2", False),
    (1, "Value: 1e-3", False),
    (3, "Value: 1.2.3", False),
    (0.0012, "Value: 1.2 × 10^{-3}", True),
    (0.0012, "Value: 1.2 × 10⁻³", True),
    (1.2, "Value: 1.2 × 10^-3", False),
    (1.2, "Value: 1.2 × 10-3", False),
])
def test_numbers_are_exact_complete_source_tokens(value, source, expected):
    assert number_in_text(value, source) is expected


def field(value, quote, source="S1"):
    return {"value": value, "evidence": [{"id": source, "quote": quote}]}


def checked(raw, evidence, paper_id="paper-a"):
    return providers._checked_extraction({"experiments": [raw]}, evidence, paper_id,
                                         {f"S{i + 1}": e["id"] for i, e in enumerate(evidence)}, 0)


def test_units_cannot_be_substrings_of_ordinary_words():
    assert not providers._field_is_literal("unit", "m", [], [{"quote": "measured value 85"}])
    assert providers._field_is_literal("unit", "m", [], [{"quote": "length 85 m"}])
    assert not providers._field_is_literal("unit", "mM", [], [{"quote": "85 Mm"}])


def test_unit_cannot_come_from_another_result_in_the_same_block(evidence_factory):
    evidence = [evidence_factory("Alpha got accuracy 85. Beta got accuracy 99%.")]
    raw = {"metric": field("accuracy", "Alpha got accuracy 85"),
           "value": field(85, "Alpha got accuracy 85"),
           "unit": field("%", "Beta got accuracy 99%")}
    with pytest.raises(ProviderError, match="unit"):
        checked(raw, evidence)


def test_method_quote_for_a_different_explicit_measurement_is_rejected(evidence_factory):
    evidence = [evidence_factory("Alpha has accuracy 85%. Beta has accuracy 99%.")]
    raw = {"method": field("Alpha", "Alpha has accuracy 85%"),
           "metric": field("accuracy", "Beta has accuracy 99%"),
           "value": field(99, "Beta has accuracy 99%"),
           "unit": field("%", "Beta has accuracy 99%")}
    with pytest.raises(ProviderError, match="method"):
        checked(raw, evidence)


def test_grounded_task_and_field_quotes_preserve_actual_source_pages(evidence_factory):
    evidence = [evidence_factory("Image classification: Alpha on Demo got accuracy 85%.")]
    evidence[0]["page"] = 7
    raw = {"task": field("Image classification", "Image classification: Alpha"),
           "method": field("Alpha", "Alpha on Demo"),
           "metric": field("accuracy", "accuracy 85%"),
           "value": field(85, "accuracy 85%"), "unit": field("%", "accuracy 85%")}
    item = checked(raw, evidence)[0]
    assert item["task"] == "Image classification"
    assert item["source_pages"] == [7]
    assert item["review_required"] is True
    assert item["split"] is None
    for field_name, refs in item["evidence_quotes"].items():
        assert [ref["id"] for ref in refs] == item["field_evidence"][field_name]
        assert all(quote_matches_source(ref["quote"], evidence[0]["text"]) for ref in refs)


@pytest.mark.parametrize("page", [None, 0, -1, "7", True])
def test_extraction_never_invents_a_source_page(evidence_factory, page):
    evidence = [evidence_factory("Value: .5")]
    evidence[0]["page"] = page
    assert checked({"value": field(0.5, "Value: .5")}, evidence)[0]["source_pages"] == []


@pytest.mark.parametrize("source,value,unit", [
    ("RMSE was 1e-3", 0.001, None),
    ("MAE is −1.5e−2", -0.015, None),
    ("accuracy .5%", 0.5, "%"),
    ("MSE 0", 0, None),
    ("RMSE 1.2 × 10^-3", 0.0012, None),
    ("RMSE 1.2 × 10⁻³", 0.0012, None),
])
def test_prose_rules_preserve_scientific_numbers_and_original_quotes(evidence_factory, source, value, unit):
    item = extract_experiments([evidence_factory(source)], "paper-a")[0]
    assert item["value"] == value
    assert item["unit"] == unit
    assert item["review_required"] is True
    for refs in item["evidence_quotes"].values():
        assert all(quote_matches_source(ref["quote"], source) for ref in refs)


def test_rule_unit_conflict_uses_literal_numeric_suffix_and_requests_review(evidence_factory):
    source = "Method: Alpha\nMetric: accuracy\nValue: 85%\nUnit: fraction"
    item = extract_experiments([evidence_factory(source)], "paper-a")[0]
    assert item["value"] == 85
    assert item["unit"] == "%"
    assert any("单位字段与数值后缀不一致" in note for note in item["notes"])
    assert item["evidence_quotes"]["unit"][0]["quote"] == "Value: 85%"


def test_rules_do_not_guess_an_ambiguous_power_expression(evidence_factory):
    assert extract_experiments([evidence_factory("RMSE 1.2 × 10-3")], "paper-a") == []


def test_small_measurement_is_not_silently_replaced_by_zero(monkeypatch, evidence_factory):
    async def bad_model(messages, config):
        return {"experiments": [{"value": field(0, "Value: 1e-11")} ]}

    monkeypatch.setattr(providers, "_chat_json", bad_model)
    with pytest.raises(ProviderError, match="value"):
        asyncio.run(providers.extract_llm([evidence_factory("Value: 1e-11")], "paper-a",
                                         {"model": "offline-fixture", "enrich_metadata": False}))
