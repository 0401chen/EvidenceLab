"""Explicit table-wide units must keep benchmark columns and primary cells."""

import pytest

from backend.app import providers
from backend.app.evidence_cells import table_cells, source_fragments


def fixture(caption="Table 2. Classification accuracy (%) on two benchmarks.", cell="73 ± 0.4"):
    row = f"| Atlas-A | {cell} | 81 |"
    table = {"id": "table", "paper_id": "paper", "page": 2, "source_type": "table",
             "text": "| Model | Bench-Q | Bench-R |\n| --- | --- | --- |\n" + row,
             "caption_evidence_ids": ["caption"]}
    caption = {"id": "caption", "paper_id": "paper", "page": 2, "source_type": "text", "text": caption}
    item = {"method": "Atlas-A", "dataset": "Bench-Q", "metric": "accuracy", "value": "73", "unit": "%",
            "field_evidence": {"value": ["table"]},
            "evidence_quotes": {"value": [{"id": "table", "quote": row}],
                                "unit": [{"id": "caption", "quote": caption["text"]}]}}
    return item, [table, caption]


def test_unique_linked_caption_applies_percent_to_exact_model_benchmark_cell(monkeypatch):
    monkeypatch.setattr(providers, "_table_result_anchors", lambda *_: [])
    item, evidence = fixture()
    assert providers._unit_matches_measurement(item, evidence)


@pytest.mark.parametrize("change", ["dataset", "method", "metric", "std", "unlinked", "foreign", "quote"])
def test_caption_cannot_assign_unit_to_another_row_column_metric_or_uncertainty(change):
    item, evidence = fixture()
    if change == "dataset":
        item["dataset"] = "Bench-R"
    elif change == "method":
        item["method"] = "Atlas-B"
    elif change == "metric":
        item["metric"] = "error"
    elif change == "std":
        item["value"] = "0.4"
    elif change == "unlinked":
        evidence[0]["caption_evidence_ids"] = []
    elif change == "foreign":
        evidence[1]["paper_id"] = "another-paper"
    else:
        item["evidence_quotes"]["value"][0]["quote"] = "| Atlas-B | 73 | 81 |"
    assert not providers._unit_matches_measurement(item, evidence)


@pytest.mark.parametrize("caption", [
    "Table 2. Classification accuracy and precision (%) on two benchmarks.",
    "Table 2. Classification accuracy (%) and execution time (ms) on two benchmarks.",
    "Table 2. Classification accuracy on two benchmarks.",
])
def test_multiple_metrics_units_or_no_explicit_percent_caption_stays_unknown(caption):
    item, evidence = fixture(caption)
    assert not providers._unit_matches_measurement(item, evidence)


def test_ambiguous_duplicate_benchmark_columns_do_not_select_a_cell():
    item, evidence = fixture(cell="73")
    evidence[0]["text"] = "| Model | Bench-Q | Bench-Q |\n| --- | --- | --- |\n| Atlas-A | 73 | 73 |"
    item["evidence_quotes"]["value"][0]["quote"] = "| Atlas-A | 73 | 73 |"
    assert not providers._unit_matches_measurement(item, evidence)


@pytest.mark.parametrize("unit", ["s", "sec", "ms"])
def test_explicit_time_annotation_keeps_center_separate_and_does_not_infer_its_role(unit):
    item, evidence = fixture(cell=f"73 (0.4 {unit})")
    _, fragments = source_fragments(evidence)
    cells = table_cells(evidence, fragments)
    selected = next(cell for cell in cells.values() if cell["column_label"] == "Bench-Q")
    assert selected["value"] == "73"
    assert selected["time_annotation"] == {"text": f"0.4 {unit}", "value": "0.4", "unit": unit}
    assert "training" not in selected and "inference" not in selected
    assert providers._unit_matches_measurement(item, evidence)
    item["value"] = "0.4"
    assert not providers._unit_matches_measurement(item, evidence)
    item["value"], item["dataset"] = "73", "Bench-R"
    assert not providers._unit_matches_measurement(item, evidence)


def test_unitless_parenthetical_pair_remains_unresolved():
    _, evidence = fixture(cell="73 (0.4)")
    _, fragments = source_fragments(evidence)
    assert all(cell["column_label"] != "Bench-Q" for cell in table_cells(evidence, fragments).values())


def test_spelled_out_percent_caption_and_explicit_time_annotation_keep_units_separate():
    item, evidence = fixture("Table 2. Classification accuracy (in percent); time in seconds is shown in brackets.", cell="73 (0.4 s)")
    assert providers._field_is_literal("unit", "%", [evidence[1]], item["evidence_quotes"]["unit"])
    assert providers._unit_matches_measurement(item, evidence)
    item["value"] = "0.4"
    assert not providers._unit_matches_measurement(item, evidence)


def test_mean_spread_percent_can_supply_unit_only_to_the_primary_result():
    item, evidence = fixture("Table 2. Classification accuracies on two benchmarks.", cell="73 ± 0.4%")
    item["evidence_quotes"]["unit"] = [{"id": "table", "quote": item["evidence_quotes"]["value"][0]["quote"]}]
    assert providers._unit_matches_measurement(item, evidence)
    item["value"] = "0.4"
    assert not providers._unit_matches_measurement(item, evidence)


def test_plural_accuracy_caption_is_a_narrow_metric_alias_with_original_quote_retained():
    _, evidence = fixture("Table 2. Classification accuracies on the benchmarks.")
    quote = {"id": "caption", "quote": evidence[1]["text"]}
    assert providers._field_is_literal("metric", "accuracy", [evidence[1]], [quote])
    assert quote["quote"] == evidence[1]["text"]


def test_literal_percent_elsewhere_in_a_table_cannot_bypass_exact_column_matching():
    item, evidence = fixture(cell="73%")
    item["evidence_quotes"]["unit"] = [{"id": "table", "quote": evidence[0]["text"]}]
    assert providers._unit_matches_measurement(item, evidence)
    item["dataset"] = "Bench-R"
    assert not providers._unit_matches_measurement(item, evidence)


def test_percent_on_training_labels_does_not_assign_a_unit_to_bare_accuracy_results():
    item, evidence = fixture("Table 2. Classification accuracy using 5% training labels.", cell="73")
    assert not providers._unit_matches_measurement(item, evidence)


def test_standalone_explicit_standard_deviation_remains_a_valid_percent_measurement():
    source = {"id": "std", "paper_id": "toy", "page": 1, "source_type": "text", "text": "The standard deviation was 0.7%."}
    item = {"metric": "standard deviation", "value": "0.7", "unit": "%", "field_evidence": {"value": ["std"]},
            "evidence_quotes": {"value": [{"id": "std", "quote": source["text"]}], "unit": [{"id": "std", "quote": source["text"]}]}}
    assert providers._unit_matches_measurement(item, [source])
