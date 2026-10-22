"""Public synthetic PDFs: bind model rows, dataset columns and caption metrics."""
from copy import deepcopy

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.experiment_validation import prepare_experiments
from backend.app.grounding import quote_matches_source
from backend.app.parsing import parse_document
from backend.app.providers import _unit_matches_measurement


def write_pdf(path, *, caption="Table 1: Classification accuracy (in percent) on the two benchmark datasets.",
              headers=("Method", "Bench-Q", "Bench-R"),
              rows=(("Network-A", "81.5", "78.1"), ("Network-B", "83.0 ± 0.7%", "80.2%"))):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 55), caption, fontsize=10)
    for y, cells in [(85, headers), *[(105 + i * 20, row) for i, row in enumerate(rows)]]:
        for x, text in zip((50, 210, 330), cells):
            if text:
                page.insert_text((x, y), text, fontsize=10)
    page.insert_text((50, 190), "Independent prose: mean accuracy of 100 runs with random initializations.", fontsize=10)
    if "benchmark datasets" in caption:
        page.insert_text((50, 250), "Bench-Q dataset and Bench-R dataset are evaluated.", fontsize=10)
    doc.save(path)
    doc.close()


def fixture(tmp_path, **options):
    path = tmp_path / "synthetic.pdf"
    write_pdf(path, **options)
    parsed = parse_document(path, "paper-synthetic")
    return parsed, extract_experiments(parsed["evidence"], "paper-synthetic")


def test_real_pdf_chain_keeps_each_model_dataset_and_complete_uncertainty(tmp_path):
    parsed, records = fixture(tmp_path)
    assert {(r["method"], r["dataset"], r["metric"], r["value"], r["unit"]) for r in records} == {
        ("Network-A", "Bench-Q", "accuracy", 81.5, "%"), ("Network-A", "Bench-R", "accuracy", 78.1, "%"),
        ("Network-B", "Bench-Q", "accuracy", 83.0, "%"), ("Network-B", "Bench-R", "accuracy", 80.2, "%")}
    prepared = prepare_experiments(records, {"id": "paper-synthetic", "title": "Synthetic"}, parsed["evidence"])
    source_map = {e["id"]: e for e in parsed["evidence"]}
    for record in prepared:
        assert record["review_required"] and record["extraction_method"] == "rule"
        assert record["split"] is None and record["task"] is None and record["conditions"] == {}
        for refs in record["evidence_quotes"].values():
            assert all(quote_matches_source(ref["quote"], source_map[ref["id"]]["text"]) for ref in refs)
        assert _unit_matches_measurement(record, parsed["evidence"])
        assert record["table_context"]["column_label"] == record["dataset"]
        assert record["table_context"]["row_label"] == record["method"]
        assert record["table_context"]["value_bbox"] and record["source_pages"] == [1]
    row = next(r for r in prepared if r["value"] == 83.0)
    assert row["table_context"]["value_cell_text"] == "83.0 ± 0.7%"
    assert row["table_context"]["uncertainty"] == {"notation": "±", "value": "0.7", "kind": "unspecified"}
    assert "83.0 ± 0.7%" in row["evidence_quotes"]["value"][0]["quote"]
    assert not any(r["value"] in {0.7, 100} for r in prepared)


@pytest.mark.parametrize("cell,value,annotation", [
    ("100%", 100, None), ("81.5 (4s)", 81.5, "time_annotation"),
    ("83.0 +/- 0.7%", 83.0, "uncertainty"), ("83.0 ± 0.7", 83.0, "uncertainty"),
])
def test_primary_number_never_becomes_time_or_spread(tmp_path, cell, value, annotation):
    parsed, records = fixture(tmp_path, rows=(("Network-A", cell, "79.1"), ("Network-B", "75", "80")))
    row = next(r for r in records if r["method"] == "Network-A" and r["dataset"] == "Bench-Q")
    assert row["value"] == value and row["unit"] == "%"
    assert row["table_context"]["value_cell_text"] == cell
    if annotation:
        assert annotation in row["table_context"]
    assert not any(r["value"] in {0.7, 4} for r in records)


@pytest.mark.parametrize("caption", [
    "Table 1: Results on the two benchmark datasets without a stated metric.",
    "Table 1: Accuracy and precision (%) on the two benchmark datasets.",
    "Table 1: Accuracy (%) and error (%) on the two benchmark datasets.",
    "Table 1: Accuracy (%) and latency (ms) on the two benchmark datasets.",
    "Classification accuracy (in percent) on the two benchmark datasets.",
])
def test_missing_or_ambiguous_caption_cannot_supply_a_metric(tmp_path, caption):
    _, records = fixture(tmp_path, caption=caption)
    assert records == []


@pytest.mark.parametrize("headers", [
    ("Method", "epochs", "runs"), ("Method", "learning rate", "dropout"),
    ("Method", "random seeds", "samples"), ("Method", "Bench-Q", "Bench-Q"),
    ("Method", "learning_rate", "batch_size"), ("Method", "weight_decay", "momentum"),
    ("Method", "Depth", "Layers"),
])
def test_configuration_or_duplicate_columns_are_not_dataset_results(tmp_path, headers):
    _, records = fixture(tmp_path, headers=headers)
    assert records == []


@pytest.mark.parametrize("bad_cell", ["", "83.0 (0.7)", "100 runs", "80-85", "83.0 ± -0.7%"])
def test_incomplete_or_unresolved_cell_never_shifts_into_another_dataset(tmp_path, bad_cell):
    _, records = fixture(tmp_path, rows=(("Network-A", bad_cell, "78.1"), ("Network-B", "76", "80")))
    assert records == []


def test_explicit_missing_placeholder_does_not_invent_an_experiment(tmp_path):
    # Base-14 PDF fonts do not portably encode an em dash. Use the literal
    # ASCII missing-value marker here; the real PDF retains its original dash.
    _, records = fixture(tmp_path, rows=(("Network-A", "-", "78.1"), ("Network-B", "76", "80")))
    assert len(records) == 3
    assert not any(r["method"] == "Network-A" and r["dataset"] == "Bench-Q" for r in records)


def test_shifted_column_cannot_be_recovered_by_number_order(tmp_path):
    path = tmp_path / "misaligned.pdf"
    write_pdf(path)
    with pymupdf.open(path) as doc:
        # A separate third row is outside the established result-column anchor.
        for x, text in ((50, "Network-C"), (250, "92.0"), (330, "94.0")):
            doc[0].insert_text((x, 145), text, fontsize=10)
        altered = tmp_path / "shifted.pdf"
        doc.save(altered)
    parsed = parse_document(altered, "paper-synthetic")
    assert extract_experiments(parsed["evidence"], "paper-synthetic") == []


def test_derived_difference_row_is_not_a_model_accuracy(tmp_path):
    _, records = fixture(tmp_path, rows=(("Network-A", "81.5", "78.1"),
                                        ("Network-B", "83.0", "80.2"), ("Difference", "1.5", "2.1")))
    assert len(records) == 4
    assert {r["method"] for r in records} == {"Network-A", "Network-B"}


@pytest.mark.parametrize("mutation", ["column", "row", "caption_page", "caption_paper", "layout", "duplicate_row"])
def test_forged_or_inconsistent_coordinate_links_are_rejected(tmp_path, mutation):
    parsed, _ = fixture(tmp_path)
    sources = deepcopy(parsed["evidence"])
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    caption = next(e for e in sources if e["id"] == table["caption_evidence_ids"][0])
    if mutation == "column":
        table["coordinate_rows"][0]["cells"][1]["column_label"] = "Bench-R"
    elif mutation == "row":
        table["coordinate_rows"][0]["row_label"] = "Network-B"
    elif mutation == "caption_page":
        caption["page"] = 2
    elif mutation == "caption_paper":
        caption["paper_id"] = "foreign"
    elif mutation == "duplicate_row":
        table["text"] = table["text"].replace("Network-B", "Network-A")
    else:
        table["coordinate_rows"][0]["cells"][1]["text"] = "99.9"
    assert extract_experiments(sources, "paper-synthetic") == []


def test_wrong_model_dataset_or_metric_still_fails_exact_cell_unit_check(tmp_path):
    parsed, rows = fixture(tmp_path)
    correct = next(r for r in rows if r["value"] == 83)
    for field, wrong in (("method", "Network-A"), ("dataset", "Bench-R"), ("metric", "error_rate"), ("value", 0.7)):
        invalid = {**correct, field: wrong}
        assert not _unit_matches_measurement(invalid, parsed["evidence"])


def test_rule_api_returns_pending_candidates_not_adopted_experiments(client, tmp_path):
    path = tmp_path / "source.pdf"
    write_pdf(path)
    response = client.post("/api/papers/upload", files={"files": ("synthetic.pdf", path.read_bytes(), "application/pdf")})
    paper = response.json()["papers"][0]
    before = client.get(f"/api/papers/{paper['id']}").json()
    result = client.post(f"/api/papers/{paper['id']}/extract", json={"mode": "rule", "reparse": False, "max_results": 30})
    assert result.status_code == 200
    after = result.json()
    assert after["experiments"] == before["experiments"]
    candidate = next(c for c in after["extraction_candidates"] if c["id"] == after["latest_extraction_candidate_id"])
    assert candidate["status"] == "pending" and candidate["mode"] == "rule"
    assert len(candidate["experiments"]) == 4
    assert all(r["review_required"] for r in candidate["experiments"])
