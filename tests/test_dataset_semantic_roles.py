"""Second-round scientific-role regressions, exclusively synthetic sources."""
from copy import deepcopy
import math

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.parsing import parse_document


def write_pdf(path, *, caption="Table 1: Classification accuracy (in percent) on the two benchmark datasets.",
              headers=("Method", "Bench-Q", "Bench-R"),
              rows=(("Network-A", "81.5", "78.1"), ("Network-B", "83.0 ± 0.7%", "80.2%"))):
    # The existing recovery clips to the caption's horizontal bounds. Make
    # those bounds cover ALL synthetic columns, including short negative titles;
    # otherwise clipping can accidentally make a semantic negative test pass.
    spaces = max(1, math.ceil((430 - pymupdf.get_text_length(caption, fontsize=10)) /
                              pymupdf.get_text_length(" ", fontsize=10)))
    doc = pymupdf.open()
    page = doc.new_page()
    remaining = page.insert_textbox((50, 35, 550, 73), caption.replace(": ", ": " + " " * spaces, 1), fontsize=10)
    assert remaining >= 0, "Synthetic caption must be complete, never clipped"
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


@pytest.mark.parametrize("caption,headers", [
    ("Table 1: Classification accuracy (%) with different optimizers.", ("Method", "Adam", "SGD")),
    ("Table 1: Classification accuracy (%).", ("Method", "Alpha", "Beta")),
    ("Table 1: Classification accuracy (%) for different hardware.", ("Method", "CPU", "GPU")),
])
def test_p1_unknown_axes_are_not_invented_datasets(tmp_path, caption, headers):
    parsed, rows = fixture(tmp_path, caption=caption, headers=headers)
    assert any(e.get("source_type") == "table" for e in parsed["evidence"])
    assert rows == []


@pytest.mark.parametrize("other", ["calibration loss", "perplexity", "BLEU", "latency (ms)", "QXZ", "QXZ (bits)"])
def test_p1_column_specific_metric_cannot_inherit_accuracy(tmp_path, other):
    _, rows = fixture(tmp_path, caption=f"Table 1: Accuracy (%) on Bench-Q and {other} on Bench-R.")
    assert rows == []


@pytest.mark.parametrize("suffix", [
    "Bench-R reports QXZ instead.", "Bench-R uses a different scale.",
    "The second column is perplexity.", "Only Bench-Q is measured as a percentage.",
])
def test_p1_later_caption_sentence_cannot_change_metric_or_unit_scope(tmp_path, suffix):
    _, rows = fixture(tmp_path, caption="Table 1: Accuracy (%) on the two benchmark datasets. " + suffix)
    assert rows == []


@pytest.mark.parametrize("method", ["Change Transformer", "GainNet", "RatioNet", "Ratio Model", "Difference Transformer"])
def test_p2_model_names_are_not_keyword_filtered(tmp_path, method):
    _, rows = fixture(tmp_path, rows=((method, "81.5", "78.1"), ("Network-B", "83.0", "80.2")))
    assert len(rows) == 4 and sum(r["method"] == method for r in rows) == 2


@pytest.mark.parametrize("label", ["Difference", "Difference (B - A)", "Absolute difference"])
def test_p2_explicit_difference_summary_is_not_a_model(tmp_path, label):
    _, rows = fixture(tmp_path, rows=(("Network-A", "81.5", "78.1"),
                                    ("Network-B", "83.0", "80.2"), (label, "1.5", "2.1")))
    assert len(rows) == 4


def test_p2_same_physical_table_different_ids_has_one_result_per_cell(tmp_path):
    parsed, _ = fixture(tmp_path)
    sources = parsed["evidence"]
    original = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    duplicate = {**deepcopy(original), "id": "second-immutable-id"}
    sources.append(duplicate)
    before = deepcopy(sources)
    rows = extract_experiments(sources, "paper-synthetic")
    assert len(rows) == 4
    assert sources == before
    assert all(r["table_context"]["table_evidence_id"] == original["id"] for r in rows)
    assert [r["id"] for r in rows] == [r["id"] for r in extract_experiments(sources[:-1], "paper-synthetic")]


@pytest.mark.parametrize("change", ["page", "physical_table", "paper"])
def test_p2_independent_equal_measurements_are_not_globally_deduplicated(tmp_path, change):
    parsed, _ = fixture(tmp_path)
    table = next(e for e in parsed["evidence"] if e.get("table_structure") == "caption_bounded_dataset_grid")
    caption = next(e for e in parsed["evidence"] if e["id"] == table["caption_evidence_ids"][0])
    other, other_caption = deepcopy(table), deepcopy(caption)
    other["id"] = "independent-table"
    other_caption["id"] = "independent-caption"
    other["caption_evidence_ids"] = [other_caption["id"]]
    if change == "page":
        other["page"] = other_caption["page"] = 2
        for row in other["coordinate_rows"]:
            for cell in row["cells"]:
                cell["page"] = 2
    elif change == "physical_table":
        other["bbox"] = [v + (200 if i % 2 else 0) for i, v in enumerate(other["bbox"])]
        other_caption["bbox"] = [v + (200 if i % 2 else 0) for i, v in enumerate(other_caption["bbox"])]
        for row in other["coordinate_rows"]:
            for cell in row["cells"]:
                cell["bbox"] = [v + (200 if i % 2 else 0) for i, v in enumerate(cell["bbox"])]
    else:
        other["paper_id"] = other_caption["paper_id"] = "paper-independent"
    sources = [*parsed["evidence"], other, other_caption]
    if change == "paper":
        sources.append({"id": "other-roles", "paper_id": "paper-independent", "page": 1,
                        "source_type": "text", "section": "Datasets",
                        "text": "Bench-Q dataset and Bench-R dataset are evaluated."})
    rows = extract_experiments(sources, "paper-synthetic")
    if change == "paper":
        rows += extract_experiments(sources, "paper-independent")
    assert len(rows) == 8


def test_explicit_named_scope_uses_literal_dataset_declarations(tmp_path):
    parsed, _ = fixture(tmp_path, caption="Table 1: Classification accuracies, for Bench-Q and Bench-R.",
                        rows=(("Network-A", "81.5%", "78.1%"), ("Network-B", "83%", "80%")))
    declaration = {"id": "dataset-definition", "paper_id": "paper-synthetic", "page": 2,
                   "source_type": "text", "section": "Datasets", "bbox": [50, 40, 500, 70],
                   "text": "Bench-Q dataset and Bench-R dataset are evaluated."}
    rows = extract_experiments([*parsed["evidence"], declaration], "paper-synthetic")
    assert len(rows) == 4
    assert all(declaration["id"] in r["field_evidence"]["dataset"] for r in rows)
    assert all(r["conditions"] == {} for r in rows)


def test_dataset_declaration_elsewhere_without_table_link_is_not_enough(tmp_path):
    parsed, _ = fixture(tmp_path, caption="Table 1: Classification accuracy (%).")
    declaration = {"id": "dataset-definition", "paper_id": "paper-synthetic", "page": 2,
                   "source_type": "text", "section": "Datasets", "bbox": None,
                   "text": "Bench-Q dataset and Bench-R dataset are evaluated."}
    assert extract_experiments([*parsed["evidence"], declaration], "paper-synthetic") == []


@pytest.mark.parametrize("caption,headers", [
    ("Table 1: Classification accuracy (%) with different optimizers.", ("Method", "Adam", "SGD")),
    ("Table 1: Accuracy (%) on Bench-Q and calibration loss on Bench-R.", ("Method", "Bench-Q", "Bench-R")),
])
def test_initial_upload_and_comparison_do_not_persist_wrong_roles(client, tmp_path, caption, headers):
    path = tmp_path / "ambiguous.pdf"
    write_pdf(path, caption=caption, headers=headers)
    response = client.post("/api/papers/upload", files={"files": ("ambiguous.pdf", path.read_bytes(), "application/pdf")})
    assert response.status_code == 200 and response.json()["errors"] == []
    paper = response.json()["papers"][0]
    detail = client.get(f"/api/papers/{paper['id']}").json()
    assert detail["experiments"] == []
    assert any(e["source_type"] == "table" for e in detail["evidence"])
    good = tmp_path / "explicit.pdf"
    write_pdf(good)
    good_id = client.post("/api/papers/upload", files={"files": ("explicit.pdf", good.read_bytes(), "application/pdf")}).json()["papers"][0]["id"]
    report = client.post("/api/compare", json={"paper_ids": [paper["id"], good_id]}).json()
    assert all(r["paper_id"] != paper["id"] for r in report["experiments"])
    assert all(r["review_required"] for r in report["experiments"])


def test_initial_upload_review_required_is_not_a_candidate_queue(client, tmp_path):
    path = tmp_path / "explicit.pdf"
    write_pdf(path)
    paper = client.post("/api/papers/upload", files={"files": ("explicit.pdf", path.read_bytes(), "application/pdf")}).json()["papers"][0]
    detail = client.get(f"/api/papers/{paper['id']}").json()
    assert len(detail["experiments"]) == 4 and detail["extraction_candidates"] == []
    assert all(r["review_required"] for r in detail["experiments"])


def test_model_source_anchors_keep_required_role_context_and_reject_wrong_owner(tmp_path):
    from backend.app import providers
    parsed, _ = fixture(tmp_path, caption="Table 1: Accuracy (%), for Bench-Q and Bench-R.")
    definition = {"id": "roles", "paper_id": "paper-synthetic", "page": 2, "source_type": "text",
                  "section": "Datasets", "text": "Bench-Q dataset and Bench-R dataset are evaluated."}
    table = next(e for e in parsed["evidence"] if e.get("table_structure") == "caption_bounded_dataset_grid")
    sources = [*parsed["evidence"], definition]
    anchors = providers._table_result_anchors(table, sources)
    assert len(anchors) == 4 and all("roles" in r["evidence_ids"] for r in anchors)
    row = next(r for r in anchors if r["value"] == 83)
    assert providers._measurement_anchor_conflicts({**row, "method": "Network-A", "unit": None}, sources)
    assert providers._measurement_anchor_conflicts({**row, "dataset": "Bench-R", "unit": None}, sources)
    assert providers._measurement_anchor_conflicts({**row, "value": 0.7, "unit": None}, sources)
    assert not providers._table_result_anchors(table, parsed["evidence"])
    assert providers._measurement_anchor_conflicts({**row, "unit": None}, parsed["evidence"])


@pytest.mark.parametrize("link,valid", [
    ("Results are summarized in Table 7. Each dataset is evaluated independently.", True),
    ("Results are summarized in Table 8. Each dataset is evaluated independently.", False),
    ("Results are in Table 7 and Table 8. Each dataset is evaluated independently.", False),
    ("Results are summarized in Table 7.", False),
])
def test_unscoped_caption_needs_both_dataset_definitions_and_unique_table_reference(tmp_path, link, valid):
    parsed, _ = fixture(tmp_path, caption="Table 7: Summary of results in terms of accuracy (%).")
    definitions = {"id": "definitions", "paper_id": "paper-synthetic", "page": 2, "source_type": "text",
                   "section": "Datasets", "text": "Datasets: Bench-Q and Bench-R."}
    context = {"id": "scope", "paper_id": "paper-synthetic", "page": 1, "source_type": "text",
               "section": "Results", "text": link}
    rows = extract_experiments([*parsed["evidence"], definitions, context], "paper-synthetic")
    assert len(rows) == (4 if valid else 0)
    if valid:
        assert all({"scope", "definitions"}.issubset(r["field_evidence"]["dataset"]) for r in rows)
        assert all(r["conditions"] == {} for r in rows)


@pytest.mark.parametrize("text", [
    "These are not datasets: Bench-Q and Bench-R.",
    "Unlike datasets: Bench-Q and Bench-R.",
    "There is no Bench-Q dataset or Bench-R dataset.",
])
def test_negated_dataset_descriptions_are_not_positive_role_proof(tmp_path, text):
    parsed, _ = fixture(tmp_path, caption="Table 1: Accuracy (%), for Bench-Q and Bench-R.")
    definition = {"id": "denial", "paper_id": "paper-synthetic", "page": 2,
                  "source_type": "text", "section": "Definitions", "text": text}
    assert extract_experiments([*parsed["evidence"], definition], "paper-synthetic") == []


def test_explicit_feature_definition_is_not_a_second_metric_or_inferred_condition(tmp_path):
    parsed, _ = fixture(tmp_path, caption="Table 1: Accuracy (%), for Bench-Q and Bench-R. "
                        "Network-B corresponds to the best Network-A result computing 23 hidden features (using Tanh or ReLU).")
    definition = {"id": "roles", "paper_id": "paper-synthetic", "page": 2, "source_type": "text",
                  "section": "Datasets", "text": "Bench-Q dataset and Bench-R dataset are evaluated."}
    rows = extract_experiments([*parsed["evidence"], definition], "paper-synthetic")
    assert len(rows) == 4 and all(r["conditions"] == {} for r in rows)


@pytest.mark.parametrize("metric", ["perplexity", "QXZ", "latency"])
def test_feature_footnote_cannot_launder_an_unknown_metric_as_a_model_name(tmp_path, metric):
    parsed, _ = fixture(tmp_path, caption="Table 1: Accuracy (%), for Bench-Q and Bench-R. "
                        f"Network-B corresponds to the best {metric} result computing 23 hidden features (using Tanh).")
    definition = {"id": "roles", "paper_id": "paper-synthetic", "page": 2, "source_type": "text",
                  "section": "Datasets", "text": "Bench-Q dataset and Bench-R dataset are evaluated."}
    assert extract_experiments([*parsed["evidence"], definition], "paper-synthetic") == []
