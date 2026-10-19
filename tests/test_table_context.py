"""Strict table-caption association and numeric matrix regressions, offline."""
from pathlib import Path

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.experiment_validation import prepare_experiments
from backend.app.grounding import quote_matches_source
from backend.app.parsing import (_associate_table_captions, _numeric_matrix_markdown,
                                 parse_document)
from backend.app.providers import _field_is_literal


def write_matrix(path: Path, caption: str, *, caption_page=1):
    doc = pymupdf.open()
    page = doc.new_page()
    xs, ys = [60, 180, 270, 360, 450], [80, 110, 140, 170, 200]
    for x in xs:
        page.draw_line((x, 80), (x, 200))
    for y in ys:
        page.draw_line((60, y), (450, y))
    matrix = (("batch size", "10", "20", "40"), ("Method-A", "12.3", "13.4", "14.5"),
              ("Method-B", "11.2", "12.3", "13.4"), ("difference", "-1.1", "-1.1", "-1.1"))
    for row, cells in enumerate(matrix):
        for column, text in enumerate(cells):
            page.insert_text((xs[column] + 5, ys[row] + 20), text, fontsize=10)
    if caption_page == 2:
        page = doc.new_page()
    page.insert_textbox((60, 215, 450, 270), caption, fontsize=10)
    doc.save(path)
    doc.close()


def test_parameter_columns_bind_to_unique_caption_with_separate_field_sources(tmp_path):
    path = tmp_path / "matrix.pdf"
    write_matrix(path, "Table 7. ModelZ's validation error (%) on StudySet dataset. Comparison of methods.")
    parsed = parse_document(path, "paper-matrix")
    table = next(e for e in parsed["evidence"] if e.get("caption_evidence_ids"))
    caption = next(e for e in parsed["evidence"] if e["id"] == table["caption_evidence_ids"][0])
    assert "Table 7." not in table["text"]
    assert caption["page"] == table["page"] == 1
    rows = extract_experiments(parsed["evidence"], "paper-matrix")
    rows = [row for row in rows if row.get("table_context")]
    assert len(rows) == 6
    assert {(r["method"], r["conditions"]["batch size"], r["value"]) for r in rows} == {
        (method, batch, value) for method, values in (("Method-A", (12.3, 13.4, 14.5)), ("Method-B", (11.2, 12.3, 13.4)))
        for batch, value in zip(("10", "20", "40"), values)}
    for row in rows:
        assert row["dataset"] == "StudySet"
        assert row["split"] == "validation"
        assert row["metric"] == "validation error"
        assert row["unit"] == "%"
        assert row["conditions"]["ModelZ"] == "ModelZ"
        assert row["field_evidence"]["value"] == [table["id"]]
        assert row["field_evidence"]["metric"] == [caption["id"]]
        for field, references in row["evidence_quotes"].items():
            assert _field_is_literal(field, row[field], [], references)
            assert all(quote_matches_source(ref["quote"], next(e["text"] for e in parsed["evidence"] if e["id"] == ref["id"])) for ref in references)
    assert len(prepare_experiments(rows, {"id": "paper-matrix", "title": "Synthetic"}, parsed["evidence"])) == 6
    assert [e["id"] for e in parse_document(path, "paper-matrix")["evidence"]] == [e["id"] for e in parsed["evidence"]]


@pytest.mark.parametrize("caption,page", [
    ("Table 7. Error (%) and accuracy (%) of the methods.", 1),
    ("Table 7. Error (%) and F1 of the methods.", 1),
    ("The methods report validation error (%).", 1),
    ("Table 7. The methods report validation error (%).", 2),
])
def test_ambiguous_or_unlocated_caption_does_not_supply_a_metric(tmp_path, caption, page):
    path = tmp_path / "ambiguous.pdf"
    write_matrix(path, caption, caption_page=page)
    parsed = parse_document(path, "paper-matrix")
    assert not any(row.get("table_context") for row in extract_experiments(parsed["evidence"], "paper-matrix"))


def test_unqualified_capitalized_location_is_not_assumed_to_be_a_dataset(tmp_path):
    path = tmp_path / "location.pdf"
    write_matrix(path, "Table 9. ModelZ's validation error (%) in NorthCity.")
    parsed = parse_document(path, "paper-matrix")
    rows = [row for row in extract_experiments(parsed["evidence"], "paper-matrix") if row.get("table_context")]
    assert len(rows) == 6
    assert all(row["dataset"] is None and row["method"] == "ModelZ" for row in rows)
    assert {row["table_context"]["row_label"] for row in rows} == {"Method-A", "Method-B"}


def test_equally_close_numbered_captions_are_not_associated(evidence_factory):
    table = evidence_factory("grid", "table")
    table.update(source_type="table", bbox=[60, 100, 450, 200])
    before = evidence_factory("Table 1. Error (%).", "before")
    before["bbox"] = [60, 70, 450, 90]
    after = evidence_factory("Table 2. Error (%).", "after")
    after["bbox"] = [60, 210, 450, 230]
    _associate_table_captions([table, before, after], 1)
    assert "caption_evidence_ids" not in table


def test_one_caption_cannot_be_assigned_to_two_equally_near_tables(evidence_factory):
    left = evidence_factory("grid", "left")
    left.update(source_type="table", bbox=[60, 100, 250, 200])
    right = evidence_factory("grid", "right")
    right.update(source_type="table", bbox=[260, 100, 450, 200])
    caption = evidence_factory("Table 1. Error (%).", "caption")
    caption["bbox"] = [60, 210, 450, 230]
    _associate_table_captions([left, right, caption], 1)
    assert "caption_evidence_ids" not in left and "caption_evidence_ids" not in right


def test_numeric_matrix_recovery_requires_complete_aligned_columns():
    def word(x, y, text):
        return (x, y, x + 10, y + 8, text)

    words = [word(0, 0, "size"), word(40, 0, "10"), word(80, 0, "20"),
             word(0, 12, "Alpha"), word(40, 12, "1.3"), word(80, 12, "2.4"),
             word(0, 24, "Beta"), word(40, 24, "3.5"), word(80, 24, "4.6")]
    assert _numeric_matrix_markdown(words)
    assert _numeric_matrix_markdown(words[:-1]) is None
    shifted = [*words[:-1], word(95, 24, "4.6")]
    assert _numeric_matrix_markdown(shifted) is None
