"""Source preservation is a prerequisite for every downstream citation."""

import fitz
import pytest

from backend.app.parsing import parse_document


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_text_preserves_unicode_and_paper_identity(tmp_path, suffix):
    source = tmp_path / f"experiment{suffix}"
    source.write_text(
        "# 合成研究示例\n\n研究对象为合成材料，样本量为100。\n\n"
        "The measured accuracy is 90 percent on test-A.\n",
        encoding="utf-8",
    )
    parsed = parse_document(source, "paper-unicode")
    assert parsed["page_count"] >= 1
    assert parsed["evidence"]
    combined_text = "\n".join(item["text"] for item in parsed["evidence"])
    assert "研究对象为合成材料" in combined_text
    assert "90 percent on test-A" in combined_text
    assert len({item["id"] for item in parsed["evidence"]}) == len(parsed["evidence"])
    for item in parsed["evidence"]:
        assert item["paper_id"] == "paper-unicode"
        assert item["page"] >= 1
        assert item["text"].strip()


def test_pdf_evidence_retains_page_and_coordinate_provenance(tmp_path):
    source = tmp_path / "two-pages.pdf"
    document = fitz.open()
    for text in ("Synthetic study\nMethod: AlphaNet", "Results\nAccuracy: 90 percent"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(source)
    document.close()

    parsed = parse_document(source, "paper-pdf")
    assert parsed["page_count"] == 2
    assert "AlphaNet" in " ".join(e["text"] for e in parsed["evidence"] if e["page"] == 1)
    results = [e for e in parsed["evidence"] if "90 percent" in e["text"]]
    assert results and all(e["page"] == 2 for e in results)
    for item in parsed["evidence"]:
        assert item["paper_id"] == "paper-pdf"
        assert item["bbox"] is not None and len(item["bbox"]) == 4
        assert item["bbox"][0] <= item["bbox"][2]
        assert item["bbox"][1] <= item["bbox"][3]


def test_evidence_identifiers_are_stable_for_repeated_parse(tmp_path):
    source = tmp_path / "stable.md"
    source.write_text("# Synthetic paper\n\nA fixed source sentence.", encoding="utf-8")
    first = parse_document(source, "paper-stable")
    second = parse_document(source, "paper-stable")
    assert [e["id"] for e in first["evidence"]] == [e["id"] for e in second["evidence"]]
