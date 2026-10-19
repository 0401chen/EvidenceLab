"""Caption bounded grids must retain row/metric/protocol membership."""
from copy import deepcopy

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.parsing import _recover_caption_bounded_tables
from backend.app import providers
from backend.app.errors import ProviderError


def paired_sources(factory, caption="Table 8: The top-1 and top-5 error rates on DemoSet validation set, with single-crop (12-\ncrop) testing."):
    table = factory("|Model|top-1|top-5|\n|---|---|---|\n|Alpha-L (k=17)|24.3 (22.4)|7.2 (6.1)|\n|Alpha-XL (k=29)|23.2 (21.7)|6.4 (5.8)|", "table")
    table.update(source_type="table", caption_evidence_ids=["caption"], caption_association="same_page_unique_geometry")
    return [table, factory(caption, "caption")]


def test_parentheses_map_only_to_the_explicit_caption_protocols(evidence_factory):
    sources = paired_sources(evidence_factory)
    rows = extract_experiments(sources, "paper-a")
    assert len(rows) == 8
    row = next(item for item in rows if item["method"] == "Alpha-XL (k=29)" and item["metric"] == "top1_error"
               and item["conditions"] == {"testing": "12-crop"})
    assert row["value"] == 21.7
    assert row["unit"] is None
    assert row["split"] == "validation" and row["dataset"] == "DemoSet"
    assert row["table_context"]["protocol_position"] == "inside_parentheses"
    assert row["table_context"]["column_index"] == 1
    assert row["table_context"]["value_cell_text"] == "23.2 (21.7)"
    assert row["field_evidence"]["metric"] == ["table", "caption"]


@pytest.mark.parametrize("caption", [
    "Table 8: The top-1 and top-5 error rates, mean (standard deviation).",
    "Table 8: The top-1 and top-5 error rates, with single-crop and 12-crop testing.",
    "Table 8: Accuracy and error rates with single-crop (12-crop) testing.",
    "Table 8: The top-1 and top-5 error rates with single-crop (12-crop) testing and multi-view (16-view) evaluation.",
])
def test_ambiguous_parentheses_do_not_generate_measurements(evidence_factory, caption):
    assert extract_experiments(paired_sources(evidence_factory, caption), "paper-a") == []


def test_missing_protocol_cell_rejects_the_entire_grid(evidence_factory):
    sources = paired_sources(evidence_factory)
    sources[0]["text"] = sources[0]["text"].replace("6.4 (5.8)", "6.4")
    # The ordinary scalar branch may retain that explicit single value without
    # assigning a crop protocol, but no paired protocol claims can be produced.
    rows = extract_experiments(sources, "paper-a")
    assert not any(row.get("table_context", {}).get("protocol_position") for row in rows)


@pytest.mark.parametrize("protocol,valid", [("12-crop", True), ("single-crop", False)])
def test_model_cannot_swap_the_parenthesized_protocol(evidence_factory, protocol, valid):
    sources = paired_sources(evidence_factory)
    row_quote = sources[0]["text"].splitlines()[-1]
    header_quote = sources[0]["text"].splitlines()[0]
    field = lambda value, quote, identifier="S1": {"value": value, "evidence": [{"id": identifier, "quote": quote}]}
    row = {"method": field("Alpha-XL (k=29)", row_quote),
           "metric": {"value": "top1_error", "evidence": [{"id": "S1", "quote": header_quote}, {"id": "S2", "quote": sources[1]["text"]}]},
           "value": field(21.7, row_quote),
           "conditions": field({"testing": protocol}, sources[1]["text"], "S2")}
    if not valid:
        with pytest.raises(ProviderError, match="表格行列"):
            providers._checked_extraction({"experiments": [row]}, sources, "paper-a", {"S1": "table", "S2": "caption"}, 0)
        return
    result = providers._checked_extraction({"experiments": [row]}, sources, "paper-a", {"S1": "table", "S2": "caption"}, 0)
    assert result[0]["value"] == 21.7 and result[0]["conditions"] == {"testing": "12-crop"}
    contexts = providers._result_context_payload(sources, {"S1": "table", "S2": "caption"})
    selected = next(item for item in contexts if item["value"] == 21.7 and item["metric"] == "top1_error")
    assert selected["protocol_position"] == "inside_parentheses"
    assert selected["value_cell_text"] == "23.2 (21.7)"


def test_logical_table_excludes_adjacent_graph_and_preserves_native_text(evidence_factory):
    doc = pymupdf.open()
    page = doc.new_page()
    for x, word in [(70, "Model"), (150, "top-1"), (220, "top-5")]:
        page.insert_text((x, 80), word, fontsize=9)
    for y, model, first, second in [(100, "Alpha-L (k=17)", "24.3 (22.4)", "7.2 (6.1)"),
                                    (120, "Alpha-XL (k=29)", "23.2 (21.7)", "6.4 (5.8)")]:
        for x, word in [(40, model), (142, first), (212, second)]:
            page.insert_text((x, y), word, fontsize=9)
    page.insert_text((320, 90), "Graph 27.5 Alpha-L", fontsize=9)
    page.insert_text((320, 110), "Graph 26.5 Alpha-XL", fontsize=9)
    header = evidence_factory("Model\ntop-1\ntop-5", "header")
    header["bbox"] = [70, 70, 242, 83]
    caption = evidence_factory("Table 8: Error rates with single-crop (12-crop) testing.", "caption")
    caption["bbox"] = [38, 140, 270, 175]
    native = evidence_factory("native malformed grid including Graph 27.5", "native")
    native.update(source_type="table", bbox=[38, 60, 560, 130])
    evidence = [header, caption, native]
    before = deepcopy(evidence)
    _recover_caption_bounded_tables(page, evidence, 1)
    assert evidence[:3] == before
    logical = evidence[3:]
    assert len(logical) == 1
    assert "Graph" not in logical[0]["text"]
    assert "|Alpha-XL (k=29)|23.2 (21.7)|6.4 (5.8)|" in logical[0]["text"]
    assert logical[0]["bbox"][2] < 300
    assert logical[0]["header_evidence_ids"] == ["header"]
    assert logical[0]["caption_evidence_ids"] == ["caption"]
    doc.close()


def test_comparison_retains_bound_header_caption_provenance(evidence_factory):
    from backend.app.orchestration import run_comparison
    first = paired_sources(evidence_factory)
    second = deepcopy(first)
    for source in second:
        source["paper_id"] = "paper-b"
        source["id"] += "-b"
        if "caption_evidence_ids" in source:
            source["caption_evidence_ids"] = ["caption-b"]
    left = extract_experiments(first, "paper-a")[0]
    right = extract_experiments(second, "paper-b")[0]
    result = run_comparison([left, right], first + second)["comparisons"][0]
    # Missing unit/task are real; the explicitly bound error metric is not missing.
    assert result["verdict"] == "insufficient" and result["delta"] is None
    assert not any("缺少有效依据的字段" in reason for reason in result["reasons"])
    right["conditions"] = {"testing": "12-crop"}  # Value belongs outside parentheses.
    rejected = run_comparison([left, right], first + second)["comparisons"][0]
    assert any("result_association" in reason for reason in rejected["reasons"])
