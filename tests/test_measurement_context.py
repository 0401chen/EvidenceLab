"""Result anchors bind values to an actual row and preserve uncertain roles."""
import asyncio
import json

import pytest

from backend.app import providers
from backend.app.analysis import extract_experiments
from backend.app.errors import ProviderError
from backend.app.parsing import _aligned_result_markdown, _associate_table_captions


def field(value, quote, source="S1"):
    return {"value": value, "evidence": [{"id": source, "quote": quote}]}


def table_sources(factory):
    grid = factory("|method|top-1 err.|top-5 err.|\n|---|---|---|\n|Alpha-L|21.1|5.3|\n|Alpha-XL|19.2|4.6|", "table")
    grid.update(source_type="table", bbox=[50, 100, 290, 180])
    caption = factory("Table 3. Error rates (%) of single-model results on the DemoData validation set.", "caption")
    caption["bbox"] = [50, 185, 290, 205]
    _associate_table_captions([grid, caption], 1)
    return [grid, caption]


def test_geometric_flat_result_grid_preserves_width_and_columns():
    def word(x, y, text, width=18):
        return (x, y, x + width, y + 8, text)
    words = [word(0, 0, "Network", 30), word(110, 0, "mAP"), word(150, 0, "Params", 24),
             word(0, 14, "Alpha", 25), word(29, 14, "(1.7)", 22), word(110, 14, "26.4"), word(150, 14, "4.6M"),
             word(0, 28, "Beta", 20), word(110, 28, "25.2"), word(150, 28, "3.8M")]
    result = _aligned_result_markdown(words)
    assert result == "|Network|mAP|Params|\n|---|---|---|\n|Alpha (1.7)|26.4|4.6M|\n|Beta|25.2|3.8M|"
    assert _aligned_result_markdown(words[:-1]) is None
    assert _aligned_result_markdown([*words[:-1], word(180, 28, "3.8M")]) is None


def test_table_row_metric_and_caption_are_separate_sources(evidence_factory):
    sources = table_sources(evidence_factory)
    rows = extract_experiments(sources, "paper-a")
    selected = next(row for row in rows if row["method"] == "Alpha-XL" and row["metric"] == "top5_error")
    assert selected["value"] == 4.6
    assert selected["unit"] == "%"
    assert selected["dataset"] == "DemoData"
    assert selected["split"] == "validation"
    assert selected["conditions"] == {"single-model": "single-model"}
    assert selected["table_context"]["row_quote"] == "|Alpha-XL|19.2|4.6|"
    assert selected["field_evidence"]["method"] == [sources[0]["id"]]
    assert selected["field_evidence"]["split"] == [sources[1]["id"]]


def test_llm_cannot_borrow_a_model_from_another_table_row(evidence_factory):
    sources = table_sources(evidence_factory)
    aliases = {"S1": sources[0]["id"], "S2": sources[1]["id"]}
    row = {"method": field("Alpha-L", "|Alpha-L|21.1|5.3|"),
           "metric": field("top-5 err.", "|method|top-1 err.|top-5 err.|"),
           "value": field(4.6, "|Alpha-XL|19.2|4.6|"),
           "unit": field("%", sources[1]["text"], "S2")}
    with pytest.raises(ProviderError, match="测量所在"):
        providers._checked_extraction({"experiments": [row]}, sources, "paper-a", aliases, 0)
    row["method"] = field("Alpha-XL", "|Alpha-XL|19.2|4.6|")
    checked = providers._checked_extraction({"experiments": [row]}, sources, "paper-a", aliases, 0)
    assert checked[0]["method"] == "Alpha-XL"
    row["metric"] = field("top-1 err.", "|method|top-1 err.|top-5 err.|")
    row["unit"] = None
    with pytest.raises(ProviderError, match="表格行列"):
        providers._checked_extraction({"experiments": [row]}, sources, "paper-a", aliases, 0)


def test_short_identity_quote_cannot_jump_between_result_sentences(evidence_factory):
    source = evidence_factory("Earlier results used OldNet ensemble. The latter reports top-5 error of 5.17%. Here we report test error of 4.63% on test server.")
    row = {"method": field("OldNet ensemble", "OldNet ensemble"),
           "metric": field("test error", "test error of 4.63%"),
           "value": field(4.63, "test error of 4.63%"), "unit": field("%", "test error of 4.63%")}
    with pytest.raises(ProviderError, match="测量所在"):
        providers._checked_extraction({"experiments": [row]}, [source], "paper-a", {"S1": source["id"]}, 0)


def test_bare_top_k_header_does_not_invent_accuracy_or_unit(evidence_factory):
    source = evidence_factory("|Network|Top 1|Params|\n|---|---|---|\n|Alpha (1.7)|73.4|4.6M|\n|Beta|72.8|3.8M|")
    source["source_type"] = "table"
    rows = extract_experiments([source], "paper-a")
    assert len(rows) == 2
    assert rows[0]["method"] == "Alpha (1.7)"
    assert rows[0]["metric"] == "Top 1"
    assert rows[0]["unit"] is None


def test_model_gets_row_contexts_and_an_explicit_total_result_limit(monkeypatch, evidence_factory):
    sources = table_sources(evidence_factory)
    seen = []
    async def reply(messages, config):
        payload = json.loads(messages[-1]["content"])
        seen.append(payload)
        assert payload["result_limit"] == 1
        assert len(payload["result_contexts"]) == 4
        return {"experiments": [
            {"method": field("Alpha-L", "|Alpha-L|21.1|5.3|"), "metric": field("top-5 err.", "|method|top-1 err.|top-5 err.|"),
             "value": field(5.3, "|Alpha-L|21.1|5.3|")},
            {"method": field("Alpha-XL", "|Alpha-XL|19.2|4.6|"), "metric": field("top-5 err.", "|method|top-1 err.|top-5 err.|"),
             "value": field(4.6, "|Alpha-XL|19.2|4.6|")}]}
    monkeypatch.setattr(providers, "_chat_json", reply)
    rows = asyncio.run(providers.extract_llm(sources, "paper-a", {"model": "offline-test", "max_results": 1, "enrich_metadata": False}))
    assert len(rows) == 1 and len(seen) == 1
    assert rows[0]["extraction_method"] == "llm"
    assert any("最多 1 条候选" in note for note in rows[0]["notes"])
