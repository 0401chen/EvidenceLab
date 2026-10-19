"""Recover only complete coordinate-backed benchmark-column matrices."""
from copy import deepcopy

from backend.app.parsing import _aligned_result_markdown, _recover_caption_bounded_tables


def word(x, y, text, width=22):
    return (x, y, x + width, y + 10, text)


def words():
    return [word(10, 30, "Method", 35), word(120, 30, "Bench-Q", 42), word(210, 30, "Bench-R", 42),
            word(10, 46, "Atlas-A", 40), word(120, 46, "73"), word(146, 46, "±", 8), word(158, 46, "0.4%", 30), word(210, 46, "81%"),
            word(10, 62, "Atlas-B", 40), word(120, 62, "75%"), word(210, 62, "82%")]


def test_dataset_columns_require_unique_percentage_metric_and_complete_geometry():
    source = words()
    assert _aligned_result_markdown(source) is None
    recovered = _aligned_result_markdown(source, "Table 1. Classification accuracies on benchmarks.", with_layout=True)
    assert recovered["markdown"] == "|Method|Bench-Q|Bench-R|\n|---|---|---|\n|Atlas-A|73 ± 0.4%|81%|\n|Atlas-B|75%|82%|"
    assert recovered["rows"][0]["cells"][1]["bbox"] == [120.0, 46.0, 188.0, 56.0]
    assert _aligned_result_markdown(source[:-1], "Table 1. Accuracy (%).") is None
    shifted = deepcopy(source)
    shifted[-1] = word(242, 62, "82%")
    assert _aligned_result_markdown(shifted, "Table 1. Accuracy (%).") is None


def test_ambiguous_metric_and_unitless_pair_are_not_recovered():
    assert _aligned_result_markdown(words(), "Table 1. Accuracy and precision (%).") is None
    assert _aligned_result_markdown(words(), "Table 1. Top-1 accuracy and top-5 accuracy (%).") is None
    source = [word(10, 30, "Method", 35), word(120, 30, "Bench-Q", 42), word(210, 30, "Bench-R", 42),
              word(10, 46, "Atlas-A", 40), word(120, 46, "73 (2)", 50), word(210, 46, "81"),
              word(10, 62, "Atlas-B", 40), word(120, 62, "75"), word(210, 62, "82")]
    assert _aligned_result_markdown(source, "Table 1. Accuracy (in percent).") is None


def test_caption_above_unruled_table_preserves_raw_text_and_cell_positions():
    class Page:
        def get_text(self, kind, **kwargs):
            return words()
    caption = {"id": "c", "paper_id": "paper", "page": 1, "source_type": "text", "section": "Results",
               "bbox": [0, 5, 260, 15], "text": "Table 1. Classification accuracies on benchmarks."}
    raw = {"id": "raw", "paper_id": "paper", "page": 1, "source_type": "text", "section": "Results",
           "bbox": [10, 30, 252, 72], "text": "Method\nBench-Q\nBench-R\nAtlas-A\n73 ± 0.4%\n81%\nAtlas-B\n75%\n82%"}
    evidence = [deepcopy(caption), deepcopy(raw)]
    _recover_caption_bounded_tables(Page(), evidence, 1)
    tables = [item for item in evidence if item["source_type"] == "table"]
    assert len(tables) == 1
    assert evidence[1] == raw
    assert tables[0]["caption_evidence_ids"] == ["c"]
    assert tables[0]["source_text_evidence_ids"] == ["raw"]
    assert tables[0]["coordinate_rows"][0]["cells"][1]["page"] == 1
    cross_page = [deepcopy(caption), deepcopy(raw)]
    cross_page[0]["page"] = 2
    _recover_caption_bounded_tables(Page(), cross_page, 1)
    assert all(item["source_type"] != "table" for item in cross_page)


def test_missing_cell_rejects_whole_table_instead_of_dropping_a_row():
    class Page:
        def get_text(self, kind, **kwargs):
            return words()[:-1]
    evidence = [{"id": "c", "paper_id": "paper", "page": 1, "source_type": "text", "section": "Results",
                 "bbox": [0, 5, 260, 15], "text": "Table 1. Classification accuracy (%)."}]
    _recover_caption_bounded_tables(Page(), evidence, 1)
    assert all(item["source_type"] != "table" for item in evidence)


def test_new_dataset_branch_does_not_mislabel_existing_metric_columns():
    source = words()
    source[1] = word(120, 30, "accuracy", 42)
    source[2] = word(210, 30, "Params", 42)
    # The established generic metric parser remains available, but it must not
    # acquire dataset-column semantics from the new caption-above branch.
    grid = _aligned_result_markdown(source, "Table 1. Classification accuracy (%).", with_layout=True)
    assert grid is None or grid["dataset_columns"] is False
    class Page:
        def get_text(self, kind, **kwargs):
            return source
    evidence = [{"id": "c", "paper_id": "paper", "page": 1, "source_type": "text", "section": "Results",
                 "bbox": [0, 5, 260, 15], "text": "Table 1. Classification accuracy (%)."}]
    _recover_caption_bounded_tables(Page(), evidence, 1)
    assert all(item.get("table_structure") != "caption_bounded_dataset_grid" for item in evidence)
