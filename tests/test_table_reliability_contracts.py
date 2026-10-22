"""Third-round public synthetic PDFs; scientific role and physical identity."""
from copy import deepcopy
import math

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.experiment_validation import prepare_experiments
from backend.app.grounding import quote_matches_source
from backend.app.parsing import parse_document
from backend.app import providers
from backend.app.errors import ProviderError


CAPTION = "Table 1: Classification accuracy (in percent) on the two benchmark datasets."
ROWS = (("Network-A", "81.5", "78.1"), ("Network-B", "83.0 ± 0.7%", "80.2%"))


def pdf(path, *, caption=CAPTION, headers=("Method", "Bench-Q", "Bench-R"), rows=ROWS,
        note="", definitions="Bench-Q dataset and Bench-R dataset are evaluated.", require_grid=True):
    doc = pymupdf.open()
    page = doc.new_page()
    spaces = max(1, math.ceil((430 - pymupdf.get_text_length(caption, fontsize=10)) /
                             pymupdf.get_text_length(" ", fontsize=10)))
    assert page.insert_textbox((50, 35, 550, 73), caption.replace(": ", ": " + " " * spaces, 1), fontsize=10) >= 0
    for y, cells in [(85, headers), *[(105 + i * 20, row) for i, row in enumerate(rows)]]:
        for x, value in zip((50, 210, 330), cells):
            page.insert_text((x, y), value, fontsize=10)
    if note:
        assert page.insert_textbox((50, 225, 550, 290), note, fontsize=10) >= 0
    if definitions:
        assert page.insert_textbox((50, 315, 550, 390), definitions, fontsize=10) >= 0
    doc.save(path)
    doc.close()
    sources = parse_document(path, "synthetic-reliability")["evidence"]
    if require_grid:
        assert any(e.get("table_structure") == "caption_bounded_dataset_grid" for e in sources)
    return sources


def records(sources):
    rows = extract_experiments(sources, "synthetic-reliability")
    return prepare_experiments(rows, {"id": "synthetic-reliability", "title": "Synthetic"}, sources)


def model_row(sources, method="Network-A", dataset="Bench-Q", value=81.5):
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    caption = next(e for e in sources if e["id"] == table["caption_evidence_ids"][0])
    line = next(line for line in table["text"].splitlines() if line.startswith("|" + method + "|"))
    def field(v, source, quote):
        return {"value": v, "evidence": [{"id": source["id"], "quote": quote}]}
    return {"method": field(method, table, line), "dataset": field(dataset, table, table["text"].splitlines()[0]),
            "metric": field("accuracy", caption, caption["text"]), "value": field(value, table, line),
            "unit": field("%", caption, caption["text"])}


@pytest.mark.parametrize("headers,role", [(("Method", "Adam", "SGD"), "optimizers"),
    (("Method", "Opt-X", "Opt-Y"), "optimization configurations"),
    (("Method", "CPU", "GPU"), "hardware devices"),
    (("Method", "Warm", "Cold"), "experimental conditions")])
def test_evaluation_scope_is_not_axis_identity(tmp_path, headers, role):
    sources = pdf(tmp_path / "scope.pdf", headers=headers,
                  note=f"Table 1 compares {headers[1]} and {headers[2]}. The columns are {role}, not datasets.")
    assert records(sources) == []
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [model_row(sources, dataset=headers[1])]}, sources,
                                      "synthetic-reliability", {}, 0)


def test_generic_dataset_scope_without_named_role_proof_abstains(tmp_path):
    assert records(pdf(tmp_path / "scope.pdf", definitions="")) == []


def test_named_axis_caption_does_not_hide_available_conflicting_body_from_model_guard(tmp_path):
    sources = pdf(tmp_path / "named.pdf", caption="Table 1: Accuracy (%) on datasets Bench-Q and Bench-R.",
                  note="In Table 1, the Bench-R column reports error rather than accuracy.")
    assert records(sources) == []
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    assert providers._table_result_anchors(table, sources) == []
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [model_row(sources, dataset="Bench-R", value=78.1)]}, sources,
                                      "synthetic-reliability", {}, 0)


@pytest.mark.parametrize("note", [
    "In Table 1, the Bench-R column reports classification error rather than accuracy. Only Bench-Q reports accuracy.",
    "In Table 1, Bench-R reports QXZ instead of accuracy.",
    "Table 1 uses milliseconds for the Bench-R column.",
    "Table 1: Only Bench-Q is measured as a percentage.",
    "Table 1 reports different metrics for the two columns.",
    "In Table 1, the Bench-R column reports\nperplexity, not accuracy.",
    "Table 1 shows Bench-R yields QXZ.",
    "In Table 1, values for Bench-R are expressed in bits.",
    "In Table 1, Bench-R: QXZ.",
    "In Table 1, QXZ for Bench-R.",
])
def test_separate_scoped_metric_or_unit_constraint_rejects_whole_grid(tmp_path, note):
    sources = pdf(tmp_path / "scope.pdf", note=note)
    assert records(sources) == []
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [model_row(sources, dataset="Bench-R", value=78.1)]}, sources,
                                      "synthetic-reliability", {}, 0)


def test_other_table_constraint_does_not_poison_current_table(tmp_path):
    sources = pdf(tmp_path / "scope.pdf", note="In Table 8, Bench-R reports QXZ rather than accuracy.")
    assert len(records(sources)) == 4


@pytest.mark.parametrize("label", ["Gain over Network-A", "Ratio (B/A)", "Improvement relative to Network-A",
                                    "Difference (B - A)", "Relative gain", "Ratio", "Network-B / Network-A",
                                    "Network-B - Network-A", "Mean", "Standard deviation",
                                    "Mean of Network-A and Network-B", "Average across models", "Ratio B/A"])
def test_explicit_derived_or_unproven_statistic_is_not_absolute_accuracy(tmp_path, label):
    sources = pdf(tmp_path / "summary.pdf", rows=(*ROWS, (label, "1.5", "2.1")))
    rows = records(sources)
    assert len(rows) == 4 and all(r["method"] in {"Network-A", "Network-B"} for r in rows)
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [model_row(sources, method=label, value=1.5)]}, sources,
                                      "synthetic-reliability", {}, 0)


@pytest.mark.parametrize("method", ["Delta", "GainNet", "Ratio Model", "Change Transformer", "Difference Transformer"])
def test_literal_model_identity_survives_statistical_name_overlap(tmp_path, method):
    sources = pdf(tmp_path / "model.pdf", rows=((method, "81.5", "78.1"), ROWS[1]),
                  note=f"{method} is the name of our proposed classifier, not a difference statistic. Table 1 evaluates it.")
    rows = records(sources)
    assert len(rows) == 4
    row = next(r for r in rows if r["method"] == method and r["dataset"] == "Bench-Q")
    assert row["value"] == 81.5
    checked = providers._checked_extraction({"experiments": [model_row(sources, method=method)]}, sources,
                                          "synthetic-reliability", {}, 0)
    assert len(checked) == 1
    source_map = {e["id"]: e for e in sources}
    assert all(quote_matches_source(ref["quote"], source_map[ref["id"]]["text"])
               for refs in row["evidence_quotes"].values() for ref in refs)


@pytest.mark.parametrize("jitter", [0.005, 0.01])
@pytest.mark.parametrize("target", ["parent", "cells", "caption"])
def test_same_physical_cells_with_bounded_float_jitter_are_unique(tmp_path, jitter, target):
    sources = pdf(tmp_path / "duplicate.pdf")
    original = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    duplicate = deepcopy(original)
    duplicate["id"] = "duplicate-table"
    if target == "parent":
        duplicate["bbox"][0] -= jitter
        duplicate["bbox"][2] += jitter
    elif target == "cells":
        for row in duplicate["coordinate_rows"]:
            for cell in row["cells"]:
                cell["bbox"][0] += jitter
                cell["bbox"][2] += jitter
    else:
        caption = deepcopy(next(e for e in sources if e["id"] == original["caption_evidence_ids"][0]))
        caption["id"] = "duplicate-caption"
        caption["bbox"][0] += jitter
        caption["bbox"][2] += jitter
        duplicate["caption_evidence_ids"] = [caption["id"]]
        sources.append(caption)
    old = records(sources)
    sources.append(duplicate)
    snapshot = deepcopy(sources)
    rows = records(sources)
    assert len(rows) == 4 and [r["id"] for r in rows] == [r["id"] for r in old]
    assert sources == snapshot


@pytest.mark.parametrize("kind", ["optimizer", "metric", "derived"])
def test_initial_upload_cannot_persist_invalid_scientific_rows(client, tmp_path, kind):
    path = tmp_path / "upload.pdf"
    options = {"headers": ("Method", "Adam", "SGD"),
               "note": "Table 1 columns are optimizers, not datasets."} if kind == "optimizer" else {
        "note": "In Table 1, Bench-R reports QXZ instead of accuracy."} if kind == "metric" else {
        "rows": (*ROWS, ("Gain over Network-A", "1.5", "2.1"))}
    pdf(path, **options)
    result = client.post("/api/papers/upload", files={"files": ("synthetic.pdf", path.read_bytes(), "application/pdf")})
    assert result.status_code == 200 and result.json()["errors"] == []
    pid = result.json()["papers"][0]["id"]
    rows = client.get(f"/api/papers/{pid}").json()["experiments"]
    assert len(rows) == (4 if kind == "derived" else 0)
    assert all(r["method"] in {"Network-A", "Network-B"} and r["dataset"] in {"Bench-Q", "Bench-R"} for r in rows)
    good = tmp_path / "good.pdf"
    pdf(good, rows=(("Different-A", "85", "79"), ("Different-B", "86", "81")))
    other = client.post("/api/papers/upload", files={"files": ("good.pdf", good.read_bytes(), "application/pdf")}).json()["papers"][0]["id"]
    report = client.post("/api/compare", json={"paper_ids": [pid, other]}).json()
    assert all(r["method"] != "Gain over Network-A" and r["dataset"] not in {"Adam", "SGD"} for r in report["experiments"])


@pytest.mark.parametrize("headers", [("Method", "Study-X", "Study-Y"), ("Method", "Atlas", "Nova")])
def test_explicit_result_column_roles_are_positive_not_evaluation_counts(tmp_path, headers):
    sources = pdf(tmp_path / "axis.pdf", headers=headers, definitions="",
                  note=f"Table 1 result columns are datasets: {headers[1]} and {headers[2]}.")
    rows = records(sources)
    assert len(rows) == 4 and {r["dataset"] for r in rows} == set(headers[1:])


def test_same_bbox_chunk_suffix_is_checked_without_joining_source_quotes(tmp_path):
    sources = pdf(tmp_path / "chunks.pdf", note="Table 1 reports results.")
    anchor = next(e for e in sources if "Table 1 reports results." in e["text"])
    sources.append({**deepcopy(anchor), "id": "split-suffix", "text": "Only Bench-Q is measured as a percentage."})
    assert records(sources) == []


def test_same_number_on_different_rows_cannot_launder_wrong_dataset(tmp_path):
    sources = pdf(tmp_path / "equal.pdf", rows=(("Network-A", "81.5", "83.0"), ("Network-B", "83.0", "80.2")))
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [model_row(sources, method="Network-B", dataset="Bench-R", value=83)]},
                                      sources, "synthetic-reliability", {}, 0)


def test_classification_accuracy_alias_uses_the_same_literal_caption_measurement(tmp_path):
    from backend.app.analysis import _canonical_metric
    sources = pdf(tmp_path / "metric.pdf")
    raw = model_row(sources)
    raw["metric"]["value"] = "classification accuracy"
    rows = providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)
    assert len(rows) == 1 and _canonical_metric(rows[0]["metric"]) == "accuracy"
    assert _canonical_metric("classification error") == "error_rate"


def test_context_id_is_not_a_citation_and_dataset_header_is_not_metric_evidence(tmp_path):
    sources = pdf(tmp_path / "metric.pdf")
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    aliases = {f"S{i+1}": e["id"] for i,e in enumerate(sources)}
    payload = providers._result_context_payload(sources, aliases)
    assert payload and all(c["column_role"] == "dataset" for c in payload)
    raw = model_row(sources)
    raw["value"]["evidence"][0]["id"] = "R1"
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", aliases, 0)
    raw = model_row(sources)
    raw["metric"]["evidence"] = [{"id": table["id"], "quote": table["text"].splitlines()[0]}]
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)


@pytest.mark.parametrize("mutation", [None, "other_table", "foreign_paper", "training_percentage"])
def test_only_explicit_same_table_uniform_body_units_support_a_measurement(tmp_path, mutation):
    note = "Results are summarized in Table 1. Reported numbers denote classification accuracy in percent."
    if mutation == "other_table":
        note = note.replace("Table 1", "Table 8")
    sources = pdf(tmp_path / "body-units.pdf", note=note)
    source = next(e for e in sources if "Reported numbers denote" in e["text"])
    if mutation == "foreign_paper":
        source["paper_id"] = "foreign"
    raw = model_row(sources)
    raw["metric"]["value"] = "classification accuracy"
    quote = "classification accuracy in percent"
    if mutation == "training_percentage":
        source["text"] += " Training labels cover 5%."
        quote = "Training labels cover 5%."
    raw["unit"]["evidence"] = [{"id": source["id"], "quote": quote}]
    if mutation is None:
        assert len(providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)) == 1
    else:
        with pytest.raises(ProviderError):
            providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)


def test_context_field_sources_are_literal_and_do_not_accept_wrong_returned_ownership(tmp_path):
    sources = pdf(tmp_path / "sources.pdf")
    aliases = {f"S{i+1}": e["id"] for i, e in enumerate(sources)}
    source_map = {e["id"]: e for e in sources}
    contexts = providers._result_context_payload(sources, aliases)
    assert len(contexts) == 4
    for context in contexts:
        for field in ("method", "dataset", "metric", "value", "unit"):
            assert context["field_sources"][field]
            for ref in context["field_sources"][field]:
                assert ref["id"] in aliases
                assert quote_matches_source(ref["quote"], source_map[aliases[ref["id"]]]["text"])
        assert context["column_label"] in context["field_sources"]["dataset"][0]["quote"]
    raw = model_row(sources, dataset="Bench-R")  # value is from Bench-Q, not Bench-R.
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)


@pytest.mark.parametrize("label", ["NewNet (ours)", "NewNet (this work)"])
def test_explicit_author_row_marker_only_prioritizes_context_without_relabeling(tmp_path, label):
    sources = pdf(tmp_path / "priority.pdf", rows=(ROWS[0], (label, "83.0 ± 0.7%", "80.2%")))
    original = deepcopy(sources)
    aliases = {f"S{i+1}": e["id"] for i, e in enumerate(sources)}
    contexts = providers._result_context_payload(sources, aliases)
    assert len(contexts) == 4 and [c["row_label"] for c in contexts[:2]] == [label, label]
    assert {c["row_label"] for c in contexts[2:]} == {"Network-A"}
    assert sources == original and len({c["context_id"] for c in contexts}) == 4


@pytest.mark.parametrize("append_caption", [False, True])
def test_dataset_column_name_cannot_also_prove_an_unreported_split(tmp_path, append_caption):
    sources = pdf(tmp_path / "split.pdf")
    raw = model_row(sources)
    raw["split"] = deepcopy(raw["dataset"])
    if append_caption:
        raw["split"]["evidence"].extend(raw["metric"]["evidence"])
    with pytest.raises(ProviderError):
        providers._checked_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {}, 0)
    accepted, rejected = providers._screen_extraction({"experiments": [raw]}, sources, "synthetic-reliability", {})
    assert not accepted and "split:field_role_conflict" in rejected[0]
    assert "field_literal_and_quote_passed" not in rejected[0]


@pytest.mark.parametrize("metric", ["Accuracy", "F1", "AUC"])
def test_unitless_pdf_is_not_promoted_to_percentage_results(tmp_path, metric):
    sources = pdf(tmp_path / "unitless.pdf", caption=f"Table 1: {metric} on datasets Bench-Q and Bench-R.",
                  rows=(("Network-A", "0.815", "0.781"), ("Network-B", "0.830", "0.802")), require_grid=False)
    # The existing PDF recovery path requires a caption or cell percentage.
    # Do not claim to have reproduced a downstream unit bug when it abstains
    # at this earlier boundary.
    assert not any(e.get("table_structure") == "caption_bounded_dataset_grid" for e in sources)
    assert records(sources) == []


@pytest.mark.parametrize("claim,verdict,expected", [
    ("Network-A在Bench-Q的准确率为81.5%。", "supported", "supported"),
    ("Network-A在Bench-R的准确率为81.5%。", "supported", "insufficient"),
    ("Network-B在Bench-Q的准确率为81.5%。", "supported", "insufficient"),
    ("Network-A在Bench-Q的错误率为81.5%。", "supported", "insufficient"),
    ("Network-A在Bench-Q的准确率为0.7%。", "supported", "insufficient"),
    ("Network-A在Bench-Q的准确率为80%。", "contradicted", "contradicted"),
])
def test_semantic_verdict_cannot_launder_a_number_from_another_table_cell(tmp_path, claim, verdict, expected):
    sources = pdf(tmp_path / "verdict.pdf")
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    caption = next(e for e in sources if e["id"] == table["caption_evidence_ids"][0])
    # All literal strings are present. Global number/name co-occurrence is NOT
    # enough to establish the row-column-measurement relationship.
    result = {"checks": [{"claim": claim, "verdict": verdict, "explanation": "See the table.",
               "evidence": [{"id": table["id"], "quote": table["text"]},
                            {"id": caption["id"], "quote": caption["text"]}]}]}
    checked, _, _ = providers._checked_verdict(result, sources, claim=claim)
    assert checked == expected


def test_plural_accuracies_caption_is_still_literal_metric_direction(tmp_path):
    sources = pdf(tmp_path / "plural.pdf", caption=CAPTION.replace("accuracy", "accuracies"))
    table = next(e for e in sources if e.get("table_structure") == "caption_bounded_dataset_grid")
    caption = next(e for e in sources if e["id"] == table["caption_evidence_ids"][0])
    claim = "Network-A在Bench-Q的准确率为81.5%。"
    result = {"checks": [{"claim": claim, "verdict": "supported", "explanation": "The exact table cell.",
              "evidence": [{"id": table["id"], "quote": table["text"]},
                           {"id": caption["id"], "quote": caption["text"]}]}]}
    assert providers._checked_verdict(result, sources, claim=claim)[0] == "supported"
