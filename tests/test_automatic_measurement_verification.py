"""Synthetic raw-PDF checks independent of the extracted row's assertions."""
from copy import deepcopy
import hashlib

import pytest

from backend.app import experiment_validation as validation
from backend.app.analysis import extract_experiments
from test_table_reliability_contracts import pdf


def case(tmp_path, **kwargs):
    path = tmp_path / "original.pdf"
    sources = pdf(path, **kwargs)
    paper = {"id": "synthetic-reliability", "title": "Synthetic"}
    rows = validation.prepare_experiments(extract_experiments(sources, paper["id"]), paper, sources)
    return path, sources, paper, rows


def verify(path, sources, paper, rows, digest=None):
    return validation.verify_pdf_measurements(rows, paper, sources, path,
                                             digest or hashlib.sha256(path.read_bytes()).hexdigest())


def test_raw_pdf_verifies_measurement_without_claiming_human_or_protocol_review(tmp_path):
    path, sources, paper, rows = case(tmp_path)
    result = verify(path, sources, paper, rows)
    assert len(result) == 4
    for item in result:
        check = item["machine_verification"]
        assert check["status"] == "verified"
        assert check["scope"] == ["method", "dataset", "metric", "value", "unit"]
        assert check["pdf_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["review_required"] is True  # Existing manual workflow unchanged.
        assert check["human_reviewed"] is False
        assert "split" in check["unverified_fields"]
        assert validation.machine_measurement_verified(item, sources)
    annotated = next(r for r in result if "±" in r["table_context"]["value_cell_text"])
    assert annotated["table_context"]["uncertainty"]["kind"] == "unspecified"


@pytest.mark.parametrize("field,value", [("dataset", "Bench-R"), ("method", "Network-B"),
    ("value", .7), ("value", 100), ("metric", "error_rate"), ("unit", "ms"),
    ("split", "test"), ("sample_size", "100"), ("conditions", {"runs": "100"})])
def test_wrong_or_extra_fields_cannot_receive_machine_approval(tmp_path, field, value):
    path, sources, paper, rows = case(tmp_path)
    rows[0][field] = value
    assert verify(path, sources, paper, rows)[0]["machine_verification"]["status"] == "abstained"


def test_coherently_forged_parser_row_is_rejected_against_original_pdf(tmp_path):
    path, sources, paper, rows = case(tmp_path)
    # The table, row, quotes and coordinates all agree with each other but lie
    # about the raw PDF. Re-running the same extractor is not an independent check.
    for source in sources:
        if source.get("table_structure") == "caption_bounded_dataset_grid":
            source["text"] = source["text"].replace("81.5", "91.5")
            for row in source["coordinate_rows"]:
                for cell in row["cells"]:
                    cell["text"] = cell["text"].replace("81.5", "91.5")
    rows = validation.prepare_experiments(extract_experiments(sources, paper["id"]), paper, sources)
    assert rows[0]["value"] == 91.5
    assert verify(path, sources, paper, rows)[0]["machine_verification"]["status"] == "abstained"


def test_equal_number_in_other_row_cannot_supply_this_cells_quote(tmp_path):
    path, sources, paper, rows = case(tmp_path, rows=(("Network-A", "81.5%", "78.1"),
                                                     ("Network-B", "81.5%", "80.2%")))
    rows[0]["evidence_quotes"]["value"] = deepcopy(rows[1]["evidence_quotes"]["value"])
    assert verify(path, sources, paper, rows)[0]["machine_verification"]["status"] == "abstained"


@pytest.mark.parametrize("change", ["header", "caption", "role", "column_box", "page", "hash"])
def test_missing_or_conflicting_physical_source_abstains(tmp_path, change):
    path, sources, paper, rows = case(tmp_path)
    table = next(s for s in sources if s.get("coordinate_rows"))
    if change == "header":
        table["text"] = table["text"].replace("|Bench-Q|Bench-R|", "|Bench-R|Bench-Q|")
    elif change == "caption":
        sources[:] = [s for s in sources if s["id"] not in table["caption_evidence_ids"]]
    elif change == "role":
        sources[:] = [s for s in sources if "dataset and" not in s["text"]]
    elif change == "column_box":
        table["coordinate_rows"][0]["cells"][1]["bbox"] = table["coordinate_rows"][0]["cells"][2]["bbox"]
    elif change == "page":
        table["page"] = 2
    result = verify(path, sources, paper, rows, "0" * 64 if change == "hash" else None)
    assert result[0]["machine_verification"]["status"] == "abstained"


def test_certificate_invalidates_on_field_or_source_change(tmp_path):
    path, sources, paper, rows = case(tmp_path)
    item = verify(path, sources, paper, rows)[0]
    changed = deepcopy(item)
    changed["value"] += 1
    assert not validation.machine_measurement_verified(changed, sources)
    changed = deepcopy(sources)
    changed[0]["text"] += " changed"
    assert not validation.machine_measurement_verified(item, changed)
    item["id"] = "candidate-new-id"
    assert validation.machine_measurement_verified(item, sources)


def test_literal_100_percent_is_a_valid_measurement(tmp_path):
    path, sources, paper, rows = case(tmp_path, rows=(("Network-A", "100%", "78.1"),
                                                     ("Network-B", "83.0 ± 0.7%", "80.2%")))
    result = verify(path, sources, paper, rows)
    assert result[0]["value"] == 100
    assert result[0]["machine_verification"]["status"] == "verified"


def test_default_upload_and_bounded_candidate_prioritize_source_marked_methods(client, tmp_path):
    path = tmp_path / "many.pdf"
    rows = tuple((f"Baseline-{i}", str(60 + i), str(70 + i)) for i in range(6))
    rows += (("NewNet (ours)", "88.0 ± 0.2%", "86.5"),)
    # Both datasets occur in the first twelve under the new generic priority;
    # the extractor must not use a known paper/model/dataset/value whitelist.
    # Use a caption with explicit roles to avoid the fixture's fixed body at y315.
    pdf(path, rows=rows, caption="Table 1: Accuracy (%) on datasets Bench-Q and Bench-R.", definitions="")
    upload = client.post("/api/papers/upload", files={"files": ("synthetic.pdf", path.read_bytes(), "application/pdf")})
    assert upload.status_code == 200
    pid = upload.json()["papers"][0]["id"]
    detail = client.get(f"/api/papers/{pid}").json()
    assert len(detail["experiments"]) == 14
    assert detail["experiments"][0]["method"] == "NewNet (ours)"
    assert detail["experiments"][0]["machine_verification"]["status"] == "verified"
    extraction = client.post(f"/api/papers/{pid}/extract", json={})
    assert extraction.status_code == 200
    candidate = extraction.json()["extraction_candidates"][0]
    assert len(candidate["experiments"]) == 12 and candidate["status"] == "pending"
    assert candidate["extraction_scope"]["truncated"]
    assert len([r for r in candidate["experiments"] if r["method"] == "NewNet (ours)"]) == 2
    assert client.get(f"/api/papers/{pid}").json()["experiments"] == detail["experiments"]


def test_edit_invalidates_current_certificate_but_retains_historical_record(client, tmp_path):
    path, _, _, _ = case(tmp_path)
    pid = client.post("/api/papers/upload", files={"files": ("synthetic.pdf", path.read_bytes(), "application/pdf")}).json()["papers"][0]["id"]
    detail = client.get(f"/api/papers/{pid}").json()
    item = detail["experiments"][0]
    assert item["machine_verification"]["status"] == "verified"
    response = client.patch(f"/api/experiments/{item['id']}", json={"value": 99})
    assert response.status_code == 200
    assert "machine_verification" not in response.json()
    assert client.get(f"/api/experiments/{item['id']}/history").status_code == 200


def test_comparison_checks_each_experiment_once_and_keeps_all_rejections(monkeypatch, experiment_factory, evidence_factory):
    from backend.app import providers
    from backend.app.orchestration import run_comparison
    from collections import Counter
    experiments = [experiment_factory(f"e{i}", f"p{i}") for i in range(4)]
    sources = [evidence_factory("Value: 90", f"ev-e{i}", f"p{i}") for i in range(4)]
    original = providers._measurement_anchor_conflicts
    calls = Counter()
    def tracked(item, evidence):
        calls[item["id"]] += 1
        return original(item, evidence)
    monkeypatch.setattr(providers, "_measurement_anchor_conflicts", tracked)
    output = run_comparison(experiments, sources)
    assert len(output["comparisons"]) == 6
    assert all(c["comparability"] == "insufficient" and c["delta"] is None for c in output["comparisons"])
    assert list(calls.values()) == [1] * 4
    run_comparison(experiments, sources)
    assert list(calls.values()) == [2] * 4  # No cross-request/global cache.


def test_abstained_physical_check_is_not_overridden_by_literal_comparison(tmp_path):
    from backend.app.orchestration import run_comparison
    path, sources, paper, rows = case(tmp_path)
    items = verify(path, sources, paper, rows, "0" * 64)
    other = deepcopy(items[0])
    other["id"] = "other"
    other["paper_id"] = "other-paper"
    result = run_comparison([items[0], other], sources)
    assert result["comparisons"][0]["verdict"] == "insufficient"
    assert any("machine_measurement" in r for r in result["comparisons"][0]["reasons"])
