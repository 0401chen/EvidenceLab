"""Public synthetic source material only; original-PDF task/measurement scopes."""
from copy import deepcopy
import hashlib

import pymupdf
import pytest

from backend.app.analysis import extract_experiments
from backend.app.experiment_validation import verify_pdf_measurements, machine_measurement_verified
from backend.app.parsing import parse_document
from backend.app.task_attribution import machine_task_verified
from test_table_reliability_contracts import pdf


SETUP = ("We use Bench-Q dataset and Bench-R dataset. In these datasets, nodes are documents. "
         "Each document has a class label. For training, we only use 5 labels per class.")


def task_case(tmp_path, text=SETUP, heading="SEMI-SUPERVISED NODE CLASSIFICATION"):
    original = tmp_path / "base.pdf"
    pdf(original, rows=(("GraphNet (ours)", "81.5", "78.1"), ("Baseline", "83.0 ± 0.7%", "80.2%")),
        definitions=text)
    path = tmp_path / "scoped.pdf"
    with pymupdf.open(original) as doc:
        if heading:
            doc[0].insert_text((50, 22), heading, fontsize=10)
        doc.save(path)
    sources = parse_document(path, "synthetic-reliability")["evidence"]
    paper = {"id": "synthetic-reliability", "title": "Synthetic"}
    rows = extract_experiments(sources, paper["id"])
    return path, sources, paper, rows


def checked(case):
    path, sources, paper, rows = case
    return verify_pdf_measurements(rows, paper, sources, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_dataset_subset_task_has_pdf_quotes_without_affecting_measurement_scope(tmp_path):
    case = task_case(tmp_path)
    result = checked(case)
    assert len(result) == 4
    author = result[0]
    assert author["task"] == "semi-supervised node classification"
    assert author["task_label_zh"] == "半监督节点分类"
    assert machine_task_verified(author, case[1])
    assert machine_measurement_verified(author, case[1])
    assert author["machine_verification"]["scope"] == ["method", "dataset", "metric", "value", "unit"]
    assert author["review_required"] is True
    assert all(r["page"] == 1 and r["quote"] for r in author["evidence_quotes"]["task"])
    assert all(r["task"] is None for r in result if r["method"] == "Baseline")
    assert all(r.get("split") is None and not r.get("conditions") for r in result)
    # Rechecking a certified task is valid; arbitrary task additions still fail.
    case[3][:] = result
    assert checked(case)[0]["machine_verification"]["status"] == "verified"


@pytest.mark.parametrize("text,heading", [
    ("Bench-Q dataset and Bench-R dataset. We present semi-supervised node classification.", "SEMI-SUPERVISED NODE CLASSIFICATION"),
    (SETUP, "IMAGE CLASSIFICATION"),
    (SETUP, "FULLY-SUPERVISED NODE CLASSIFICATION"),
    (SETUP, "NODE CLUSTERING"),
    (SETUP, ""),
    (SETUP.replace("only use 5 labels per class", "use all labels"), "SEMI-SUPERVISED NODE CLASSIFICATION"),
    (SETUP.replace("Bench-Q", "Other-Q").replace("Bench-R", "Other-R"), "SEMI-SUPERVISED NODE CLASSIFICATION"),
    (SETUP + " We evaluate image classification on Bench-Q and Bench-R.", "SEMI-SUPERVISED NODE CLASSIFICATION"),
])
def test_unknown_cross_experiment_conflicting_or_paper_goal_cannot_fill_task(tmp_path, text, heading):
    result = checked(task_case(tmp_path, text, heading))
    assert all(r.get("task") is None for r in result)


@pytest.mark.parametrize("change", ["task", "quote", "source", "label", "table"])
def test_task_or_source_edits_invalidate_scope_certificates(tmp_path, change):
    case = task_case(tmp_path)
    item = checked(case)[0]
    sources = deepcopy(case[1])
    if change == "task": item["task"] = "object detection"
    if change == "quote": item["evidence_quotes"]["task"][0]["quote"] = "Forged quotation"
    if change == "source": sources[-1]["text"] += " changed"
    if change == "label": item["task_label_zh"] = "物体检测"
    if change == "table": item["table_context"]["column_label"] = "Other-Q"
    assert not machine_task_verified(item, sources)
    assert not machine_measurement_verified(item, sources)


def test_unverified_nonempty_task_does_not_bypass_protocol_guard(tmp_path):
    case = task_case(tmp_path)
    case[3][0]["task"] = "semi-supervised node classification"
    item = checked(case)[0]
    assert item["machine_verification"]["reason"] == "protocol_outside_verified_scope"


def test_explicit_dataset_specific_task_is_not_tied_to_node_classification(tmp_path):
    case = task_case(tmp_path,
        text="We evaluate image classification on Bench-Q and Bench-R. Bench-Q dataset and Bench-R dataset are evaluated.",
        heading="IMAGE CLASSIFICATION")
    item = checked(case)[0]
    assert item["task"] == "image classification"
    assert machine_task_verified(item, case[1]) and machine_measurement_verified(item, case[1])


def test_reanalysis_is_pending_and_does_not_overwrite_saved_experiments(client, tmp_path):
    path, _, _, _ = task_case(tmp_path)
    pid = client.post("/api/papers/upload", files={"files": ("source.pdf", path.read_bytes(), "application/pdf")}).json()["papers"][0]["id"]
    before = client.get(f"/api/papers/{pid}").json()
    response = client.post(f"/api/papers/{pid}/extract", json={"mode": "rule"})
    assert response.status_code == 200
    after = response.json()
    assert after["experiments"] == before["experiments"]
    candidate = after["extraction_candidates"][0]
    assert candidate["status"] == "pending"
    assert candidate["experiments"][0]["task"] == "semi-supervised node classification"


def test_forged_task_source_never_receives_task_certificate_or_erases_valid_measurement(tmp_path):
    case = task_case(tmp_path, text="Bench-Q dataset and Bench-R dataset are evaluated.")
    source = next(e for e in case[1] if "are evaluated" in e["text"])
    source["text"] = SETUP  # Parser metadata lies about the PDF text.
    result = checked(case)
    assert result[0].get("task") is None
    assert not machine_task_verified(result[0], case[1])
    # Dataset-role support itself was tampered with, so this measurement also
    # correctly abstains rather than trusting a forged parser support paragraph.
    assert result[0]["machine_verification"]["status"] == "abstained"


def test_unproven_supplemental_task_cannot_invalidate_correct_measurement(tmp_path):
    case = task_case(tmp_path, heading="UNRELATED HEADING")
    source = next(e for e in case[1] if e["text"] == "UNRELATED HEADING")
    source["text"] = "SEMI-SUPERVISED NODE CLASSIFICATION"
    result = checked(case)
    assert result[0].get("task") is None
    assert result[0]["machine_verification"]["status"] == "verified"


def test_manual_edit_clears_task_certificate_but_preserves_historical_record(client, tmp_path):
    path, _, _, _ = task_case(tmp_path)
    pid = client.post("/api/papers/upload", files={"files": ("source.pdf", path.read_bytes(), "application/pdf")}).json()["papers"][0]["id"]
    item = client.get(f"/api/papers/{pid}").json()["experiments"][0]
    assert "task_verification" in item
    edited = client.patch(f"/api/experiments/{item['id']}", json={"task": "image classification"})
    assert edited.status_code == 200
    assert "task_verification" not in edited.json() and "task_label_zh" not in edited.json()
    history = client.get(f"/api/experiments/{item['id']}/history").json()
    assert any("task_verification" in h["experiment"] for h in history)
