"""Synthetic measurements: counts/configuration are not evaluation results."""
import pytest

from backend.app.analysis import extract_experiments
from backend.app.experiment_validation import prepare_experiments
from backend.app.grounding import quote_matches_source


@pytest.mark.parametrize("text", [
    "accuracy evaluated over 100 runs",
    "accuracy averaged across 100 trials",
    "accuracy measured after 100 epochs",
    "accuracy evaluated using 100 random seeds",
    "The mean accuracy of 37 runs is reported in the table.",
    "We report mean accuracy\nof 100 runs with independent initializations.",
    "The accuracy of 43 trials is averaged for evaluation.",
    "accuracy was 100 epochs",
    "accuracy of 17 iterations",
    "accuracy of 21 random seeds",
    "accuracy of 72 independent test samples",
    "accuracy of 48 examples",
    "accuracy of 12 batches",
    "accuracy of 100 (runs)",
    "accuracy of 100-run evaluations",
    "accuracy of 1e2 independent runs",
    "accuracy of 0.01 learning rate",
    "accuracy of 0.5 dropout rate",
    "accuracy of 42 seed",
    "accuracy of 100 train-\ning samples",
    "准确率为100次重复实验",
    "The learning rate was 0.01; accuracy is discussed separately.",
])
def test_configuration_counts_are_not_accuracy_results(evidence_factory, text):
    assert extract_experiments([evidence_factory(text)], "paper-a") == []


@pytest.mark.parametrize("text,metric,value,unit", [
    ("accuracy 81.5%", "accuracy", 81.5, "%"),
    ("accuracy was 81.5%", "accuracy", 81.5, "%"),
    ("accuracy of 83%", "accuracy", 83, "%"),
    ("accuracy 100%", "accuracy", 100, "%"),
    ("The accuracy was 0.815.", "accuracy", 0.815, None),
    ("The accuracy equals 1.0.", "accuracy", 1.0, None),
    ("F1-score was 0.83.", "f1", 0.83, None),
    ("AUC is 0.95.", "auc", 0.95, None),
    ("AUROC reached 0.96.", "auc", 0.96, None),
    ("RMSE was 1.2e-3.", "rmse", 0.0012, None),
    ("MAE was 0.02.", "mae", 0.02, None),
    ("MSE was 0.001.", "mse", 0.001, None),
    ("准确率为100％。", "accuracy", 100, "%"),
    ("accuracy was 100% over 73 runs", "accuracy", 100, "%"),
    ("accuracy was 81.5%; runs were repeated 100 times.", "accuracy", 81.5, "%"),
    ("accuracy was 0.815; sample size was 100.", "accuracy", 0.815, None),
    ("accuracy was 0.815 (random seeds: 100).", "accuracy", 0.815, None),
    ("accuracy was 100% (runs = 37).", "accuracy", 100, "%"),
])
def test_direct_measurements_preserve_literal_field_provenance(evidence_factory, text, metric, value, unit):
    source = evidence_factory(text, "stable-source", page=3)
    rows = extract_experiments([source], "paper-a")
    assert len(rows) == 1
    item = prepare_experiments(rows, {"id": "paper-a", "title": "Synthetic"}, [source])[0]
    assert (item["metric"], item["value"], item["unit"]) == (metric, value, unit)
    assert item["review_required"] and item["method"] is None and item["dataset"] is None
    assert item["source_pages"] == [3]
    assert item["evidence_ids"] == [source["id"]]
    for field in ("metric", "value", "unit") if unit else ("metric", "value"):
        assert item["field_evidence"][field] == [source["id"]]
        assert all(ref["id"] == source["id"] and quote_matches_source(ref["quote"], source["text"])
                   for ref in item["evidence_quotes"][field])


@pytest.mark.parametrize("text,expected", [
    ("The accuracy was 81.5% over 100 runs.", [("accuracy", 81.5)]),
    ("After 100 epochs, accuracy was 83% using 100 random seeds.", [("accuracy", 83)]),
    ("We summarize accuracy of 100 trials. Accuracy was 0.83.", [("accuracy", 0.83)]),
    ("The mean accuracy of 100 runs is summarized; F1 was 0.77.", [("f1", 0.77)]),
    ("With learning rate 0.01 and 64 samples, RMSE was 0.15.", [("rmse", 0.15)]),
])
def test_measurements_and_configuration_in_one_block_do_not_exchange_roles(evidence_factory, text, expected):
    source = evidence_factory(text)
    rows = extract_experiments([source], "paper-a")
    assert [(row["metric"], row["value"]) for row in rows] == expected
    assert all(row["value"] != 100 for row in rows)
    # Configuration remains accessible in immutable source context; no inferred
    # owner/dataset or second numeric result is created from it.
    assert all(row["evidence_ids"] == [source["id"]] for row in rows)


def test_same_numeric_value_in_a_count_and_real_result_keeps_the_correct_quote(evidence_factory):
    source = evidence_factory("Mean accuracy of 100 runs was aggregated. The accuracy was 100%.")
    rows = extract_experiments([source], "paper-a")
    assert len(rows) == 1 and rows[0]["value"] == 100 and rows[0]["unit"] == "%"
    assert rows[0]["evidence_quotes"]["value"] == [{"id": source["id"], "quote": "accuracy was 100%"}]


def test_rule_api_rejects_count_only_without_modifying_saved_work(client):
    text = b"# Synthetic protocol\n\nWe report mean accuracy of 37 runs with random initializations."
    p = client.post("/api/papers/upload", files={"files": ("protocol.md", text, "text/markdown")}).json()["papers"][0]
    before = client.get(f"/api/papers/{p['id']}").json()
    assert before["experiments"] == []
    result = client.post(f"/api/papers/{p['id']}/extract", json={"mode": "rule", "reparse": False})
    assert result.status_code == 422
    assert client.get(f"/api/papers/{p['id']}").json() == before


def test_rule_api_creates_only_the_real_result_as_a_pending_candidate(client):
    p = client.post("/api/papers/upload", files={"files": (
        "mixed.md", b"# Synthetic result\n\nThe accuracy was 81.5% over 100 runs.", "text/markdown")}).json()["papers"][0]
    before = client.get(f"/api/papers/{p['id']}").json()
    result = client.post(f"/api/papers/{p['id']}/extract", json={"mode": "rule", "reparse": False})
    assert result.status_code == 200
    after = result.json()
    assert after["experiments"] == before["experiments"]
    candidate = next(c for c in after["extraction_candidates"] if c["id"] == after["latest_extraction_candidate_id"])
    assert candidate["status"] == "pending" and candidate["mode"] == "rule"
    assert [(e["metric"], e["value"], e["unit"]) for e in candidate["experiments"]] == [("accuracy", 81.5, "%")]
