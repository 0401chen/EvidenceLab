"""The workflow must reduce unsupported results rather than trust agent labels."""

from backend.app.orchestration import run_comparison


def test_comparison_workflow_records_all_four_actual_stages(experiment_factory, evidence_factory):
    from backend.app.analysis import extract_experiments

    evidence = [
        evidence_factory("Method: AlphaNet\nTask: classification\nDataset: DemoSet\nSplit: test-A\n"
                         "Metric: accuracy\nValue: 90\nUnit: %\nSample size: 100\n"
                         "Conditions: temperature=25 C; protocol=fixed", "ev-exp-a", "paper-a"),
        evidence_factory("Method: AlphaNet\nTask: classification\nDataset: DemoSet\nSplit: test-A\n"
                         "Metric: accuracy\nValue: 85\nUnit: %\nSample size: 100\n"
                         "Conditions: temperature=25 C; protocol=fixed", "ev-exp-b", "paper-b"),
    ]
    experiments = [*extract_experiments(evidence, "paper-a"), *extract_experiments(evidence, "paper-b")]
    result = run_comparison(experiments, evidence)
    assert len(result["traces"]) == 4
    assert [trace["agent"] for trace in result["traces"]] == ["实验整理 Agent", "条件对齐 Agent", "差异分析 Agent", "证据核验 Agent"]
    assert all(trace["status"] == "completed" for trace in result["traces"])
    assert result["comparisons"][0]["verdict"] == "different"
    assert set(result["comparisons"][0]["evidence_ids"]) <= {e["id"] for e in evidence}
    for trace in result["traces"]:
        assert trace["kind"] == "rule"
        assert trace["input"] and trace["output"]
        assert trace["duration_ms"] >= 0
        assert trace["started_at"] <= trace["finished_at"]


def test_workflow_downgrades_a_comparison_with_fabricated_evidence(experiment_factory, evidence_factory):
    experiments = [
        experiment_factory(evidence_ids=["ev-real"]),
        experiment_factory("exp-b", "paper-b", evidence_ids=["ev-fabricated"]),
    ]
    evidence = [evidence_factory("A real source block.", "ev-real")]
    result = run_comparison(experiments, evidence)
    comparison = result["comparisons"][0]
    assert comparison["verdict"] == "insufficient"
    assert "ev-fabricated" not in comparison["evidence_ids"]
    assert comparison["evidence_ids"] == ["ev-real"]


def test_existing_evidence_from_the_wrong_paper_cannot_validate_a_comparison(experiment_factory, evidence_factory):
    evidence = [evidence_factory("A real passage in paper A.", "own", "paper-a"),
                evidence_factory("A real passage in paper B.", "foreign", "paper-b")]
    result = run_comparison([experiment_factory(evidence_ids=["foreign"]),
                             experiment_factory("exp-b", "paper-b", evidence_ids=["foreign"], value=85)], evidence)
    comparison = result["comparisons"][0]
    assert comparison["verdict"] == "insufficient"
    assert comparison["comparability"] == "insufficient"
    assert comparison["delta"] is None


def test_a_manually_changed_number_without_its_own_source_cannot_be_ranked(experiment_factory, evidence_factory):
    evidence = [evidence_factory("Value: 90", "ev-exp-a"), evidence_factory("Value: 85", "ev-exp-b", "paper-b")]
    result = run_comparison([experiment_factory(value=0, field_evidence={}), experiment_factory("exp-b", "paper-b", value=85)], evidence)
    assert result["comparisons"][0]["verdict"] == "insufficient"
    assert result["comparisons"][0]["delta"] is None


def test_a_different_number_in_the_same_paragraph_is_not_result_evidence(experiment_factory, evidence_factory):
    evidence = [evidence_factory("Accuracy: 90%. Sample size: 100.", "ev-exp-a"),
                evidence_factory("Value: 85", "ev-exp-b", "paper-b")]
    row = experiment_factory(value=100, evidence_quotes={})
    result = run_comparison([row, experiment_factory("exp-b", "paper-b", value=85)], evidence)
    assert result["comparisons"][0]["verdict"] == "insufficient"
    assert result["comparisons"][0]["delta"] is None
