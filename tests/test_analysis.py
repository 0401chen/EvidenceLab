"""Regression cases for retrieval and conservative experimental comparison."""

import pytest

from backend.app.analysis import (
    answer_local,
    compare_experiments,
    extract_experiments,
    retrieve_evidence,
)


def test_offline_answer_without_relevant_evidence_is_insufficient(evidence_factory):
    evidence = [evidence_factory("A cobalt catalyst was examined at room temperature.")]
    result = answer_local("How does a lunar telescope correct atmospheric distortion?", evidence)
    assert result["verdict"] == "insufficient"
    assert result["citations"] == []


@pytest.mark.parametrize(
    ("question", "relevant", "unrelated"),
    [
        (
            "What is the measured catalyst efficiency?",
            "The measured catalyst efficiency is 84 percent.",
            "A lunar telescope uses an optical mirror.",
        ),
        (
            "催化剂的转化率是多少？",
            "实验报告催化剂的转化率为84%，温度为25摄氏度。",
            "望远镜采用光学镜面观察月球。",
        ),
    ],
)
def test_retrieval_handles_english_and_chinese(question, relevant, unrelated, evidence_factory):
    evidence = [
        evidence_factory(unrelated, "irrelevant"),
        evidence_factory(relevant, "relevant"),
    ]
    found = retrieve_evidence(question, evidence)
    assert found and found[0]["id"] == "relevant"
    result = answer_local(question, evidence)
    assert result["verdict"] == "evidence_found"
    assert "relevant" in {e["id"] for e in result["citations"]}


def test_rule_extraction_keeps_zero_value_and_field_evidence(evidence_factory):
    evidence = [evidence_factory(
        "Experiment: zero baseline\nMethod: AlphaNet\nDataset: DemoSet\n"
        "Split: test-A\nMetric: error rate\nValue: 0\nUnit: %\nSample size: 100"
    )]
    experiments = extract_experiments(evidence, "paper-a")
    assert experiments
    assert experiments[0]["value"] == 0
    assert experiments[0]["evidence_ids"] == ["ev-a"]
    assert "ev-a" in experiments[0]["field_evidence"]["value"]


def test_missing_metric_does_not_make_experiments_directly_comparable(experiment_factory):
    comparisons = compare_experiments([
        experiment_factory(),
        experiment_factory("exp-b", "paper-b", metric=None),
    ])
    assert len(comparisons) == 1
    assert comparisons[0]["comparability"] == "insufficient"
    assert comparisons[0]["verdict"] == "insufficient"


def test_unreported_conditions_prevent_a_direct_comparison(experiment_factory):
    comparison = compare_experiments([
        experiment_factory(conditions={}),
        experiment_factory("exp-b", "paper-b", conditions={}),
    ])[0]
    assert comparison["comparability"] == "insufficient"
    assert comparison["verdict"] == "insufficient"


def test_different_test_splits_prevent_direct_comparison(experiment_factory):
    comparisons = compare_experiments([
        experiment_factory(),
        experiment_factory("exp-b", "paper-b", split="test-B", value=85.0),
    ])
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["comparability"] in {"conditional", "incomparable"}
    assert comparison["verdict"] != "potential_conflict"
    assert comparison["reasons"]


def test_different_datasets_are_not_directly_comparable(experiment_factory):
    comparison = compare_experiments([
        experiment_factory(),
        experiment_factory("exp-b", "paper-b", dataset="OtherSet"),
    ])[0]
    assert comparison["comparability"] in {"conditional", "incomparable"}
    assert comparison["verdict"] != "potential_conflict"


def test_different_sample_sizes_require_conditional_interpretation(experiment_factory):
    comparison = compare_experiments([
        experiment_factory(sample_size="100"),
        experiment_factory("exp-b", "paper-b", sample_size="200"),
    ])[0]
    assert comparison["comparability"] == "conditional"
    assert comparison["verdict"] != "potential_conflict"


def test_unequal_results_alone_do_not_establish_a_conclusion_conflict(experiment_factory):
    comparison = compare_experiments([
        experiment_factory(),
        experiment_factory("exp-b", "paper-b", value=85.0),
    ])[0]
    assert comparison["comparability"] == "direct"
    assert comparison["verdict"] == "different"
    assert abs(comparison["delta"]) == 5.0


def test_zero_is_a_valid_comparable_result(experiment_factory):
    comparison = compare_experiments([
        experiment_factory(value=0.0),
        experiment_factory("exp-b", "paper-b", value=0.0),
    ])[0]
    assert comparison["comparability"] == "direct"
    assert comparison["verdict"] == "consistent"
    assert comparison["delta"] == 0.0


def test_opposite_explicit_claims_under_the_same_protocol_are_flagged(experiment_factory, evidence_factory):
    shared = {"protocol": "fixed", "claim": "AlphaNet improves accuracy"}
    positive = "Under the fixed protocol, results support the claim that AlphaNet improves accuracy."
    negative = "Under the fixed protocol, results contradict the claim that AlphaNet improves accuracy."
    comparison = compare_experiments([
        experiment_factory(conditions={**shared, "direction": "support"}, field_evidence={"conditions": ["ev-exp-a"]},
                           evidence_quotes={"conditions": [{"id": "ev-exp-a", "quote": positive}]}),
        experiment_factory(
            "exp-b", "paper-b", value=85,
            conditions={**shared, "direction": "oppose"},
            field_evidence={"conditions": ["ev-exp-b"]},
            evidence_quotes={"conditions": [{"id": "ev-exp-b", "quote": negative}]},
        ),
    ], [evidence_factory(positive, "ev-exp-a"), evidence_factory(negative, "ev-exp-b", "paper-b")])[0]
    assert comparison["comparability"] == "direct"
    assert comparison["verdict"] == "potential_conflict"
    assert comparison["reasons"]
    assert len(comparison["conflict_evidence"]) == 2


@pytest.mark.parametrize("field,value", [("task", None), ("sample_size", None)])
def test_unknown_task_or_sample_size_prevents_direct_comparison(experiment_factory, field, value):
    result = compare_experiments([experiment_factory(), experiment_factory("exp-b", "paper-b", **{field: value})])[0]
    assert result["comparability"] == ("insufficient" if field == "task" else "conditional")
    assert result["verdict"] == "insufficient"
    assert result["delta"] is None
    if field == "sample_size":
        assert result["alignment"]["sample_size"]["status"] == "missing"
        assert any("不计算差值" in reason for reason in result["reasons"])


def test_different_tasks_and_units_never_rank_values(experiment_factory):
    result = compare_experiments([experiment_factory(), experiment_factory("exp-b", "paper-b", task="regression", unit="fraction")])[0]
    assert result["comparability"] == "incomparable"
    assert result["verdict"] == "conditions_differ"
    assert result["delta"] is None
    assert result["unit_conversion"] is None


def test_condition_change_is_not_itself_a_numerical_difference(experiment_factory):
    result = compare_experiments([experiment_factory(value=90), experiment_factory("exp-b", "paper-b", value=90, method="BetaNet")])[0]
    assert result["comparability"] == "conditional"
    assert result["verdict"] == "conditions_differ"
    assert result["delta"] is None


def test_tiny_real_difference_is_not_rounded_to_consistency(experiment_factory):
    result = compare_experiments([experiment_factory(value=0), experiment_factory("exp-b", "paper-b", value=1e-11)])[0]
    assert result["verdict"] == "different"
    assert result["delta"] == 1e-11


def test_claim_labels_without_opposite_source_assertions_do_not_establish_conflict(experiment_factory):
    shared = {"protocol": "fixed", "claim": "AlphaNet improves accuracy"}
    result = compare_experiments([experiment_factory(conditions={**shared, "direction": "support"}),
                                 experiment_factory("exp-b", "paper-b", value=85, conditions={**shared, "direction": "oppose"})])[0]
    assert result["verdict"] == "different"
    assert result["conflict_evidence"] == []


@pytest.mark.parametrize("quote,direction", [
    ("We do not confirm that AlphaNet improves accuracy.", "support"),
    ("No evidence supports the claim that AlphaNet improves accuracy.", "support"),
    ("The results do not contradict the claim that AlphaNet improves accuracy.", "oppose"),
    ("The results do not support the claim that AlphaNet improves accuracy.", "oppose"),
    ("Our findings support investigating whether alphanet improves accuracy.", "support"),
    ("Our findings support a different hypothesis; alphanet improves accuracy is undecided.", "support"),
])
def test_negation_and_absence_of_support_are_not_opposite_source_assertions(experiment_factory, evidence_factory, quote, direction):
    from backend.app.analysis import _grounded_claim
    experiment = experiment_factory(field_evidence={"conditions": ["ev-exp-a"]},
                                    evidence_quotes={"conditions": [{"id": "ev-exp-a", "quote": quote}]})
    known = {"ev-exp-a": evidence_factory(quote, "ev-exp-a")}
    assert _grounded_claim(experiment, known, "alphanet improves accuracy", direction) == []


def test_equivalent_sample_count_notation_does_not_change_conditions(experiment_factory):
    comparison = compare_experiments([experiment_factory(sample_size="1000"),
                                      experiment_factory("exp-b", "paper-b", sample_size="1,000")])[0]
    assert comparison["comparability"] == "direct"
    assert comparison["alignment"]["sample_size"]["status"] == "same"
