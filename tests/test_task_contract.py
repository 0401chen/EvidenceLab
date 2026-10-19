"""Task completeness and experiment identity must precede result labels."""
from copy import deepcopy

import pytest

from backend.app.errors import ProviderError
from backend.app.task_contract import check_findings, semantic_pairs, validate_plan


def sources(*texts):
    evidence = [{"id": f"e{index}", "paper_id": "paper", "page": index,
                 "text": text, "source_type": "text"} for index, text in enumerate(texts, 1)]
    fragments = {f"F{index}": {"source": item, "quote": item["text"], "start": 0, "end": len(item["text"])}
                 for index, item in enumerate(evidence, 1)}
    return evidence, fragments


def field(text, handle="F1"):
    return {"text": text, "evidence": [{"handle": handle}]}


def observation(subject="Model-Z", value="12", protocol="single-crop", quote=None, handle="F1"):
    fields = {"subject": field(subject, handle), "metric": field("error", handle), "value": field(value, handle)}
    if protocol:
        fields["protocol"] = field(protocol, handle)
    return {"fields": fields, "binding": {"kind": "text", "handle": handle,
                                           "quote": quote or f"{subject} error is {value} under {protocol}."}}


def requirement(mode="verify", value="12", protocol="single-crop", rid="R1"):
    expected = {"subject": "Model-Z", "metric": "error", "value": value, "protocol": protocol} if mode == "verify" else {}
    return {"id": rid, "request": "Model-Z result", "subject": "Model-Z",
            "fields": ["subject", "metric", "value", "protocol"], "expected": expected,
            "operator": "eq", "mode": mode}


def findings(obs, missing=(), rid="R1"):
    return {"items": [{"requirement_id": rid, "observations": obs, "missing_fields": list(missing)}]}


def test_question_plan_cannot_omit_a_subquestion_or_negation():
    question = "Model-Z速度和硬件是什么？不要混合协议。"
    result = {"requirements": [{"id": "R1", "request": "Model-Z速度和硬件是什么？", "subject": "Model-Z",
                                "fields": ["value", "hardware"], "expected": {}}]}
    with pytest.raises(ProviderError, match="遗漏"):
        validate_plan(result, question, "qa")
    result["requirements"][0]["request"] = question
    assert validate_plan(result, question, "qa")[0]["mode"] == "qa"


@pytest.mark.parametrize("change", [
    {"subject": "Model-Y"}, {"expected": {"value": "13"}}, {"fields": ["invented"]},
    {"operator": "approximately"}, {"id": "R13"},
])
def test_claim_plan_does_not_change_the_claim(change):
    question = "Model-Z error 12"
    proposed = {"id": "R1", "request": question, "subject": "Model-Z", "fields": ["value"],
                "expected": {"value": "12"}}
    proposed.update(change)
    with pytest.raises(ProviderError):
        validate_plan({"requirements": [proposed]}, question, "verify")


def test_question_plan_cannot_preload_the_answer():
    with pytest.raises(ProviderError, match="预设答案"):
        validate_plan({"requirements": [{"id": "R1", "request": "Model-Z 12?", "subject": "Model-Z",
                                         "fields": ["value"], "expected": {"value": "12"}}]}, "Model-Z 12?", "qa")


def test_plan_supports_bounded_scientific_attributes_beyond_fixed_metric_fields():
    question = "Model-Z有多少层和attention heads？"
    plan = {"requirements": [{"id": "R1", "request": question, "subject": "Model-Z",
                               "fields": ["num_layers", "attention_heads", "hidden_dim"], "expected": {}}]}
    checked = validate_plan(plan, question, "qa")
    assert checked[0]["fields"] == ["num_layers", "attention_heads", "hidden_dim"]
    plan["requirements"][0]["fields"] = [f"attribute_{index}" for index in range(24)]
    assert len(validate_plan(plan, question, "qa")[0]["fields"]) == 24
    plan["requirements"][0]["fields"].append("one_too_many")
    with pytest.raises(ProviderError):
        validate_plan(plan, question, "qa")


@pytest.mark.parametrize("name", ["Heads", "layer-count", "__proto__", "", "a" * 49])
def test_custom_scientific_attribute_names_are_still_bounded_snake_case(name):
    with pytest.raises(ProviderError):
        validate_plan({"requirements": [{"id": "R1", "request": "配置？", "subject": None,
                                         "fields": [name], "expected": {}}]}, "配置？", "qa")


@pytest.mark.parametrize("operator", [None, "null", "None", "", "unknown"])
def test_only_qa_can_normalize_unknown_unused_operator(operator):
    plan = {"requirements": [{"id": "R1", "request": "Model-Z结果？", "subject": "Model-Z",
                               "fields": ["value"], "expected": {}, "operator": operator}]}
    assert validate_plan(plan, "Model-Z结果？", "qa")[0]["operator"] == "eq"
    with pytest.raises(ProviderError):
        validate_plan(plan, "Model-Z结果？", "verify")


def test_partial_answer_explicitly_accounts_for_missing_hardware():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement("qa")
    req["fields"].append("hardware")
    result = check_findings(findings([observation()], ["hardware"]), [req], evidence, fragments)
    assert result["rows"][0]["status"] == "partial"
    assert result["rows"][0]["missing_fields"] == ["hardware"]
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["source_complete"] is False
    assert "硬件：本次所检原文未能确认" in result["answer"]
    assert "论文未报告" not in result["answer"]
    quote = result["rows"][0]["observations"][0]["fields"]["value"]["evidence"][0]
    assert quote["quote"] == evidence[0]["text"] and quote["id"] == "e1"


def test_silent_task_or_field_omission_is_not_complete():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement("qa")
    req["fields"].append("hardware")
    missing_field = check_findings(findings([observation()]), [req], evidence, fragments)
    assert missing_field["coverage"]["complete"] is False
    missing_row = check_findings({"items": []}, [req], evidence, fragments)
    assert missing_row["coverage"] == {"planned": 1, "addressed": 0, "answered": 0, "missing": 5, "unaddressed": 1,
                                       "complete": False, "source_complete": False}


@pytest.mark.parametrize("rid", ["R2", "unknown"])
def test_unknown_task_id_cannot_disappear(rid):
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    with pytest.raises(ProviderError, match="未知或重复"):
        check_findings(findings([], rid=rid), [requirement()], evidence, fragments)


def test_duplicate_task_id_is_not_last_write_wins():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    result = findings([observation()])
    result["items"].append(deepcopy(result["items"][0]))
    with pytest.raises(ProviderError, match="未知或重复"):
        check_findings(result, [requirement()], evidence, fragments)


def test_model_explanation_and_label_cannot_override_numeric_comparison():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    raw = findings([observation()])
    raw.update(verdict="supported", explanation="The model wants support", answer="Wrong answer")
    result = check_findings(raw, [requirement(value="19")], evidence, fragments)
    assert result["verdict"] == "contradicted"
    assert result["rows"][0]["field_comparisons"]["value"]["status"] == "conflict"
    assert "Wrong answer" not in result["answer"] and "The model wants support" not in result["explanation"]


def test_different_protocol_is_insufficient_even_when_value_conflicts():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    result = check_findings(findings([observation()]), [requirement(value="19", protocol="10-crop")], evidence, fragments)
    assert result["verdict"] == "insufficient"
    assert result["rows"][0]["field_comparisons"]["protocol"]["status"] == "conflict"


def test_missing_protocol_does_not_become_a_controlled_refutation():
    text = "Model-Z error is 12."
    evidence, fragments = sources(text)
    raw = observation(protocol=None, quote=text)
    result = check_findings(findings([raw], ["protocol"]), [requirement(value="19")], evidence, fragments)
    assert result["verdict"] == "insufficient"


def test_omitted_expected_metric_does_not_allow_numeric_only_support():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement()
    del req["expected"]["metric"]
    result = check_findings(findings([observation()]), [req], evidence, fragments)
    assert result["verdict"] == "insufficient"


@pytest.mark.parametrize("operator,target,verdict", [("lt", "13", "supported"), ("lt", "12", "contradicted"),
                                                   ("le", "12", "supported"), ("gt", "13", "contradicted")])
def test_decimal_inequality_is_computed_after_identity_alignment(operator, target, verdict):
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement(value=target)
    req["operator"] = operator
    assert check_findings(findings([observation()]), [req], evidence, fragments)["verdict"] == verdict


def test_multimodel_sentence_is_withheld_instead_of_borrowing_a_number():
    text = "Model-Z error is 12 and Model-Q error is 19 under single-crop."
    evidence, fragments = sources(text)
    raw = observation(value="19", quote=text)
    result = check_findings(findings([raw]), [requirement(value="19")], evidence, fragments)
    assert result["verdict"] == "insufficient"
    assert result["rows"][0]["observations"][0]["status"] == "withheld"
    assert "不同结果句" in result["rows"][0]["observations"][0]["reasons"][0]


def test_model_cannot_mask_a_conflicting_measurement_by_putting_it_in_other_metadata():
    text = "Model-Z error is 12, Model-Q error is 19 under single-crop."
    evidence, fragments = sources(text)
    raw = observation(value="19", quote=text)
    raw["fields"]["other"] = field("12")
    result = check_findings(findings([raw]), [requirement(value="19")], evidence, fragments)
    assert result["verdict"] == "insufficient" and result["observations"][0]["status"] == "withheld"


def test_different_result_sentences_are_not_a_relation():
    text = "Model-Z error is 12. Model-Q error is 19 under single-crop."
    evidence, fragments = sources(text)
    result = check_findings(findings([observation(value="19", quote=text)]), [requirement(value="19")], evidence, fragments)
    assert result["observations"][0]["status"] == "withheld"


def test_metric_cannot_be_borrowed_from_another_fragment():
    evidence, fragments = sources("Model-Z accuracy is 12 under single-crop.", "The reported metric elsewhere is error.")
    raw = observation(quote=evidence[0]["text"])
    raw["fields"]["metric"] = field("error", "F2")
    result = check_findings(findings([raw]), [requirement()], evidence, fragments)
    assert result["verdict"] == "insufficient" and result["observations"][0]["status"] == "withheld"


def test_model_variant_cannot_fill_another_task_even_with_valid_own_relation():
    text = "Model-Q error is 12 under single-crop."
    evidence, fragments = sources(text)
    result = check_findings(findings([observation(subject="Model-Q", quote=text)]), [requirement()], evidence, fragments)
    assert result["observations"][0]["status"] == "withheld"
    assert result["verdict"] == "insufficient"


def test_model_and_explicit_configuration_numbers_are_not_extra_measurements():
    text = "Model-42 error is 12 under batch size 8."
    evidence, fragments = sources(text)
    raw = observation(subject="Model-42", protocol="batch size 8", quote=text)
    req = requirement()
    req["subject"] = "Model-42"
    req["expected"] = {"subject": "Model-42", "metric": "error", "value": "12", "protocol": "batch size 8"}
    assert check_findings(findings([raw]), [req], evidence, fragments)["verdict"] == "supported"


def test_model_version_number_cannot_become_a_measurement_or_repeat_count():
    text = "Model-12 error has not been measured."
    evidence, fragments = sources(text)
    raw = observation(subject="Model-12", protocol=None, quote=text)
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement()], evidence, fragments)
    raw = {"fields": {"subject": field("Model-12"), "repeats": field("12")}}
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement()], evidence, fragments)


def test_table_cell_binds_exact_model_row_and_original_value_source():
    text = "Model-Z error 12 single-crop"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "error",
                    "metric": "error", "value_cell_text": "12", "row_quote": text}}
    raw = observation(quote=text)
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    assert check_findings(findings([raw]), [requirement()], evidence, fragments, cells)["verdict"] == "supported"
    cells["T1"]["row_label"] = "Model-Q"
    swapped = check_findings(findings([raw]), [requirement()], evidence, fragments, cells)
    assert swapped["verdict"] == "insufficient"
    assert swapped["observations"][0]["status"] == "withheld"


def test_unresolved_paired_table_values_do_not_choose_a_protocol():
    text = "Model-Z error 12 (10) single-crop (10-crop)"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "error",
                    "value_cell_text": "12 (10)", "row_quote": text}}
    raw = observation(quote=text)
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    result = check_findings(findings([raw]), [requirement()], evidence, fragments, cells)
    assert result["verdict"] == "insufficient"


def test_mean_plus_minus_cell_never_uses_uncertainty_as_the_result():
    text = "Model-Z error 12 ± 0.3 single-crop"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "error",
                    "value_cell_text": "12 ± 0.3", "row_quote": text}}
    raw = observation(quote=text)
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    center = check_findings(findings([raw]), [requirement()], evidence, fragments, cells)
    assert center["verdict"] == "supported"
    assert "uncertainty" not in center["observations"][0]["fields"]
    raw["fields"]["value"] = field("0.3")
    deviation = check_findings(findings([raw]), [requirement(value="0.3")], evidence, fragments, cells)
    assert deviation["verdict"] == "insufficient" and deviation["observations"][0]["status"] == "withheld"


def test_table_dataset_column_cannot_be_swapped_for_another_dataset():
    text = "Model-Z Corpus-X error 12 single-crop Corpus-Y error 19"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "Corpus-X",
                    "column_labels": ["Corpus-X", "Corpus-Y"], "value_cell_text": "12", "row_quote": text}}
    raw = observation(quote=text)
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    raw["fields"]["dataset"] = field("Corpus-Y")
    result = check_findings(findings([raw]), [requirement()], evidence, fragments, cells)
    assert result["verdict"] == "insufficient"
    assert "数据集" in result["observations"][0]["reasons"][0]


def test_table_metric_cannot_be_borrowed_from_another_column():
    text = "Model-Z top-1 error 12 top-5 accuracy 88 single-crop"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "top-1 error",
                    "value_cell_text": "12", "row_quote": text}}
    raw = observation(quote=text)
    raw["fields"]["metric"] = field("top-5 accuracy")
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    result = check_findings(findings([raw]), [requirement()], evidence, fragments, cells)
    assert result["verdict"] == "insufficient" and result["observations"][0]["status"] == "withheld"


def test_bare_protocol_label_does_not_select_a_value_from_a_paired_cell():
    text = "Model-Z error 12 (10) single-crop (10-crop)"
    evidence, fragments = sources(text)
    cells = {"T1": {"source_id": "e1", "row_label": "Model-Z", "column_label": "error",
                    "value_cell_text": "12 (10)", "protocol": "single-crop", "row_quote": text}}
    raw = observation(quote=text)
    raw["binding"] = {"kind": "table_cell", "cell_id": "T1"}
    assert check_findings(findings([raw]), [requirement()], evidence, fragments, cells)["verdict"] == "insufficient"
    cells["T1"]["value"] = "12"
    assert check_findings(findings([raw]), [requirement()], evidence, fragments, cells)["verdict"] == "supported"
    raw["fields"]["value"] = field("10")
    assert check_findings(findings([raw]), [requirement(value="10")], evidence, fragments, cells)["verdict"] == "insufficient"


def test_observation_cannot_join_metadata_from_another_paper():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.", "The hardware is Device-X.")
    evidence[1]["paper_id"] = "another-paper"
    raw = observation()
    raw["fields"]["hardware"] = field("Device-X", "F2")
    result = check_findings(findings([raw]), [requirement()], evidence, fragments)
    assert result["verdict"] == "insufficient"
    assert "不同论文" in result["observations"][0]["reasons"][0]


@pytest.mark.parametrize("mutator", [
    lambda raw: raw["fields"]["value"].update(text="99"),
    lambda raw: raw["fields"]["value"].update(evidence=[{"handle": "F404"}]),
    lambda raw: raw["fields"]["value"].update(evidence=[{"handle": "F1", "quote": "fabricated"}]),
    lambda raw: raw["binding"].update(quote="Model-Z error is 99 under single-crop."),
])
def test_invalid_literal_or_fabricated_binding_is_protocol_error(mutator):
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    raw = observation()
    mutator(raw)
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement()], evidence, fragments)


@pytest.mark.parametrize("unknown", [None, {"text": None, "evidence": []},
                                      {"text": "未找到", "evidence": []}, {"text": "not reported", "evidence": []}])
def test_explicit_empty_unknown_is_audited_as_missing_without_losing_other_facts(unknown):
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement("qa")
    req["fields"].append("hardware")
    raw = observation()
    raw["fields"]["hardware"] = unknown
    result = check_findings(findings([raw]), [req], evidence, fragments)
    assert result["rows"][0]["missing_fields"] == ["hardware"]
    assert result["coverage"]["complete"] is True and result["coverage"]["source_complete"] is False
    assert result["rows"][0]["status"] == "partial"
    assert "hardware" not in result["observations"][0]["fields"]
    assert result["normalization_audit"] == [{"field": "hardware", "original": unknown,
                                              "action": "explicit_unknown_as_missing"}]


@pytest.mark.parametrize("invalid", [
    {"text": "Device-X", "evidence": []},
    {"text": "未找到", "evidence": [{"handle": "F404"}]},
    {"text": "未找到", "evidence": [{"handle": "F1"}]},
    {"text": None, "evidence": [], "value": "Device-X"},
])
def test_unknown_normalization_does_not_swallow_facts_or_invalid_references(invalid):
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    raw = observation()
    raw["fields"]["hardware"] = invalid
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement("qa")], evidence, fragments)


def test_all_explicit_unknown_observation_does_not_become_a_fact_or_source_complete():
    evidence, fragments = sources("A paper with no hardware details.")
    req = {"id": "R1", "request": "硬件？", "subject": None, "fields": ["hardware"], "mode": "qa", "expected": {}}
    result = check_findings(findings([{"fields": {"hardware": None}}]), [req], evidence, fragments)
    assert result["verdict"] == "insufficient" and result["rows"][0]["status"] == "insufficient"
    assert result["coverage"]["complete"] is True and result["coverage"]["source_complete"] is False
    assert result["citations"] == []


def test_compound_claim_server_aggregates_support_and_explicit_unknown():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    second = requirement(rid="R2")
    second.update(subject=None, fields=["repeats"], expected={"repeats": "3"})
    raw = findings([observation()])
    raw["items"].append({"requirement_id": "R2", "observations": [], "missing_fields": ["repeats"]})
    result = check_findings(raw, [requirement(), second], evidence, fragments)
    assert result["verdict"] == "partial"
    assert result["coverage"]["complete"] is True and result["coverage"]["source_complete"] is False


def test_bilingual_metric_alias_never_converts_error_to_accuracy():
    text = "Model-Z 准确率 is 12 under single-crop."
    evidence, fragments = sources(text)
    raw = observation(quote=text)
    raw["fields"]["metric"] = field("accuracy")
    result = check_findings(findings([raw]), [requirement()], evidence, fragments)
    assert result["verdict"] == "insufficient"
    assert result["rows"][0]["field_comparisons"]["metric"]["status"] == "conflict"


def test_source_quote_is_never_rewritten_by_field_aliases():
    text = "Model-Z 准确率 is 12 under single-crop."
    evidence, fragments = sources(text)
    raw = observation(quote=text)
    raw["fields"]["metric"] = field("accuracy")
    req = requirement()
    req["expected"]["metric"] = "准确率"
    result = check_findings(findings([raw]), [req], evidence, fragments)
    assert result["verdict"] == "supported"
    assert result["citation_quotes"][0]["quote"] == text


def test_sparse_experiments_cannot_complete_each_other_and_are_rendered_separately():
    text = "Model-Z error is 12. The independent protocol is single-crop."
    evidence, fragments = sources(text)
    first = observation(protocol=None, quote="Model-Z error is 12.")
    second = {"fields": {"subject": field("Model-Z"), "protocol": field("single-crop")}}
    result = check_findings(findings([first, second]), [requirement("qa")], evidence, fragments)
    assert result["rows"][0]["status"] == "partial"
    assert result["rows"][0]["missing_fields"] == ["protocol"]
    assert result["coverage"]["source_complete"] is False
    assert "观察1：" in result["answer"] and "观察2：" in result["answer"]
    assert result["answer"].index("数值：12") < result["answer"].index("观察2：")
    assert result["answer"].index("评价协议：single-crop") > result["answer"].index("观察2：")


def test_metadata_fields_from_distinct_models_do_not_count_as_one_complete_record():
    text = "Model-Z uses Device-X. Model-Q batch size is 8."
    evidence, fragments = sources(text)
    req = {"id": "R1", "request": "hardware and batch?", "subject": None,
           "fields": ["hardware", "batch_size"], "expected": {}, "mode": "qa"}
    first = {"fields": {"subject": field("Model-Z"), "hardware": field("Device-X")}}
    second = {"fields": {"subject": field("Model-Q"), "batch_size": field("8")}}
    result = check_findings(findings([first, second]), [req], evidence, fragments)
    assert result["coverage"]["source_complete"] is False


def test_metadata_counts_keep_the_original_train_validation_test_sentence():
    sentence = "The split has 140 training nodes, 500 validation nodes, and 1000 test nodes."
    evidence, fragments = sources(sentence)
    req = {"id": "R1", "request": "划分计数？", "subject": None, "fields": ["sample_size"],
           "expected": {}, "mode": "qa"}
    result = check_findings(findings([{"fields": {"sample_size": field(sentence)}}]), [req], evidence, fragments)
    assert sentence in result["answer"]
    assert result["rows"][0]["observations"][0]["fields"]["sample_size"]["text"] == sentence


def test_numeric_metadata_uses_decimal_without_a_semantic_model():
    text = "Model-Z repeated the run 3.0 times."
    evidence, fragments = sources(text)
    req = {"id": "R1", "request": "Model-Z重复3次", "subject": "Model-Z", "fields": ["subject", "repeats"],
           "expected": {"subject": "Model-Z", "repeats": "3"}, "mode": "verify"}
    raw = {"fields": {"subject": field("Model-Z"), "repeats": field("3.0")}}
    result = check_findings(findings([raw]), [req], evidence, fragments)
    assert result["verdict"] == "supported"
    assert result["rows"][0]["field_comparisons"]["repeats"]["basis"] == "literal"


def test_custom_condition_is_not_ignored_when_value_would_otherwise_refute():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.", "Model-Z has 4 layers.")
    req = requirement(value="19")
    req["fields"].append("num_layers")
    req["expected"]["num_layers"] = "3"
    raw = observation()
    raw["fields"]["num_layers"] = field("4", "F2")
    checked = check_findings(findings([raw]), [req], evidence, fragments)
    assert checked["verdict"] == "insufficient"
    assert checked["rows"][0]["field_comparisons"]["num_layers"]["status"] == "conflict"
    assert "num_layers：4" in checked["answer"]
    req["expected"]["num_layers"] = "4.0"
    assert check_findings(findings([raw]), [req], evidence, fragments)["verdict"] == "contradicted"


def test_custom_numeric_attribute_cannot_borrow_a_model_version_number():
    evidence, fragments = sources("Model-4 error is 12 under single-crop.")
    raw = observation(subject="Model-4", quote=evidence[0]["text"])
    raw["fields"]["num_layers"] = field("4")
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement()], evidence, fragments)


def test_unplanned_unknown_is_audited_without_faking_the_missing_required_field():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    req = requirement("qa")
    req["fields"].append("hardware")
    raw = findings([observation()], missing=["num_layers", "num_layers"])
    checked = check_findings(raw, [req], evidence, fragments)
    assert checked["rows"][0]["unplanned_missing_fields"] == ["num_layers"]
    assert checked["rows"][0]["explicit_missing_fields"] == []
    assert checked["coverage"]["complete"] is False and checked["coverage"]["source_complete"] is False
    assert checked["rows"][0]["missing_fields"] == ["hardware"]
    assert checked["normalization_audit"][0]["action"] == "unplanned_unknown_ignored_for_coverage"


def test_extra_unknown_does_not_discard_a_complete_source_record():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    checked = check_findings(findings([observation()], missing=["attention_heads"]), [requirement("qa")], evidence, fragments)
    assert checked["coverage"]["source_complete"] is True
    assert checked["rows"][0]["unplanned_missing_fields"] == ["attention_heads"]


def test_unplanned_missing_name_still_cannot_be_malformed():
    evidence, fragments = sources("Model-Z error is 12 under single-crop.")
    with pytest.raises(ProviderError):
        check_findings(findings([observation()], missing=["number-of-layers"]), [requirement("qa")], evidence, fragments)


def test_narrow_language_mapping_changes_condition_match_but_not_observed_measurement():
    text = "Model-Z error is 12 under independent runs."
    evidence, fragments = sources(text)
    req = requirement(protocol="独立重复运行", value="19")
    raw = findings([observation(protocol="independent runs", quote=text)])
    initial = check_findings(raw, [req], evidence, fragments)
    assert initial["verdict"] == "insufficient"
    pairs = semantic_pairs(initial)
    assert len(pairs) == 1 and pairs[0]["id"] == "R1:protocol"
    assert pairs[0]["expected"] == "独立重复运行" and pairs[0]["observed"] == "independent runs"
    assert pairs[0]["citation_quotes"][0]["quote"] == text
    checked = check_findings(raw, [req], evidence, fragments,
                             alignments={"alignments": [{"id": "R1:protocol", "relation": "match",
                                                          "reason": "表示同一独立运行条件。"}]})
    assert checked["verdict"] == "contradicted"
    comparison = checked["rows"][0]["field_comparisons"]["protocol"]
    assert comparison["basis"] == "semantic_model" and comparison["literal_status"] == "conflict"
    assert checked["rows"][0]["field_comparisons"]["value"]["basis"] == "literal"
    assert checked["rows"][0]["observations"][0]["fields"]["value"]["text"] == "12"
    assert "模型，需复核" in checked["explanation"]


@pytest.mark.parametrize("relation", ["conflict", "unknown"])
def test_semantic_condition_conflict_or_unknown_never_refutes_another_experiment(relation):
    text = "Model-Z error is 12 under independent runs."
    evidence, fragments = sources(text)
    req = requirement(protocol="独立重复运行", value="19")
    raw = findings([observation(protocol="independent runs", quote=text)])
    result = check_findings(raw, [req], evidence, fragments,
                            alignments={"alignments": [{"id": "R1:protocol", "relation": relation, "reason": "尚未建立相同条件。"}]})
    assert result["verdict"] == "insufficient"


def test_omitted_semantic_pair_is_unknown():
    text = "Model-Z error is 12 under independent runs."
    evidence, fragments = sources(text)
    result = check_findings(findings([observation(protocol="independent runs", quote=text)]),
                            [requirement(protocol="独立重复运行")], evidence, fragments,
                            alignments={"alignments": []})
    assert result["rows"][0]["field_comparisons"]["protocol"]["status"] == "unknown"
    assert result["verdict"] == "insufficient"


@pytest.mark.parametrize("field_name,expected,observed", [
    ("dataset", "Corpus-Y", "Corpus-X"), ("hardware", "Device-Y", "Device-X"),
    ("unit", "fps", "%"), ("repeats", "3", "4"),
])
def test_identity_units_hardware_and_numeric_metadata_are_never_semantic_tasks(field_name, expected, observed):
    text = f"Model-Z error is 12 under single-crop. {observed}"
    evidence, fragments = sources(text)
    raw = observation()
    raw["fields"][field_name] = field(observed)
    req = requirement()
    req["fields"].append(field_name)
    req["expected"][field_name] = expected
    result = check_findings(findings([raw]), [req], evidence, fragments)
    assert semantic_pairs(result) == []


@pytest.mark.parametrize("expected,observed", [
    ("batch size 8", "batch size 16"),
    ("DemoSet-C evaluation", "DemoSet evaluation"),
    ("Version-X evaluation", "Version-Y evaluation"),
])
def test_semantic_mapping_cannot_repair_different_numbers_or_strong_identifiers(expected, observed):
    text = f"Model-Z error is 12 under {observed}."
    evidence, fragments = sources(text)
    result = check_findings(findings([observation(protocol=observed, quote=text)]),
                            [requirement(protocol=expected)], evidence, fragments)
    assert semantic_pairs(result) == []
    with pytest.raises(ProviderError):
        check_findings(findings([observation(protocol=observed, quote=text)]), [requirement(protocol=expected)],
                       evidence, fragments, alignments={"alignments": [{"id": "R1:protocol", "relation": "match", "reason": "强行相同"}]})


def test_semantic_mapping_cannot_resurrect_accuracy_error_confusion():
    text = "Model-Z accuracy is 12 under single-crop."
    evidence, fragments = sources(text)
    raw = observation(quote=text)
    raw["fields"]["metric"] = field("accuracy")
    checked = check_findings(findings([raw]), [requirement()], evidence, fragments)
    assert checked["verdict"] == "insufficient" and semantic_pairs(checked) == []
    with pytest.raises(ProviderError):
        check_findings(findings([raw]), [requirement()], evidence, fragments,
                       alignments={"alignments": [{"id": "R1:metric", "relation": "match", "reason": "相同数值不是同指标"}]})


def test_semantic_mapping_cannot_equate_known_test_and_validation_splits():
    text = "Model-Z error is 12 under single-crop on test."
    evidence, fragments = sources(text)
    raw = observation(quote=text)
    raw["fields"]["split"] = field("test")
    req = requirement()
    req["fields"].append("split")
    req["expected"]["split"] = "验证集"
    checked = check_findings(findings([raw]), [req], evidence, fragments)
    assert checked["verdict"] == "insufficient" and semantic_pairs(checked) == []


@pytest.mark.parametrize("alignments", [
    [{"id": "R1:value", "relation": "match", "reason": "不能更改数字"}],
    [{"id": "R1:protocol", "relation": "match", "reason": "", "observed": "rewrite"}],
    [{"id": "R1:protocol", "relation": "supported", "reason": "不能输出标签"}],
    [{"id": "R1:protocol", "relation": "match", "reason": "same"},
     {"id": "R1:protocol", "relation": "match", "reason": "duplicate"}],
])
def test_semantic_model_cannot_change_pairs_values_or_final_labels(alignments):
    text = "Model-Z error is 12 under independent runs."
    evidence, fragments = sources(text)
    with pytest.raises(ProviderError):
        check_findings(findings([observation(protocol="independent runs", quote=text)]),
                       [requirement(protocol="独立重复运行")], evidence, fragments,
                       alignments={"alignments": alignments})
