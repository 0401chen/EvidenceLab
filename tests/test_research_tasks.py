"""Current product protocol, separate from archived V1 response fixtures."""
import asyncio
import json

from backend.app import providers
from backend.app.research_tasks import table_cells


def _observation(subject, value):
    values = {"subject": subject, "metric": "accuracy", "value": value,
              "unit": "%", "dataset": "Bench-Q", "split": "test"}
    return {"fields": {name: {"text": text, "evidence": [{"handle": "F1"}]}
                       for name, text in values.items()},
            "binding": {"kind": "text", "handle": "F1",
                        "quote": "Atlas-A achieves accuracy 73% on Bench-Q test."}}


def test_default_qa_plans_without_papers_and_preserves_explicit_unknown(monkeypatch, evidence_factory):
    question = "请给出 Atlas-A 的准确率和硬件。"
    calls = []

    async def reply(messages, config):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        if len(calls) == 1:
            assert payload["source_data"] == []  # Planning cannot chase an easy answer.
            return {"requirements": [{"id": "R1", "request": question, "subject": "Atlas-A",
                                      "fields": ["subject", "metric", "value", "hardware"], "expected": {}}]}
        assert payload["requirements"][0]["fields"][-1] == "hardware"
        return {"answer": "invented 99% on GPU", "verdict": "supported",
                "items": [{"requirement_id": "R1", "observations": [_observation("Atlas-A", "73")],
                           "missing_fields": ["hardware"]}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.answer_llm(question, [evidence_factory(
        "Atlas-A achieves accuracy 73% on Bench-Q test.")], {"model": "offline"}))
    assert len(calls) == 2
    assert result["task_protocol"] == "requirements_observations_v1"
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["source_complete"] is False
    assert result["coverage"]["missing"] == 1
    assert "硬件：本次所检原文未能确认" in result["answer"]
    assert "99" not in result["answer"] and "GPU" not in result["answer"]


def test_default_verification_compares_observed_number_not_model_label(monkeypatch, evidence_factory):
    claim = "Atlas-A 在 Bench-Q 测试集的准确率为91%。"
    calls = []

    async def reply(messages, config):
        calls.append(json.loads(messages[-1]["content"]))
        if len(calls) == 1:
            expected = {"subject": "Atlas-A", "dataset": "Bench-Q", "split": "测试集",
                        "metric": "准确率", "value": "91", "unit": "%"}
            return {"requirements": [{"id": "R1", "request": claim, "subject": "Atlas-A",
                                      "fields": list(expected), "expected": expected}]}
        return {"verdict": "supported", "explanation": "错误的总标签不得采用。",
                "items": [{"requirement_id": "R1", "observations": [_observation("Atlas-A", "73")],
                           "missing_fields": []}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory(
        "Atlas-A achieves accuracy 73% on Bench-Q test.")], {"model": "offline"}))
    assert result["verdict"] == "contradicted"
    comparisons = {item["field"]: item for item in result["task_rows"][0]["field_comparisons"]}
    assert comparisons["value"]["relation"] == "conflict"
    assert comparisons["value"]["expected"] == "91"
    assert comparisons["value"]["observed"] == ["73"]
    assert "错误的总标签" not in result["explanation"]


def test_table_cells_preserve_benchmark_axes_and_mean_uncertainty(evidence_factory):
    source = evidence_factory("| Model | Bench-Q | Bench-R |\n| --- | --- | --- |\n| Atlas-A | 73 ± 0.4 | 81 |", "table")
    source["source_type"] = "table"
    _, fragments = providers._citation_fragments([source])
    cells = table_cells([source], fragments)
    assert [(cell["row_label"], cell["column_label"], cell["value_cell_text"]) for cell in cells.values()] == [
        ("Atlas-A", "Bench-Q", "73 ± 0.4"), ("Atlas-A", "Bench-R", "81")]
    assert all(cell["source_id"] == source["id"] for cell in cells.values())


def test_compact_record_inherits_only_its_selected_source(evidence_factory):
    from backend.app.research_tasks import bound_findings, task_fragments
    from backend.app.task_contract import check_findings, validate_plan
    question = "Atlas-A 的准确率？"
    evidence = [evidence_factory("Atlas-A achieves accuracy 73% on Bench-Q test.", "first"),
                evidence_factory("Another method reaches accuracy 99%.", "second")]
    _, fragments = task_fragments(evidence)
    binding = {"M1": {"kind": "text", "handle": "F1", "quote": fragments["F1"]["quote"], "source_handles": ["F1"]}}
    raw = {"items": [{"requirement_id": "R1", "records": [{"subject": ["Atlas-A"], "metric": ["accuracy"],
                    "value": ["73%"], "binding": "M1"}], "unknown": []}]}
    normalized = bound_findings(raw, binding)
    plan = validate_plan({"requirements": [{"id": "R1", "request": question, "subject": "Atlas-A",
                          "fields": ["subject", "metric", "value", "unit"], "expected": {}}]}, question, "qa")
    result = check_findings(normalized, plan, evidence, fragments)
    assert result["coverage"]["source_complete"] is True
    assert {quote["id"] for quote in result["citation_quotes"]} == {"first"}
    assert result["observations"][0]["fields"]["unit"]["text"] == "%"
    # An explicit bad handle stays invalid; selecting a valid relation never
    # authorizes guessing a replacement reference from matching source text.
    raw["items"][0]["records"][0]["value"] = ["73%", "F999"]
    import pytest
    from backend.app.errors import ProviderError
    with pytest.raises(ProviderError, match="引用不存在"):
        check_findings(bound_findings(raw, binding), plan, evidence, fragments)


def test_invalid_record_preserves_valid_observation_but_cannot_support_claim(monkeypatch, evidence_factory):
    """A single bad reference cannot disappear behind another valid record."""
    claim = "Atlas-A"
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        if len(calls) == 1:
            return {"requirements": [{"id": "R1", "request_handle": "Q1", "subject": "Atlas-A",
                                      "fields": ["subject"], "expected": {"subject": "Atlas-A"}}]}
        return {"items": [{"requirement_id": "R1", "records": [
            {"subject": ["Atlas-A", "F1"]}, {"subject": ["Atlas-A", "F999"]}], "unknown": []}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory("Atlas-A is evaluated.")], {"model": "offline"}))
    assert len(calls) == 2
    assert result["verdict"] == "insufficient"
    row = result["task_rows"][0]
    assert row["verdict"] == "insufficient" and row["status"] == "partial" and row["complete"] is False
    assert [observation["status"] for observation in row["observations"]] == ["accepted", "withheld"]
    assert result["coverage"]["answered"] == 0 and result["coverage"]["complete"] is False
    assert "· 证据不足" in result["answer"] and "· 支持" not in result["answer"]
    assert len(result["citation_quotes"]) == 1


def test_invalid_planned_operator_is_visible_gap_instead_of_crash(monkeypatch, evidence_factory):
    claim = "Atlas-A"
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        if len(calls) == 1:
            return {"requirements": [{"id": "R1", "request_handle": "Q1", "subject": "Atlas-A",
                                      "fields": ["subject"], "expected": {"subject": "Atlas-A"}, "operator": []}]}
        return {"items": [{"requirement_id": "R1", "records": [{"subject": ["Atlas-A", "F1"]}], "unknown": []}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory("Atlas-A is evaluated.")], {"model": "offline"}))
    assert result["verdict"] == result["task_rows"][0]["verdict"] == "insufficient"
    assert result["coverage"]["answered"] == 0
    assert "· 支持" not in result["answer"]


def test_unparsed_claim_is_not_certified_by_finding_its_subject(evidence_factory):
    from backend.app.research_tasks import task_fragments
    from backend.app.task_contract import check_findings, validate_plan
    claim = "Atlas-A"
    evidence = [evidence_factory("Atlas-A is evaluated.")]
    _, fragments = task_fragments(evidence)
    plan = validate_plan({"requirements": [{"id": "R1", "request": claim, "subject": "Atlas-A",
                          "fields": ["subject"], "expected": {}}]}, claim, "verify")
    result = check_findings({"items": [{"requirement_id": "R1", "observations": [
        {"fields": {"subject": {"text": "Atlas-A", "evidence": [{"handle": "F1"}]}}}]}]}, plan, evidence, fragments)
    assert result["verdict"] == "insufficient" and "· 支持" not in result["answer"]


def test_planner_cannot_drop_a_quantitative_constraint(monkeypatch, evidence_factory):
    claim = "Atlas-A 的准确率为 91%。"
    calls = []

    async def reply(messages, config):
        calls.append(messages)
        if len(calls) == 1:
            return {"requirements": [{"id": "R1", "request_handle": "Q1", "subject": "Atlas-A",
                                      "fields": ["subject"], "expected": {"subject": "Atlas-A"}}]}
        return {"items": [{"requirement_id": "R1", "records": [{"subject": ["Atlas-A", "F1"]}], "unknown": []}]}

    monkeypatch.setattr(providers, "_chat_json", reply)
    result = asyncio.run(providers.verify_claim(claim, [evidence_factory("Atlas-A is evaluated.")], {"model": "offline"}))
    assert result["verdict"] == "insufficient"
    assert result["coverage"]["complete"] is False
    assert "量化限定未进入核验字段" in result["task_rows"][0]["explanation"]
