"""Explicit LangGraph workflow; offline nodes are rules, not simulated LLMs."""
from __future__ import annotations

import operator
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, TypedDict

from .grounding import number_in_text, quote_matches_source


class ComparisonState(TypedDict):
    experiments: list[dict]
    evidence: list[dict]
    comparisons: list[dict]
    traces: Annotated[list[dict], operator.add]


def _trace(agent: str, summary: str, started: float, ids: list[str] | None = None,
           inputs: dict | None = None, outputs: dict | None = None) -> dict:
    elapsed = time.perf_counter() - started
    finished = datetime.now(timezone.utc)
    return {"agent": agent, "status": "completed", "summary": summary,
            "evidence_ids": ids or [], "duration_ms": round(elapsed * 1000), "kind": "rule",
            "stage": agent, "input": inputs or {}, "output": outputs or {},
            "started_at": (finished - timedelta(seconds=elapsed)).isoformat(), "finished_at": finished.isoformat()}


def run_comparison(experiments: list[dict], evidence: list[dict]) -> dict:
    from langgraph.graph import END, START, StateGraph
    from .analysis import compare_experiments
    from .experiment_validation import machine_measurement_verified
    from .providers import _field_is_literal, _measurement_anchor_conflicts, _metadata_measurement_conflicts, _unit_matches_measurement

    def prepare(state: ComparisonState):
        started = time.perf_counter()
        known = {e["id"]: e for e in state["evidence"]}
        complete = sum(bool(e.get("evidence_ids")) and all(
            eid in known and known[eid]["paper_id"] == e["paper_id"] for eid in e["evidence_ids"])
            for e in state["experiments"])
        return {"traces": [_trace("实验整理", f"读取 {len(state['experiments'])} 项实验，{complete} 项具有有效原文引用；使用已保存的提取结果。", started,
                                  inputs={"experiments": len(state["experiments"]), "evidence": len(known)}, outputs={"source_linked_experiments": complete})]}

    def align(state: ComparisonState):
        started = time.perf_counter()
        comparisons = compare_experiments(state["experiments"], state["evidence"])
        direct = sum(c["comparability"] == "direct" for c in comparisons)
        return {"comparisons": comparisons, "traces": [_trace("条件对齐", f"按任务、数据集、划分、指标、单位、方法、样本量和条件核对 {len(comparisons)} 组实验，其中 {direct} 组可直接比较。", started,
                                                             inputs={"experiments": len(state["experiments"])}, outputs={"comparisons": len(comparisons), "direct": direct})]}

    def analyze(state: ComparisonState):
        started = time.perf_counter()
        flagged = sum(c["verdict"] == "potential_conflict" for c in state["comparisons"])
        return {"traces": [_trace("差异分析", f"发现 {flagged} 组满足规则的潜在冲突；数值差异不等于科学结论冲突，未进行显著性推断。", started,
                                  inputs={"comparisons": len(state["comparisons"])}, outputs={"potential_conflicts": flagged,
                                  "numerical_differences": sum(c["verdict"] == "different" for c in state["comparisons"])})]}

    def verify(state: ComparisonState):
        started = time.perf_counter()
        known = {e["id"]: e for e in state["evidence"]}
        experiments_by_id = {e["id"]: e for e in state["experiments"]}
        # Every experiment is immutable within this request. Its source checks
        # do not depend on the other member of a pair: run them once, not N times.
        # The cache is request-local, and retains all failures and source checks.
        def check_experiment(exp):
            owners_valid = True
            unsupported = set()
            if "machine_verification" in exp and not machine_measurement_verified(exp, state["evidence"]):
                owners_valid = False
                unsupported.add("machine_measurement")
            own = [ref for ref in exp.get("evidence_ids", []) if ref in known and known[ref]["paper_id"] == exp["paper_id"]]
            owners_valid = owners_valid and bool(own) and len(own) == len(exp.get("evidence_ids", []))
            value_refs = exp.get("field_evidence", {}).get("value", [])
            owners_valid = owners_valid and bool(value_refs) and all(ref in own for ref in value_refs)
            value_quotes = exp.get("evidence_quotes", {}).get("value", [])
            owners_valid = owners_valid and bool(value_quotes) and all(
                isinstance(ref, dict) and ref.get("id") in value_refs and ref["id"] in own
                and quote_matches_source(ref.get("quote"), known[ref["id"]]["text"])
                and number_in_text(exp.get("value"), ref["quote"]) for ref in value_quotes)
            if not _unit_matches_measurement(exp, state["evidence"]):
                owners_valid = False
                unsupported.add("unit")
            if _measurement_anchor_conflicts(exp, state["evidence"]):
                owners_valid = False
                unsupported.add("result_association")
            for field in ("task", "method", "dataset", "split", "metric", "value", "unit", "sample_size", "conditions"):
                value = exp.get(field)
                if value is None or value == "" or value == {}:
                    continue  # The alignment stage already reports missing values.
                field_ids = exp.get("field_evidence", {}).get(field, [])
                quotations = exp.get("evidence_quotes", {}).get(field, [])
                valid = (isinstance(field_ids, list) and bool(field_ids)
                         and all(ref in own for ref in field_ids)
                         and isinstance(quotations, list) and bool(quotations)
                         and all(isinstance(ref, dict) and ref.get("id") in field_ids
                                 and quote_matches_source(ref.get("quote"), known[ref["id"]]["text"])
                                 for ref in quotations))
                if valid:
                    valid = (_field_is_literal(field, value, [known[ref] for ref in field_ids], quotations)
                             and not _metadata_measurement_conflicts(exp, field))
                if not valid:
                    owners_valid = False
                    unsupported.add(field)
            return owners_valid, set(own), unsupported

        verified = {eid: check_experiment(exp) for eid, exp in experiments_by_id.items()}
        checked = []
        all_ids = set()
        for original in state["comparisons"]:
            item = dict(original)
            requested = item.get("evidence_ids", [])
            owners_valid = True
            allowed = set()
            unsupported = set()
            for eid in (item["left_id"], item["right_id"]):
                valid_owner, own, gaps = verified[eid]
                owners_valid = owners_valid and valid_owner
                allowed.update(own)
                unsupported.update(gaps)
            valid = [x for x in requested if x in allowed]
            if not owners_valid or not requested or len(valid) != len(requested):
                item["verdict"] = "insufficient"
                item["comparability"] = "insufficient"
                item["delta"] = None
                item["conflict_evidence"] = []
                item["reasons"] = [*item.get("reasons", []), "原文引用或实验字段依据不完整，无法完成核验；人工修改字段后需重新关联相应原文。"
                                   + (" 缺少有效依据的字段：" + "、".join(sorted(unsupported)) + "。" if unsupported else "")]
            item["evidence_ids"] = valid
            all_ids.update(valid)
            checked.append(item)
        return {"comparisons": checked, "traces": [_trace("证据检查", f"检查 {len(all_ids)} 条引用的归属及可追溯性。当前为规则模式，语义支持程度需在结论核验页调用模型或人工复核。", started, sorted(all_ids),
                                                        inputs={"comparisons": len(state["comparisons"])}, outputs={"located_evidence": len(all_ids),
                                                        "insufficient": sum(c["verdict"] == "insufficient" for c in checked)})]}

    graph = StateGraph(ComparisonState)
    for name, fn in [("prepare", prepare), ("align", align), ("analyze", analyze), ("verify", verify)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "align")
    graph.add_edge("align", "analyze")
    graph.add_edge("analyze", "verify")
    graph.add_edge("verify", END)
    return graph.compile().invoke({"experiments": experiments, "evidence": evidence, "comparisons": [], "traces": []}, {"recursion_limit": 8})
