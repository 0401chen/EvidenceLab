"""Request-local task plans and evidence observations, without research writes.

Planning sees the user's request, not a tentative answer. The evidence stage
returns measurements instead of a verdict or a prose answer; the contract
validator renders those and compares the requested and observed fields.
"""
from __future__ import annotations

import re
from time import perf_counter

from .analysis import select_evidence_context
from .grounding import NUMBER_PATTERN


PROTOCOL = "requirements_observations_v1"

PLAN_INSTRUCTION = """只拆解用户的科研任务，不回答，不猜论文内容；本阶段没有论文。
把用户明确要求的每个对象/模型变体与问题子项列成任务清单。相同模型同一测量的属性可以放一项，不同模型、指标、协议的测量必须分项。不能因为某个子项可能难答就省略。
返回 {"requirements":[{"id":"R1","request_handle":"Q1","subject":"原问题中的模型/方法连续原文或null","fields":["subject","metric","value","unit","dataset","split","protocol","hardware","batch_size","sample_size","repeats","uncertainty","task","training","other"],"expected":{},"operator":"eq"}]}。
fields只列该项实际要求的属性，不要把示例里的全部字段机械加入。fields可以用固定英文名。每项request_handle只能选择输入提供的Q1；服务器原样恢复完整原问题，不需手抄request、不能改写问题删去另一模型或条件。最多12项，id为R1起连续编号。
中文问答(mode=qa)：expected必须空对象。问题问几个模型的速度/硬件/批处理，就对每个模型分别要求这些字段；问缺失或未报告信息也不能删掉该字段。subject必须是问题里的模型标识，不能自造别名。
结论核验(mode=verify)：expected列出原命题实际声明的每个字段，所有值都从原命题连续复制，禁止改数字、翻译值或补默认条件。必须包含原命题明确的对象、数据集、指标、数值、单位、split和协议等；fields包含expected的全部键。比较符用eq/lt/le/gt/ge，无法无歧义表达的否定、因果、复合统计关系保留other原文，不能改成容易支持的命题。
例如“Model-A在Bench-Q测试集准确率为72%”的expected可为{"subject":"Model-A","dataset":"Bench-Q","split":"测试集","metric":"准确率","value":"72","unit":"%"}。这个例子不是实际任务数据。
数字属于模型名时不要当测量值。不要预判支持/反驳/不足，不输出答案、verdict、论文事实或引用。"""

FINDINGS_INSTRUCTION = """逐个完成已冻结的任务清单，只从提供原文提取原子观察，不输出自由回答、解释或verdict。不得改写任务、删除难答子项或为迎合expected拼接实验。
本请求使用简洁的来源选择协议，覆盖前述手抄id/quote格式要求。返回：
{"items":[{"requirement_id":"R1","records":[{"subject":["原方法名","F1"],"metric":["原指标名","F2"],"value":["原数值","F1"],"dataset":["原数据集名","F2"],"binding":"已有关系id"}],"unknown":["hardware"]}]}。
每个record是一个测量或设置。字段用固定英文名，含数值时始终还需subject/metric/value。每个字段是数组[原文连续字串,对应F编号,其他必要F编号...]，不要再嵌套fields/evidence对象，不要返回手抄quote。binding是relation_bindings里真实存在的M/T编号。字段若就在所选binding原句/单元格表头表注中，可以省略F编号，由服务器严格继承这一关系的来源；不在关系来源中的上下文字段必须明确指定F编号，不能猜ID或全局搜索补引用。
每个任务ID都要返回。每个所需字段必须给原文观察，或列入unknown；未找到只表示本次原文未找到，不等于论文从未报告。字段文字必须保留原文语言与写法，不能翻译成中文。
每个observation只能属于一个明确的模型/配置、数据集、划分和指标。要反驳错误数字，应如实返回同身份同条件下真正报告的数字；不能把命题数字抄作观察。另一模型、另一指标、不同条件都不能成为反驳，必须保留区别。
含数值的record必须有subject、metric、value，并选择relation_bindings里的一个真实id作为binding字符串。不能自己填写kind、quote、cell_id或自造T1。
1. 表格关系id以T开头：必须逐一核对table_cells的行名、列名与完整原格。subject是该行的方法名，value是该格的中心测量数值；不可交换同一行的不同数据集/指标列。±标准差放uncertainty，不能作第二个结果。字段引文分别指向原行、表头及实际赋予指标/单位/协议的表注或设置原句。
2. 叙述关系id以M开头：对应一个source_handle，服务器恢复该来源片段。片段必须直接包含所填subject、metric及value，且不能包含多个无法唯一对应的测量。跨句共指或归属无法确定就不要生成数值观察。
table_cells是机械行列定位，不是已审阅事实；列名可能是数据集而非指标，指标、单位和协议必须另外核对。不要根据一般常识把裸数字变成百分数，不将accuracy换error，不算补数，不臆测单模型/集成、随机划分、硬件或重复次数。
上下文字段必须属于这一测量：只共享文献、页码或相邻不证明归属，需原句明确的全实验设置、表号或方法关系；不能把对比方法/先前工作条件借给当前模型。无可靠归属则该字段缺失。
字段值未找到时，省略该字段并加入该任务的unknown，不要给没有来源的占位词。
用户原命题的比较与标签由服务器根据字段对照产生；不要输出或参考先前模型的结论。"""

_FIELD_QUERY = {
    "value": "results measurement", "metric": "accuracy error evaluation metric",
    "protocol": "evaluation protocol experimental setup", "hardware": "hardware GPU CPU",
    "batch_size": "batch size processing", "split": "training validation test split",
    "sample_size": "samples nodes examples labels", "repeats": "runs trials repeated",
    "uncertainty": "standard deviation variance confidence", "training": "training setup",
}

ALIGNMENT_INSTRUCTION = """只比较固定字段对的语义是否等价，不判断整句结论，不改写任何事实或数值。
每个pair含原命题字段expected、已提取的英文原文observed和对应引用。返回 {"alignments":[{"id":"输入pair的id","relation":"match|conflict|unknown","reason":"简短中文理由"}]}。
match要求两者表达同一指标/条件/任务，不能仅因都合理就认为相同。注意固定数据划分与随机重新划分、随机权重初始化与随机数据划分、均值与单次结果、标准差与标准误、验证集与测试集、不同模型/数据集均不等价。多个限定中有缺失应unknown，不能补默认条件。
不同表达但同义的中英文可match。原文明确不同可conflict，信息不全选unknown。不得把另一个设置移接过来。只返回id/relation/reason，不返回expected/observed、总体verdict、数字或新证据。这个阶段仍是模型语义映射，不是确定性科学证明。"""


def _bound_plan(result, question, mode):
    from .errors import ProviderError
    from .task_contract import validate_plan
    gaps = {}
    if isinstance(result, dict) and isinstance(result.get("requirements"), list):
        normalized = []
        for raw in result["requirements"]:
            if isinstance(raw, dict) and "request_handle" in raw:
                if raw["request_handle"] != "Q1" or "request" in raw:
                    raise ProviderError("output_format", "任务问题handle不存在或混入自行改写的request。")
                raw = {key: value for key, value in raw.items() if key != "request_handle"}
                raw["request"] = question
            if isinstance(raw, dict):
                raw = dict(raw)
                issues = []
                if isinstance(raw.get("subject"), str) and raw["subject"].strip() not in question:
                    raw["subject"] = None
                    issues.append("对象未按原问题绑定，不能判断该项已覆盖。")
                expected = raw.get("expected", {})
                if mode == "verify" and isinstance(expected, dict):
                    if any(not isinstance(value, str) or not value.strip() or value.strip() not in question for value in expected.values()):
                        raw["expected"] = {}
                        issues.append("核验字段改写了原命题，本项不允许生成支持或反驳。")
                    operator = raw.get("operator", "eq")
                    aliases = {"=": "eq", "==": "eq", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
                    if isinstance(operator, str) and operator in aliases:
                        raw["operator"] = aliases[operator]
                    elif not isinstance(operator, str) or operator not in {"eq", "lt", "le", "gt", "ge"}:
                        raw["operator"], raw["expected"] = "eq", {}
                        issues.append("无法绑定命题比较关系，本项仅保留来源观察。")
                if issues:
                    if isinstance(raw.get("id"), str):
                        gaps[raw["id"]] = issues
                    if mode == "verify":
                        raw["expected"] = {}
            normalized.append(raw)
        result = {**result, "requirements": normalized}
    checked = validate_plan(result, question, mode)
    if mode == "verify":
        # A verbatim Q1 is not evidence that the planner represented every
        # quantitative constraint. Missing numbers cannot silently disappear
        # from a claim that later receives a positive/negative label.
        from .task_contract import _numbers
        requested_numbers = set(_numbers(question))
        represented_numbers = {number for requirement in checked
                               for value in requirement["expected"].values() for number in _numbers(value)}
        if requested_numbers - represented_numbers:
            for requirement in checked:
                gaps.setdefault(requirement["id"], []).append("命题中有量化限定未进入核验字段，不能判断整条命题。")
                requirement["expected"] = {}
    for requirement in checked:
        if requirement["id"] in gaps:
            requirement["planning_gaps"] = gaps[requirement["id"]]
    return checked


def task_fragments(evidence):
    """Bind prose sentences separately without rewriting their source bytes."""
    from .providers import _citation_fragments
    payload, fragments = [], {}
    for source_index, source in enumerate(evidence, 1):
        text = source["text"]
        if source.get("source_type") == "table":
            _, local = _citation_fragments([source])
            spans = [(item["start"], item["end"]) for item in local.values()]
        else:
            edges = [0] + [match.end() for match in re.finditer(r"[。！？!?]\s*|(?<=[.!?])\s+(?=[A-Z])", text)] + [len(text)]
            spans = []
            for begin, end in zip(edges, edges[1:]):
                if end <= begin:
                    continue
                _, local = _citation_fragments([{**source, "text": text[begin:end]}], max_chars=360)
                spans.extend((begin + item["start"], begin + item["end"]) for item in local.values())
        for begin, end in spans:
            if not text[begin:end].strip():
                continue
            handle = f"F{len(fragments) + 1}"
            fragments[handle] = {"source": source, "quote": text[begin:end], "start": begin, "end": end}
            payload.append({"id": handle, "source_group": f"S{source_index}", "paper_id": source["paper_id"],
                            "page": source.get("page"), "source_type": source.get("source_type", "text"),
                            "section": source.get("section"), "text": text[begin:end]})
    return payload, fragments


def bound_findings(result, bindings):
    from .errors import ProviderError
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        return result
    items = []
    for raw in result["items"]:
        if isinstance(raw, dict) and "records" in raw:
            if not isinstance(raw["records"], list):
                raise ProviderError("output_format", "records必须为观察数组。")
            converted = []
            unknown = list(raw.get("unknown", [])) if isinstance(raw.get("unknown", []), list) else raw.get("unknown")
            for record in raw["records"]:
                if not isinstance(record, dict):
                    raise ProviderError("output_format", "record必须为字段对象。")
                if "unknown" in record:
                    if not isinstance(record["unknown"], list) or not isinstance(unknown, list):
                        raise ProviderError("output_format", "unknown必须是字段名数组。")
                    unknown.extend(record["unknown"])
                    if set(record) <= {"subject", "unknown"}:
                        continue  # Explicit unknown record makes no source fact.
                selected_binding = bindings.get(record.get("binding")) if isinstance(record.get("binding"), str) else None
                inherited = selected_binding.get("source_handles", []) if selected_binding else []
                fields = {}
                for field, selection in record.items():
                    if field in {"binding", "unknown"}:
                        continue
                    if not isinstance(selection, list) or not selection or not isinstance(selection[0], str) or any(not isinstance(handle, str) for handle in selection[1:]):
                        raise ProviderError("output_format", "观察字段必须为[原文文字,F编号...]，未知字段请放unknown。")
                    handles = selection[1:] or inherited
                    fields[field] = {"text": selection[0], "evidence": [{"handle": handle} for handle in handles]}
                # An explicit percent suffix is a source character, not an
                # inferred benchmark convention. Preserve it as its own field.
                if "unit" not in fields and re.fullmatch(r"\s*" + NUMBER_PATTERN + r"\s*[%％]\s*", fields.get("value", {}).get("text", "")):
                    fields["unit"] = {"text": "%", "evidence": fields["value"]["evidence"]}
                observation = {"fields": fields}
                if "binding" in record:
                    observation["binding"] = {"id": record["binding"]}
                converted.append(observation)
            raw = {"requirement_id": raw.get("requirement_id"), "observations": converted,
                   "missing_fields": list(dict.fromkeys(unknown)) if isinstance(unknown, list) else unknown}
        if not isinstance(raw, dict) or not isinstance(raw.get("observations"), list):
            items.append(raw)
            continue
        observations = []
        for observation in raw["observations"]:
            if isinstance(observation, dict) and isinstance(observation.get("binding"), dict) and "id" in observation["binding"]:
                selection = observation["binding"]
                if set(selection) != {"id"} or not isinstance(selection["id"], str) or selection["id"] not in bindings:
                    raise ProviderError("grounding", "测量关系id不存在或混入自填关系；只能从relation_bindings选择。")
                observation = {**observation, "binding": {key: value for key, value in bindings[selection["id"]].items() if key != "source_handles"}}
            observations.append(observation)
        items.append({**raw, "observations": observations})
    return {**result, "items": items}


def partial_findings(result, requirements, evidence, fragments, cells, bindings):
    """Keep invalid observations out of the answer without discarding valid ones.

    Strict validators still reject invalid references/facts. This boundary
    catches that rejection per record, visibly withholds the record, and makes
    the affected task ineligible for a positive/negative claim verdict.
    """
    from .errors import ProviderError
    from .task_contract import check_findings
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise ProviderError("output_format", "模型未返回逐任务items。")
    allowed = {item["id"] for item in requirements}
    seen, safe, issues = set(), [], []
    for item in result["items"]:
        if not isinstance(item, dict) or item.get("requirement_id") not in allowed or item["requirement_id"] in seen:
            raise ProviderError("output_format", "逐任务输出含未知或重复requirement_id。")
        rid = item["requirement_id"]
        seen.add(rid)
        key = "records" if "records" in item else "observations"
        records = item.get(key)
        if not isinstance(records, list) or len(records) > 8:
            raise ProviderError("output_format", "每个任务最多8条观察记录。")
        declared_unknown = item.get("unknown", []) if key == "records" else item.get("missing_fields", [])
        if not isinstance(declared_unknown, list):
            raise ProviderError("output_format", "未知字段必须是数组。")
        observations, missing = [], list(declared_unknown)
        for record in records:
            try:
                single = {"requirement_id": rid, key: [record],
                          "unknown" if key == "records" else "missing_fields": []}
                converted = bound_findings({"items": [single]}, bindings)["items"][0]
                check_findings({"items": [converted]}, requirements, evidence, fragments, cells=cells)
                observations.extend(converted["observations"])
                missing.extend(converted.get("missing_fields", []))
            except ProviderError as exc:
                issues.append({"requirement_id": rid, "code": exc.code, "message": exc.public_message})
        safe.append({"requirement_id": rid, "observations": observations,
                     "missing_fields": list(dict.fromkeys(missing))})
    effective = [{**requirement, "expected": {}} if any(issue["requirement_id"] == requirement["id"] for issue in issues)
                 else requirement for requirement in requirements]
    normalized = {"items": safe}
    checked = check_findings(normalized, effective, evidence, fragments, cells=cells)
    checked["protocol_issues"] = issues
    return normalized, checked, effective


def _annotate_incomplete(checked, requirements, issues):
    from .task_contract import _render_row
    mode = requirements[0]["mode"]
    for row in checked.get("rows", []):
        planning = next((item.get("planning_gaps", []) for item in requirements if item["id"] == row["id"]), [])
        rejected = [issue for issue in issues if issue["requirement_id"] == row["id"]]
        if rejected:
            row["observations"].extend({"fields": {}, "status": "withheld", "reasons": [issue["message"]]} for issue in rejected)
        if planning or rejected:
            row["status"] = "partial" if any(observation.get("status") == "accepted" for observation in row["observations"]) else "insufficient"
            row["verdict"], row["complete"] = "insufficient", False
            row["explanation"] = "；".join([*planning, *[issue["message"] for issue in rejected]])
            checked["coverage"].update(complete=False, source_complete=False)
    checked["protocol_issues"] = issues
    rows = checked.get("rows", [])
    checked["coverage"]["answered"] = sum(row["status"] == "answered" for row in rows)
    checked["observations"] = [observation for row in rows for observation in row["observations"]]
    if mode == "verify":
        verdicts = {row["verdict"] for row in rows}
        checked["verdict"] = ("contradicted" if "contradicted" in verdicts else "supported" if verdicts == {"supported"}
                              else "partial" if "supported" in verdicts else "insufficient")
    checked["answer"] = checked["explanation"] = "\n\n".join(_render_row(row, mode) for row in rows)


def table_cells(evidence: list[dict], fragments: dict) -> dict:
    """Expose existing explicit rectangular tables; never recover guessed cells.

    Unlike metric extraction, a column can be a benchmark. Its literal column
    label remains a separate axis and is not guessed to be an accuracy metric.
    """
    cells = {}
    for source in evidence:
        if source.get("source_type") != "table":
            continue
        lines = [line for line in source["text"].splitlines() if "|" in line]
        if len(lines) < 3:
            continue
        def parts(line):
            return [value.strip() for value in line.strip().strip("|").split("|")]
        divider = next((index for index, line in enumerate(lines)
                        if all(re.fullmatch(r":?-{2,}:?", value.replace(" ", "")) for value in parts(line))), None)
        if divider is None or divider == 0:
            continue
        header, header_line = parts(lines[divider - 1]), lines[divider - 1]
        for line in lines[divider + 1:]:
            row = parts(line)
            if len(row) != len(header) or len(row) < 2 or not row[0] or re.fullmatch(NUMBER_PATTERN, row[0]):
                continue
            for column, text in enumerate(row[1:], 1):
                # Keep the entire original cell; the contract checks the value
                # against this cell, never another number elsewhere in the row.
                time_annotation = re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*%?\s*\(\s*("
                                               + NUMBER_PATTERN + r")\s*(ms|s|sec)\s*\)\s*", text, re.I)
                if not time_annotation and not re.fullmatch(r"\s*" + NUMBER_PATTERN + r"\s*%?(?:\s*(?:±|\+/-)\s*" + NUMBER_PATTERN + r"\s*%?)?\s*", text):
                    continue
                row_handles = [handle for handle, binding in fragments.items()
                               if binding["source"]["id"] == source["id"] and line in binding["quote"]]
                header_handles = [handle for handle, binding in fragments.items()
                                  if binding["source"]["id"] == source["id"] and header_line in binding["quote"]]
                if not row_handles or not header_handles:
                    continue
                identifier = f"T{len(cells) + 1}"
                cells[identifier] = {
                    "cell_id": identifier, "source_id": source["id"], "paper_id": source["paper_id"],
                    "row_label": row[0], "column_label": header[column], "column_index": column,
                    "value_cell_text": text, "row_quote": line, "header_quote": header_line,
                    "column_labels": header[1:],
                    "row_handles": row_handles, "header_handles": header_handles,
                    "caption_handles": [handle for handle, binding in fragments.items()
                                        if binding["source"]["id"] in source.get("caption_evidence_ids", [])],
                }
                if time_annotation:
                    cells[identifier]["value"] = time_annotation[1]
                    cells[identifier]["time_annotation"] = {"text": text[text.index("(") + 1:text.rindex(")")],
                                                               "value": time_annotation[2], "unit": time_annotation[3]}
                if len(cells) == 160:
                    return cells
    return cells


def _context(question: str, requirements: list[dict], evidence: list[dict]):
    """Balance retrieval over requested subitems instead of just the answer."""
    groups, diagnostics = [], []
    for requirement in requirements:
        expansion = " ".join(_FIELD_QUERY.get(field, field) for field in requirement["fields"])
        query = " ".join([question, requirement.get("subject") or "", expansion])
        selected, report = select_evidence_context(query, evidence, limit=14, max_chars=22000)
        groups.append(selected)
        diagnostics.append(report)
    chosen, used, characters = [], set(), 0
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index >= len(group):
                continue
            item = group[index]
            if item["id"] in used or characters + len(item["text"]) > 30000:
                continue
            chosen.append(item)
            used.add(item["id"])
            characters += len(item["text"])
    # An explicit parser-provided caption identity is stronger than adjacency.
    # Include it without modifying or synthesizing any quotation.
    caption_ids = {identifier for item in chosen for identifier in item.get("caption_evidence_ids", [])}
    for item in evidence:
        if item["id"] in caption_ids and item["id"] not in used and characters + len(item["text"]) <= 30000:
            chosen.append(item)
            used.add(item["id"])
            characters += len(item["text"])
    positions = {item["id"]: index for index, item in enumerate(evidence)}
    chosen.sort(key=lambda item: positions[item["id"]])
    return chosen, {"selected": len(chosen), "total": len(evidence), "characters": characters,
                    "papers_covered": len({item["paper_id"] for item in chosen}),
                    "papers_selected": len({item["paper_id"] for item in evidence}),
                    "coverage_limited": any(item["coverage_limited"] for item in diagnostics),
                    "task_retrievals": diagnostics}


async def run_research_task(question: str, evidence: list[dict], config: dict, mode: str) -> dict:
    from .providers import _chat_with_protocol_correction, _citation_fragments, _config, _trace
    from .task_contract import check_findings, semantic_pairs

    config = _config(config)
    started = perf_counter()
    preliminary, _ = select_evidence_context(question, evidence)
    if not preliminary:
        return {"answer": "所选文献中未检索到相关原文，未调用模型。", "explanation": "所选文献中未检索到相关原文，未调用模型。",
                "mode": "llm", "task_protocol": PROTOCOL, "task_rows": [], "citations": [],
                "verdict": "insufficient", "traces": [], "warnings": ["未拆解任务，不能据此声称全文没有相关结果。"],
                "coverage": {"planned": 0, "addressed": 0, "answered": 0, "missing": 0,
                             "unaddressed": 0, "complete": False, "source_complete": False}}
    requirements, plan_corrected = await _chat_with_protocol_correction(
        PLAN_INSTRUCTION, {"mode": mode, "question": question, "question_handles": [{"id": "Q1", "text": question}], "source_data": []}, config,
        lambda result: _bound_plan(result, question, mode))
    candidates, retrieval = _context(question, requirements, evidence)
    source_data, fragments = task_fragments(candidates)
    cells = table_cells(candidates, fragments)
    bindings = {f"M{index}": {"kind": "text", "handle": handle, "quote": item["quote"], "source_handles": [handle]}
                for index, (handle, item) in enumerate(fragments.items(), 1)
                if item["source"].get("source_type") != "table" and len(item["quote"]) <= 360}
    bindings.update({identifier: {"kind": "table_cell", "cell_id": identifier,
                                 "source_handles": list(dict.fromkeys(cell["row_handles"] + cell["header_handles"] + cell["caption_handles"]))}
                     for identifier, cell in cells.items()})
    # check_findings fills omitted tasks as insufficient, so an empty selected
    # context remains a visibly incomplete result rather than another paid call.
    payload = {"mode": mode, "question": question, "requirements": requirements,
               "citation_protocol": "server_bound_fragments_v1", "source_data": source_data,
               "table_cells": list(cells.values()),
               "relation_bindings": [{"id": identifier, "source_handle": binding.get("handle"),
                                       "cell_id": binding.get("cell_id")} for identifier, binding in bindings.items()]}
    if candidates:
        def validate_findings(result):
            return partial_findings(result, requirements, candidates, fragments, cells, bindings)
        (raw_findings, checked, requirements), findings_corrected = await _chat_with_protocol_correction(
            FINDINGS_INSTRUCTION, payload, config,
            validate_findings)
    else:
        raw_findings = {"items": []}
        checked, findings_corrected = check_findings(raw_findings, requirements, [], {}, cells={}), False
    pairs = semantic_pairs(checked) if mode == "verify" else []
    protocol_issues = checked.get("protocol_issues", [])
    alignment_corrected = False
    if pairs:
        checked, alignment_corrected = await _chat_with_protocol_correction(
            ALIGNMENT_INSTRUCTION, {"pairs": pairs, "source_data": []}, config,
            lambda result: check_findings(raw_findings, requirements, candidates, fragments,
                                          cells=cells, alignments=result))
    _annotate_incomplete(checked, requirements, protocol_issues)
    rows = checked.pop("rows", [])
    for row in rows:
        if mode == "qa":
            row.pop("verdict", None)
        comparisons = row.get("field_comparisons", {})
        if isinstance(comparisons, dict):
            row["field_comparisons"] = [{"field": field, "expected": comparison["expected"],
                                          "observed": comparison["observed"], "relation": comparison["status"]}
                                         for field, comparison in comparisons.items()]
            for item in row["field_comparisons"]:
                item.update({key: comparisons[item["field"]][key] for key in ("basis", "reason", "literal_status")
                             if key in comparisons[item["field"]]})
    return {**checked, "task_rows": rows, "task_protocol": PROTOCOL, "mode": "llm", "retrieval": retrieval,
            "traces": [_trace("任务拆分", f"先于原文分析建立 {len(requirements)} 项请求清单。", [], started),
                       _trace("逐项证据整理", f"检查原子观察与来源、所需字段及字段对照；没有采用模型自报标签。", checked.get("citations", []), started)],
            "warnings": ["任务清单由模型拆分，请核对是否覆盖你的全部要求；逐项回应不等于所有信息已找到。",
                         "精确引文和行列绑定不证明科学真实性；上下文归属与任务拆分仍可能有误。",
                         "未找到字段只表示本次所选原文上下文未找到，不能断言论文从未报告。"]
                        + (["相关来源超过当前上下文范围，结果不代表全文完整召回。"] if retrieval["coverage_limited"] else [])
                        + (["部分中英文条件由单独模型作窄语义对照；已标明模型映射，仍需人工复核。"] if pairs else [])
                        + (["部分原始观察未通过引用或结构检查，已逐条暂缓；有效观察保留，不把暂缓算作任务成功。"] if protocol_issues else [])
                        + (["本次任务结构或引用经历一次受限纠错。"] if plan_corrected or findings_corrected or alignment_corrected else [])}
