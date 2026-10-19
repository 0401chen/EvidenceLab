"""Request-local research tasks and source-bound observations.

This module compares explicit records; it is not a general entailment engine.
Unresolved attribution stays unknown and never becomes an automatic rejection
of a scientific claim. It does not write research records or call a provider.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

from .errors import ProviderError
from .grounding import NUMBER_PATTERN, NUMBER_TOKEN, compact_text, decimal_number, numeric_text, readable_text


FIELDS = frozenset({"subject", "metric", "value", "unit", "dataset", "split", "protocol",
                    "hardware", "batch_size", "sample_size", "repeats", "uncertainty",
                    "task", "training", "other"})
LABELS = {"subject": "模型/对象", "metric": "指标", "value": "数值", "unit": "单位",
          "dataset": "数据集", "split": "评价划分", "protocol": "评价协议", "hardware": "硬件",
          "batch_size": "批处理", "sample_size": "样本量", "repeats": "重复次数",
          "uncertainty": "不确定性", "task": "任务", "training": "训练条件", "other": "其他要求"}
_MAX_FIELDS = 24
_MAX_REQUIREMENTS = 12
_MAX_OBSERVATIONS = 8
_SEMANTIC_FIELDS = frozenset({"metric", "split", "protocol", "task", "training", "uncertainty", "other"})
_UNKNOWN_MARKERS = frozenset({"未找到", "未报告", "未知", "不详", "无法确认", "不能确认", "未能确认",
                              "unknown", "notfound", "notreported", "notstated", "notavailable",
                              "notmentioned", "n/a", "na", "insufficientevidence"})
_ALIASES = {
    "metric": {"准确率": "accuracy", "正确率": "accuracy", "acc": "accuracy", "acc.": "accuracy",
               "错误率": "error", "误差率": "error", "errorrate": "error", "errors": "error",
               "err": "error", "err.": "error", "framespersecond": "fps", "帧每秒": "fps",
               "每秒帧数": "fps", "图像处理速度": "fps", "处理速度": "fps"},
    "split": {"验证集": "validation", "validationset": "validation", "测试集": "test",
              "testset": "test", "训练集": "train", "trainingset": "train"},
    "unit": {"％": "%", "percent": "%", "percentage": "%", "百分比": "%",
             "framespersecond": "fps", "帧/秒": "fps", "帧每秒": "fps", "每秒帧": "fps",
             "milliseconds": "ms", "millisecond": "ms", "毫秒": "ms",
             "seconds": "s", "second": "s", "秒": "s"},
    "protocol": {"单裁剪": "singlecrop", "single-crop": "singlecrop", "singlecrop": "singlecrop",
                 "十裁剪": "10crop", "10裁剪": "10crop", "ten-crop": "10crop",
                 "tencrop": "10crop", "10-crop": "10crop"},
    "batch_size": {"无批处理": "nobatchprocessing", "不使用批处理": "nobatchprocessing",
                   "nobatching": "nobatchprocessing", "withoutbatchprocessing": "nobatchprocessing"},
}


def _error(message: str, code: str = "output_format"):
    raise ProviderError(code, message)


def _required_text(value, message: str, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _error(message)
    return value.strip()


def _valid_field(field) -> bool:
    return isinstance(field, str) and len(field) <= 48 and bool(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", field))


def _significant(char: str) -> bool:
    return not char.isspace() and not unicodedata.category(char).startswith("P")


def validate_plan(result: dict, question: str, mode: str) -> list[dict]:
    """Validate a question-only plan, including literal coverage of the request."""
    if mode not in {"qa", "verify"}:
        _error("任务模式无效。")
    if not isinstance(question, str) or not question.strip():
        _error("任务问题不能为空。")
    if not isinstance(result, dict) or not isinstance(result.get("requirements"), list):
        _error("模型未返回requirements任务清单。")
    proposed = result["requirements"]
    if not 1 <= len(proposed) <= _MAX_REQUIREMENTS:
        _error("任务清单必须包含1至12项。")
    requirements, seen, covered = [], set(), set()
    for item in proposed:
        if not isinstance(item, dict):
            _error("任务清单含无效结构。")
        rid = item.get("id")
        if (not isinstance(rid, str) or not re.fullmatch(r"R(?:[1-9]|1[0-2])", rid)
                or rid in seen):
            _error("任务ID必须为唯一的R1至R12。")
        seen.add(rid)
        request = _required_text(item.get("request"), "任务必须复制原问题连续文字。", 4000)
        if request not in question:
            _error("任务改写了原问题，不能可靠检查覆盖范围。")
        for match in re.finditer(re.escape(request), question):
            covered.update(range(match.start(), match.end()))
        subject = item.get("subject")
        if subject is not None:
            subject = _required_text(subject, "任务对象无效。", 180)
            if subject not in question:
                _error("任务对象必须来自原问题连续文字，不能补写或替换模型。")
        fields = item.get("fields")
        if (not isinstance(fields, list) or not fields or len(fields) > _MAX_FIELDS
                or any(not _valid_field(field) for field in fields)
                or len(fields) != len(set(fields))):
            _error("任务fields必须为1至24个唯一、有界的snake_case属性名。")
        expected = item.get("expected", {})
        if not isinstance(expected, dict) or any(field not in fields for field in expected):
            _error("任务expected必须对应所要求的字段。")
        if mode == "qa" and expected:
            _error("问答规划不能预设答案。")
        checked_expected = {}
        for field, value in expected.items():
            value = _required_text(value, "核验预期值必须是原命题的连续文字。", 180)
            if value not in question:
                _error("核验预期值改变了原命题，不能继续。")
            checked_expected[field] = value
        operator = item.get("operator", "eq")
        if mode == "qa" and (operator is None or isinstance(operator, str) and operator.strip().casefold() in {"", "null", "none", "n/a", "na", "unknown"}):
            operator = "eq"
        if not isinstance(operator, str) or operator not in {"eq", "lt", "le", "gt", "ge"}:
            _error("任务数值比较符无效。")
        requirements.append({"id": rid, "request": request, "subject": subject,
                             "fields": list(fields), "expected": checked_expected,
                             "operator": operator, "mode": mode})
    if any(_significant(char) and index not in covered for index, char in enumerate(question)):
        _error("任务清单遗漏了原问题文字或限定，不能开始证据分析。")
    return requirements


def _norm(value: str) -> str:
    return compact_text(value).casefold()


def _canonical(field: str, value: str) -> str:
    value = _norm(value)
    aliases = _ALIASES.get(field, {})
    if value in aliases:
        return aliases[value]
    if field == "metric":
        # Preserve top-k: top-1 accuracy must never equal top-5 accuracy.
        value = re.sub(r"top[-_ ]?(\d+)", r"top\1", value)
        value = re.sub(r"[\[(]%[\])]|%$", "", value)
        abbreviated = re.fullmatch(r"(top\d+)(acc\.?|err\.?)", value)
        if abbreviated:
            return abbreviated[1] + ("accuracy" if abbreviated[2].startswith("acc") else "error")
        for original in sorted(aliases, key=len, reverse=True):
            if original in {"acc", "err", "acc.", "err."}:
                continue
            value = value.replace(original, aliases[original])
    return value


def _literal(text: str, source: str) -> bool:
    pattern = r"\s*".join(re.escape(char) for char in compact_text(text))
    if not pattern:
        return False
    if text[:1].isascii() and text[:1].isalnum():
        pattern = r"(?<![A-Za-z0-9_])" + pattern
    if text[-1:].isascii() and text[-1:].isalnum():
        pattern += r"(?![A-Za-z0-9_])"
    return bool(re.search(pattern, readable_text(source), flags=re.I))


def _numbers(text: str) -> list[Decimal]:
    # Preserve the source; spacing only makes explicit compact units readable.
    text = re.sub(r"(?<=\d)(?=(?:fps|ms|BLEU|GB|MB|kHz|Hz)\b)", " ", numeric_text(text), flags=re.I)
    # A prose comma after a result is punctuation, unlike a thousands separator
    # or malformed numeric list such as 1,2,3. Do not hide that first result.
    text = re.sub(r"(?<=\d),(?=\s|[A-Za-z\u3400-\u9fff])", " ", text)
    return [number for match in NUMBER_TOKEN.finditer(text)
            if (number := decimal_number(match[1])) is not None]


def _number(value: str) -> Decimal | None:
    value = numeric_text(value).strip().rstrip("%％").strip()
    return decimal_number(value)


def _field_literal(field: str, text: str, source: str) -> bool:
    numeric_metadata = (field in {"sample_size", "repeats", "batch_size"} or field not in FIELDS) and _number(text) is not None
    if field == "value" or numeric_metadata:
        wanted = _number(text)
        return wanted is not None and wanted in _numbers(source)
    if _literal(text, source):
        return True
    canonical = _canonical(field, text)
    if field in _ALIASES:
        choices = [canonical, *[key for key, value in _ALIASES[field].items() if value == canonical]]
        return any(_literal(choice, source) for choice in choices)
    return False


def _resolve(references, evidence_by_id: dict, fragments: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(references, list) or not references or len(references) > 12:
        _error("每个观察字段必须有1至12个原文handle。", "grounding")
    refs, sources = [], []
    for reference in references:
        if (not isinstance(reference, dict) or set(reference) != {"handle"}
                or not isinstance(reference["handle"], str) or reference["handle"] not in fragments):
            _error("观察引用不存在或混入自行填写的引用，结果已拒绝。", "grounding")
        handle = reference["handle"]
        binding = fragments[handle]
        source = binding.get("source", {})
        source_id = source.get("id")
        quote = binding.get("quote")
        if (source_id not in evidence_by_id or not isinstance(quote, str) or not quote.strip()
                or quote not in evidence_by_id[source_id].get("text", "")):
            _error("服务器来源片段不在本次研究范围。", "grounding")
        source = evidence_by_id[source_id]
        restored = {"id": source_id, "paper_id": source["paper_id"], "page": source.get("page"),
                    "handle": handle, "quote": quote, "source_start": binding.get("start"),
                    "source_end": binding.get("end"), "protocol": "server_bound_fragment"}
        if restored not in refs:
            refs.append(restored)
        if not any(existing["id"] == source_id for existing in sources):
            sources.append(source)
    return refs, sources


def _text_binding(binding: dict, fields: dict, fragments: dict) -> list[str]:
    handle = binding.get("handle")
    quote = binding.get("quote")
    if (not isinstance(handle, str) or handle not in fragments or not isinstance(quote, str)
            or not quote.strip() or quote not in fragments[handle].get("quote", "")):
        _error("测量关系必须引用同一handle内连续原文，不能改写。", "grounding")
    if len(quote) > 360:
        return ["测量关系片段过长，不能把整段词语共现当作实验归属。"]
    subject, value, metric = (fields.get(name, {}).get("text") for name in ("subject", "value", "metric"))
    if not subject or not value or not metric:
        return ["数值观察缺少具体对象、指标或数值，无法建立测量关系。"]
    if (not _literal(subject, quote) or not _field_literal("value", value, quote)
            or not _field_literal("metric", metric, quote)):
        return ["关系原句没有同时建立对象、指标与该测量值的对应。"]
    if not any(ref["handle"] == handle for ref in fields["value"]["evidence"]):
        return ["数值字段引用与所选测量关系原句不一致。"]
    clauses = re.split(r"[;；。！？]|\.(?=\s+[A-Z])|\b(?:and|but|whereas|while)\b", readable_text(quote), flags=re.I)
    related = [clause for clause in clauses if _literal(subject, clause)
               and _field_literal("value", value, clause) and _field_literal("metric", metric, clause)]
    if not related:
        return ["对象与数值位于不同结果句，不能把段落共现当作绑定。"]
    wanted = _number(value)
    if wanted is None:
        return ["当前数值观察不是可精确比较的单一数值。"]
    for clause in related:
        # Only explicit, generic configuration syntax can exempt a number.
        # Never mask arbitrary model-chosen field text: an "other: 71.2" field
        # could otherwise erase a conflicting result and authenticate 76.4.
        masked = re.sub(r"\b(?:batch\s+size|hidden\s+(?:size|dimension)|embedding\s+(?:size|dimension)|"
                        r"(?:input|crop|image)\s+(?:size|resolution)|k|n)\s*(?:[:=]\s*|\s+)(?:"
                        + NUMBER_PATTERN + r")\b", " ", clause, flags=re.I)
        masked = re.sub(r"\b\d+[-\s]+(?:crops?|views?)\b", " ", masked, flags=re.I)
        if set(_numbers(masked)) <= {wanted}:
            return []
    return ["同一关系原句含多个未分离测量值，尚不能唯一绑定模型、指标与结果。"]


def _table_binding(binding: dict, fields: dict, cells: dict) -> list[str]:
    cell_id = binding.get("cell_id")
    if not isinstance(cell_id, str) or cell_id not in cells:
        _error("模型选择了不存在的表格单元格。", "grounding")
    cell = cells[cell_id]
    subject, value = (fields.get(name, {}).get("text") for name in ("subject", "value"))
    if not subject or not value or not fields.get("metric"):
        return ["表格数值缺少对象、指标或数值。"]
    if _norm(subject) != _norm(str(cell.get("row_label") or "")):
        return ["模型对象与所选表格行标签不一致。"]
    if _number(value) is None or _number(value) not in _numbers(str(cell.get("value_cell_text") or "")):
        return ["测量值不在所选原始单元格中。"]
    if not any(ref["id"] == cell.get("source_id") for ref in fields["value"]["evidence"]):
        return ["数值引用没有指向所选单元格所属原文。"]
    cell_text = numeric_text(str(cell.get("value_cell_text") or ""))
    uncertainty = re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*[%％]?\s*(?:±|\+\s*/\s*-)\s*("
                               + NUMBER_PATTERN + r")\s*[%％]?\s*", cell_text)
    timed = re.fullmatch(r"\s*(" + NUMBER_PATTERN + r")\s*%?\s*\(\s*(" + NUMBER_PATTERN
                        + r")\s*(ms|s|sec)\s*\)\s*", cell_text, re.I)
    if uncertainty:
        if _number(value) != decimal_number(uncertainty[1]):
            return ["带±的单元格只能将左侧中心测量作为结果，不能把右侧不确定性当结果。"]
        if fields.get("uncertainty") and _number(fields["uncertainty"]["text"]) != decimal_number(uncertainty[2]):
            return ["不确定性字段与原单元格±右侧数值不一致。"]
    elif timed:
        if _number(value) != decimal_number(timed[1]):
            return ["带显式时间注释的单元格只能把左侧中心值作为结果，不能把括号时间当成该指标值。"]
    elif len(set(_numbers(cell_text))) > 1:
        bound_value = _number(str(cell.get("value", "")))
        if (not cell.get("protocol") or bound_value is None or bound_value != _number(value)
                or not fields.get("protocol")):
            return ["原单元格有多个数值，缺少已建立的指标/协议映射。"]
    dataset = fields.get("dataset", {}).get("text")
    column_labels = cell.get("column_labels") or []
    dataset_axis = cell.get("column_axis") == "dataset" or (
        dataset is not None and isinstance(column_labels, list) and len(column_labels) > 1
        and _norm(dataset) in {_norm(str(label)) for label in column_labels})
    if dataset and dataset_axis and _norm(dataset) != _norm(str(cell.get("column_label") or "")):
        return ["数据集与所选表格结果列不一致，不能移接另一列的测量。"]
    column_metric = _canonical("metric", str(cell.get("column_label") or ""))
    bound_metric = cell.get("metric")
    if not bound_metric and re.fullmatch(r"(?:top\d+)?(?:accuracy|error|fps|bleu|map|ap)", column_metric):
        bound_metric = cell["column_label"]
    if bound_metric and _canonical("metric", fields["metric"]["text"]) != _canonical("metric", str(bound_metric)):
        return ["指标与所选表格列不一致。"]
    column_top = re.search(r"top[-_\s]?(\d+)", str(cell.get("column_label") or ""), re.I)
    observed_top = re.search(r"top[-_\s]?(\d+)", fields["metric"]["text"], re.I)
    if column_top and (not observed_top or column_top[1] != observed_top[1]):
        return ["top-k指标与所选表格列不一致。"]
    if cell.get("protocol") and fields.get("protocol") and _canonical("protocol", fields["protocol"]["text"]) != _canonical("protocol", str(cell["protocol"])):
        return ["实验协议与所选表格单元格的协议不一致。"]
    return []


def _observation(raw: dict, evidence_by_id: dict, fragments: dict, cells: dict) -> tuple[dict, list[dict]]:
    if (not isinstance(raw, dict) or not isinstance(raw.get("fields"), dict)
            or not raw["fields"] or len(raw["fields"]) > _MAX_FIELDS):
        _error("观察必须包含非空fields。")
    fields, sources, explicit_unknown, normalizations = {}, [], [], []
    for field, item in raw["fields"].items():
        if not _valid_field(field):
            _error("观察字段结构或名称无效。")
        unknown = item is None or (
            isinstance(item, dict) and set(item) <= {"text", "evidence"}
            and item.get("evidence", []) == [] and
            (item.get("text") is None or isinstance(item.get("text"), str)
             and _norm(item["text"]) in _UNKNOWN_MARKERS))
        if unknown:
            explicit_unknown.append(field)
            normalizations.append({"field": field, "original": item, "action": "explicit_unknown_as_missing"})
            continue
        if not isinstance(item, dict):
            _error("观察字段结构或名称无效。")
        text = _required_text(item.get("text"), "观察字段必须是有限长度的原文文字。")
        refs, original = _resolve(item.get("evidence"), evidence_by_id, fragments)
        if not any(_field_literal(field, text, ref["quote"]) for ref in refs):
            _error("观察字段文字没有出现在其引用的原文中。", "grounding")
        fields[field] = {"text": text, "evidence": refs}
        for source in original:
            if not any(existing["id"] == source["id"] for existing in sources):
                sources.append(source)
    reasons = []
    if len({source.get("paper_id") for source in sources}) > 1:
        reasons.append("同一观察的字段来自不同论文，不能拼接成一个实验。")
    binding = raw.get("binding")
    if "value" in fields:
        if not isinstance(binding, dict) or binding.get("kind") not in {"text", "table_cell"}:
            _error("数值观察必须选择text或table_cell关系。")
        if binding["kind"] == "text":
            reasons.extend(_text_binding(binding, fields, fragments))
        else:
            reasons.extend(_table_binding(binding, fields, cells))
    if not fields:
        reasons.append("观察仅包含明确未知占位字段，未作为事实观察展示。")
    return {"fields": fields, "binding": binding, "status": "withheld" if reasons else "accepted",
            "reasons": reasons, "explicit_unknown_fields": explicit_unknown,
            "normalization_audit": normalizations}, sources


def _compare(field: str, expected: str, observed: list[str], operator: str = "eq") -> dict:
    comparison = {"expected": expected, "observed": observed, "status": "unknown", "basis": "literal"}
    if not observed:
        return comparison
    numeric_metadata = (field in {"sample_size", "repeats", "batch_size"} or field not in FIELDS) and _number(expected) is not None
    if field == "value" or numeric_metadata:
        target = _number(expected)
        values = {_number(value) for value in observed}
        if target is None or None in values or len(values) != 1:
            return comparison
        value = next(iter(values))
        comparison_operator = operator if field == "value" else "eq"
        matches = {"eq": value == target, "lt": value < target, "le": value <= target,
                   "gt": value > target, "ge": value >= target}[comparison_operator]
        comparison["status"] = "match" if matches else "conflict"
        return comparison
    canonical = {_canonical(field, value) for value in observed}
    if len(canonical) == 1:
        comparison["status"] = "match" if _canonical(field, expected) in canonical else "conflict"
    return comparison


def _semantic_candidate(row: dict, field: str, comparison: dict) -> dict | None:
    if (field not in _SEMANTIC_FIELDS or not comparison.get("expected")
            or comparison.get("status") == "match" or comparison.get("basis", "literal") != "literal"):
        return None
    unique = {}
    for value in comparison.get("observed", []):
        if isinstance(value, str):
            unique.setdefault(_norm(value), value)
    if len(unique) != 1:
        return None
    expected, observed = comparison["expected"], next(iter(unique.values()))
    if field == "metric":
        wanted_metric, observed_metric = _canonical(field, expected), _canonical(field, observed)
        known_metric = r"(?:top\d+)?(?:accuracy|error|fps|bleu|map|ap)"
        if (re.fullmatch(known_metric, wanted_metric) and re.fullmatch(known_metric, observed_metric)
                and wanted_metric != observed_metric):
            return None
    if field == "split":
        wanted_split, observed_split = _canonical(field, expected), _canonical(field, observed)
        if wanted_split in {"train", "validation", "test"} and observed_split in {"train", "validation", "test"} and wanted_split != observed_split:
            return None
    # A semantic mapping cannot repair different numerical constraints. Include
    # digits inside identifiers so version numbers cannot disappear in prose.
    number_pattern = re.compile(NUMBER_PATTERN)
    def numbers(value):
        return {decimal_number(match[0]) for match in number_pattern.finditer(numeric_text(value))}
    if numbers(expected) != numbers(observed):
        return None
    identifier_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b|"
                                    r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")
    def identifiers(value):
        return {_norm(match[0]) for match in identifier_pattern.finditer(readable_text(value))}
    wanted_ids, source_ids = identifiers(expected), identifiers(observed)
    if wanted_ids and source_ids and wanted_ids != source_ids:
        return None
    quotations = []
    for observation in row.get("observations", []):
        item = observation.get("fields", {}).get(field)
        if observation.get("status") != "accepted" or not item or _norm(item["text"]) != _norm(observed):
            continue
        for reference in item["evidence"]:
            if reference not in quotations:
                quotations.append(reference)
    if not quotations:
        return None
    return {"id": f"{row['id']}:{field}", "requirement_id": row["id"], "field": field,
            "expected": expected, "observed": observed, "citation_quotes": quotations}


def semantic_pairs(checked: dict) -> list[dict]:
    """Fixed, source-backed language pairs eligible for a narrow model mapping.

    Identity, datasets, measurements, units and hardware are never delegated.
    A returned pair is a task for a semantic model, not an accepted equivalence.
    """
    pairs = []
    for row in checked.get("rows", []):
        for field, comparison in row.get("field_comparisons", {}).items():
            candidate = _semantic_candidate(row, field, comparison)
            if candidate is not None:
                pairs.append(candidate)
    return pairs


def _comparison_verdict(requirement: dict, missing: list[str], comparisons: dict) -> str:
    if requirement.get("mode") != "verify":
        return "insufficient"
    if not requirement.get("expected"):
        # Merely finding the named model cannot certify an unparsed claim.
        return "insufficient"
    expected_fields = set(requirement.get("expected", {}))
    if requirement.get("subject"):
        expected_fields.add("subject")
    identity = [comparison["status"] for field, comparison in comparisons.items() if field != "value"]
    compared_required = set(requirement["fields"]) <= expected_fields
    measured_identity = "value" not in expected_fields or {"subject", "metric"} <= expected_fields
    if (missing or not expected_fields or not compared_required or not measured_identity
            or not all(state == "match" for state in identity)):
        return "insufficient"
    if all(comparison["status"] == "match" for comparison in comparisons.values()):
        return "supported"
    if comparisons.get("value", {}).get("status") == "conflict" and identity and {"subject", "metric"} <= expected_fields:
        return "contradicted"
    return "insufficient"


def _apply_alignments(alignments, rows: list[dict], requirements: list[dict]) -> None:
    if not isinstance(alignments, dict) or not isinstance(alignments.get("alignments"), list):
        _error("语义对照必须返回alignments数组。")
    allowed = {pair["id"]: pair for pair in semantic_pairs({"rows": rows})}
    if len(alignments["alignments"]) > len(allowed):
        _error("语义对照超过固定字段对范围。")
    checked, seen = {}, set()
    for item in alignments["alignments"]:
        if (not isinstance(item, dict) or set(item) != {"id", "relation", "reason"}
                or not isinstance(item.get("id"), str) or item["id"] not in allowed or item["id"] in seen
                or item.get("relation") not in {"match", "conflict", "unknown"}):
            _error("语义对照包含未知/重复字段对、改写内容或无效关系。")
        reason = _required_text(item.get("reason"), "语义对照缺少有界说明。", 500)
        seen.add(item["id"])
        checked[item["id"]] = {"status": item["relation"], "reason": reason}
    for row, requirement in zip(rows, requirements):
        for field, comparison in row["field_comparisons"].items():
            pair_id = f"{row['id']}:{field}"
            if pair_id not in allowed:
                continue
            mapped = checked.get(pair_id)
            if mapped is not None:
                comparison.update(literal_status=comparison["status"], basis="semantic_model", **mapped)
            else:
                # Omission does not retain an unjustified literal conflict as
                # a semantic conclusion. Both remain insufficient for support.
                comparison.update(literal_status=comparison["status"], status="unknown")
        row["verdict"] = _comparison_verdict(requirement, row["missing_fields"], row["field_comparisons"])


def _row(requirement: dict, item: dict | None, evidence_by_id: dict, fragments: dict, cells: dict) -> tuple[dict, list[dict]]:
    observations, sources = [], []
    explicit_missing, unplanned_missing, missing_audit = [], [], []
    if item is not None:
        raw_observations = item.get("observations")
        if not isinstance(raw_observations, list) or len(raw_observations) > _MAX_OBSERVATIONS:
            _error("每个任务必须返回有界observations数组。")
        supplied_missing = item.get("missing_fields", [])
        if (not isinstance(supplied_missing, list) or len(supplied_missing) > _MAX_FIELDS
                or any(not _valid_field(field) for field in supplied_missing)):
            _error("missing_fields必须为最多24个有界snake_case属性名。")
        for field in dict.fromkeys(supplied_missing):
            if field in requirement["fields"]:
                explicit_missing.append(field)
            else:
                unplanned_missing.append(field)
                missing_audit.append({"field": field, "original": field, "action": "unplanned_unknown_ignored_for_coverage"})
        for raw in raw_observations:
            checked, original = _observation(raw, evidence_by_id, fragments, cells)
            explicit_missing.extend(field for field in checked["explicit_unknown_fields"]
                                    if field in requirement["fields"] and field not in explicit_missing)
            unplanned_missing.extend(field for field in checked["explicit_unknown_fields"]
                                     if field not in requirement["fields"] and field not in unplanned_missing)
            subject = checked["fields"].get("subject", {}).get("text")
            if checked["fields"] and requirement.get("subject") and (not subject or _norm(subject) != _norm(requirement["subject"])):
                checked["status"] = "withheld"
                checked["reasons"].append("观察未明确对应本任务要求的具体对象，不能跨模型填充子项。")
            observations.append(checked)
            for source in original:
                if not any(existing["id"] == source["id"] for existing in sources):
                    sources.append(source)
    accepted = [observation for observation in observations if observation["status"] == "accepted"]
    requested = set(requirement["fields"])
    if "value" in requested:
        # A result and its protocol must belong to one observation. A second
        # record cannot fill a missing protocol on an unrelated measurement.
        measured = [observation for observation in accepted if "value" in observation["fields"]]
        best = max(measured or accepted, key=lambda observation: len(requested & set(observation["fields"])), default=None)
        present = set(best["fields"]) if best else set()
    else:
        groups = {}
        for observation in accepted:
            fields = observation["fields"]
            identities = tuple(_canonical(name, fields[name]["text"]) if name in fields else None
                               for name in ("subject", "dataset", "split"))
            papers = tuple(sorted({str(ref["paper_id"]) for field in fields.values() for ref in field["evidence"]}))
            groups.setdefault((identities, papers), set()).update(fields)
        present = max(groups.values(), key=lambda group: len(requested & group), default=set())
    missing = [field for field in requirement["fields"] if field not in present]
    status = "answered" if not missing else "partial" if present & set(requirement["fields"]) else "insufficient"
    expected = dict(requirement.get("expected", {}))
    if requirement.get("subject") and requirement.get("mode") == "verify":
        expected.setdefault("subject", requirement["subject"])
    comparisons = {field: _compare(field, value, [observation["fields"][field]["text"] for observation in accepted
                                                  if field in observation["fields"]], requirement.get("operator", "eq"))
                   for field, value in expected.items()}
    if requirement.get("mode") == "verify":
        for field in requirement["fields"]:
            if field not in comparisons:
                comparisons[field] = {"expected": None,
                                      "observed": [observation["fields"][field]["text"] for observation in accepted
                                                   if field in observation["fields"]],
                                      "status": "unknown", "basis": "literal"}
    verdict = _comparison_verdict(requirement, missing, comparisons)
    row = {key: requirement.get(key) for key in ("id", "request", "subject", "fields")}
    row.update(status=status, observations=observations, missing_fields=missing,
               explicit_missing_fields=explicit_missing, field_comparisons=comparisons,
               unplanned_missing_fields=unplanned_missing,
               normalization_audit=[*missing_audit, *[entry for observation in observations for entry in observation["normalization_audit"]]],
               verdict=verdict, addressed=item is not None,
               complete=item is not None and all(field in explicit_missing for field in missing))
    return row, sources


def _render_row(row: dict, mode: str) -> str:
    title = f"{row['id']} · {row.get('subject') or row['request']}"
    values = []
    field_order = ["subject", "task", "dataset", "split", "metric", "value", "unit", "protocol",
                   "hardware", "batch_size", "sample_size", "repeats", "uncertainty", "training", "other"]
    accepted = [observation for observation in row["observations"] if observation["status"] == "accepted"]
    for index, observation in enumerate(accepted, 1):
        values.append(f"观察{index}：")
        for field in [*field_order, *[name for name in observation["fields"] if name not in field_order]]:
            if field not in observation["fields"]:
                continue
            item = observation["fields"][field]
            citation = " ".join(f"[{ref['id']}]" for ref in item["evidence"])
            values.append(f"{LABELS.get(field, field)}：{item['text']} {citation}".strip())
    if not accepted:
        values.append("本任务未形成通过来源与归属检查的观察。")
    if row["missing_fields"]:
        values.append("本任务尚缺的对应来源：")
        values.extend(f"{LABELS.get(field, field)}：本次所检原文未能确认" for field in row["missing_fields"])
    if mode == "verify":
        verdict_label = {"supported": "支持", "contradicted": "反驳", "insufficient": "证据不足"}[row["verdict"]]
        title += f" · {verdict_label}"
        for field, comparison in row["field_comparisons"].items():
            state = {"match": "一致", "conflict": "不一致", "unknown": "未建立"}[comparison["status"]]
            requested = comparison["expected"] if comparison["expected"] is not None else "未建立可逐字对应的命题字段"
            values.append(f"{LABELS.get(field, field)}对照：{state}（命题要求：{requested}）")
            if comparison.get("basis") == "semantic_model":
                values.append(f"{LABELS.get(field, field)}语义映射（模型，需复核）：{comparison['reason']}")
        if row["verdict"] == "insufficient":
            values.append("身份、条件或证据关系尚未全部建立；不同实验条件不作为反驳依据。")
    if any(observation["status"] == "withheld" for observation in row["observations"]):
        values.append("部分观察的测量归属未通过检查，已暂缓，不作为结论依据。")
    return title + "\n" + "\n".join(values)


def check_findings(result: dict, requirements: list[dict], evidence: list[dict], fragments: dict,
                   cells: dict | list | None = None, alignments: dict | None = None) -> dict:
    """Resolve sources, preserve missing tasks and compute all visible conclusions.

    Model-provided verdict/explanation/answer keys are deliberately ignored.
    ``complete`` means all planned fields were addressed or explicitly unknown;
    ``source_complete`` means all required fields have accepted observations.
    Neither flag certifies scientific truth or question-planning completeness.
    """
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        _error("模型未返回items逐任务观察。")
    if len(result["items"]) > _MAX_REQUIREMENTS:
        _error("逐任务观察超过允许数量。")
    if not isinstance(requirements, list) or not requirements:
        _error("缺少已校验任务清单。")
    if isinstance(cells, list):
        cells = {str(cell.get("cell_id") or cell.get("id")): cell for cell in cells}
    cells = cells or {}
    evidence_by_id = {item["id"]: item for item in evidence}
    requirement_ids = {requirement["id"] for requirement in requirements}
    items = {}
    for item in result["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("requirement_id"), str):
            _error("观察缺少requirement_id。")
        rid = item["requirement_id"]
        if rid not in requirement_ids or rid in items:
            _error("观察返回了未知或重复任务ID。")
        items[rid] = item
    rows, citations = [], []
    for requirement in requirements:
        row, sources = _row(requirement, items.get(requirement["id"]), evidence_by_id, fragments, cells)
        rows.append(row)
        for source in sources:
            if not any(item["id"] == source["id"] for item in citations):
                citations.append(source)
    mode = requirements[0].get("mode", "qa")
    if alignments is not None:
        if mode != "verify":
            _error("问答任务不接受额外语义对照。")
        _apply_alignments(alignments, rows, requirements)
    observations = [observation for row in rows for observation in row["observations"]]
    quotations = []
    for observation in observations:
        for item in observation["fields"].values():
            for ref in item["evidence"]:
                if ref not in quotations:
                    quotations.append(ref)
    verdicts = {row["verdict"] for row in rows}
    if mode == "verify":
        verdict = ("contradicted" if "contradicted" in verdicts else "supported" if verdicts == {"supported"}
                   else "partial" if "supported" in verdicts else "insufficient")
    else:
        verdict = "unverified" if any(observation["status"] == "accepted" for observation in observations) else "insufficient"
    answer = "\n\n".join(_render_row(row, mode) for row in rows)
    answered = sum(row["status"] == "answered" for row in rows)
    addressed = sum(row["addressed"] for row in rows)
    return {"rows": rows, "observations": observations, "citations": citations, "citation_quotes": quotations,
            "normalization_audit": [entry for row in rows for entry in row["normalization_audit"]],
            "coverage": {"planned": len(rows), "addressed": addressed, "answered": answered,
                         "missing": sum(len(row["missing_fields"]) for row in rows),
                         "unaddressed": len(rows) - addressed,
                         "complete": all(row["complete"] for row in rows),
                         "source_complete": answered == len(rows)},
            "verdict": verdict, "answer": answer, "explanation": answer,
            "task_protocol": "requirements_observations_v1"}
